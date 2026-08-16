"""
preprocessing.py
-----------------
Image preprocessing utilities that make degraded manuscript scans usable
for downstream layout analysis.

Everything here is deterministic, offline classical image processing
(OpenCV / NumPy / scikit-image) -- no network calls, no API keys, no
pretrained deep weights to download.

Pipeline stages implemented:
    1. Page/substrate localisation (crop away scanner background)
    2. Grayscale conversion
    3. Illumination normalisation (handles uneven lighting / bleed-through)
    4. Denoising (handles blur / sensor noise / stains)
    5. Contrast enhancement (CLAHE) for faded ink
    6. Adaptive (local) binarization -- robust to non-uniform lighting
    7. Skew estimation & correction

Each function operates on / returns a numpy array so stages can be
chained, tested, and swapped independently.
"""

from __future__ import annotations

import cv2
import numpy as np


def load_image(path: str) -> np.ndarray:
    """Load an image from disk in BGR color. Raises if unreadable."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def localize_page(bgr: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """
    Crop the manuscript/page/leaf out of a (usually darker, uniform)
    scanner background, e.g. the blue-grey backdrop behind palm-leaf
    photographs. Falls back to the full image if no confident crop
    is found.

    Returns
    -------
    cropped_bgr : np.ndarray
    bbox : (x, y, w, h) of the crop in the ORIGINAL image, so callers
           can map detected regions back if needed.
    """
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Otsu threshold to separate page (bright/parchment) from background.
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Background is usually the majority color touching the image border;
    # make sure `mask` marks the PAGE as 255 regardless of which side Otsu picked.
    border_pixels = np.concatenate(
        [mask[0, :], mask[-1, :], mask[:, 0], mask[:, -1]]
    )
    if np.mean(border_pixels) > 127:  # border is mostly white -> invert
        mask = cv2.bitwise_not(mask)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((15, 15), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return bgr, (0, 0, w, h)

    largest = max(contours, key=cv2.contourArea)
    area_ratio = cv2.contourArea(largest) / float(h * w)
    if area_ratio < 0.15:  # too small to trust -> don't crop
        return bgr, (0, 0, w, h)

    x, y, cw, ch = cv2.boundingRect(largest)

    # Guard against scans that are ALREADY just the page (no distinct
    # scanner backdrop, e.g. pre-cropped archive images or full-bleed
    # synthetic pages): in that case the "largest contour" is really
    # just the densest patch of ink, not a true page boundary, and
    # cropping to it would throw away real margins/header/footer.
    #
    # Test using MEDIANS (robust to sparse ink coverage) of a thin
    # band right at the true image border vs. the page's own interior
    # background tone. A genuine studio/scanner backdrop is uniform and
    # a different substrate color from the page; a page that already
    # fills the frame has border-band color statistically indistinguishable
    # from its own interior (both are "blank page", ink only covers a
    # small fraction of pixels either way).
    touches = sum([x <= 2, y <= 2, (x + cw) >= w - 2, (y + ch) >= h - 2])
    if touches >= 3:
        return bgr, (0, 0, w, h)

    band = max(3, int(0.015 * min(h, w)))
    border_band = np.concatenate([
        bgr[:band, :, :].reshape(-1, 3),
        bgr[-band:, :, :].reshape(-1, 3),
        bgr[:, :band, :].reshape(-1, 3),
        bgr[:, -band:, :].reshape(-1, 3),
    ]).astype(np.float32)

    inset = max(band * 3, int(0.05 * min(h, w)))
    y0, y1 = min(inset, h // 2 - 1), max(h - inset, h // 2 + 1)
    x0, x1 = min(inset, w // 2 - 1), max(w - inset, w // 2 + 1)
    interior = bgr[y0:y1, x0:x1, :].reshape(-1, 3).astype(np.float32)
    if interior.size == 0:
        interior = bgr.reshape(-1, 3).astype(np.float32)

    border_median = np.median(border_band, axis=0)
    interior_median = np.median(interior, axis=0)
    border_std = float(np.mean(np.std(border_band, axis=0)))
    color_distance = float(np.linalg.norm(border_median - interior_median))

    is_uniform_backdrop = border_std < 18.0
    has_color_contrast = color_distance > 25.0
    if not (is_uniform_backdrop and has_color_contrast):
        return bgr, (0, 0, w, h)
    # small safety padding
    pad = int(0.01 * max(cw, ch))
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(w, x + cw + pad), min(h, y + ch + pad)
    return bgr[y0:y1, x0:x1].copy(), (x0, y0, x1 - x0, y1 - y0)


def normalize_illumination(gray: np.ndarray) -> np.ndarray:
    """
    Remove large-scale uneven lighting / bleed-through gradients by
    dividing out an estimate of the background (large-kernel median blur).
    """
    bg = cv2.medianBlur(gray, 41)
    bg = np.where(bg == 0, 1, bg).astype(np.float32)
    norm = (gray.astype(np.float32) / bg) * 255.0
    norm = np.clip(norm, 0, 255).astype(np.uint8)
    return norm


def denoise(gray: np.ndarray) -> np.ndarray:
    """Edge-preserving denoise for stains / sensor noise / mild blur."""
    return cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)


def enhance_contrast(gray: np.ndarray) -> np.ndarray:
    """CLAHE local contrast enhancement -- helps recover faded ink."""
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    return clahe.apply(gray)


def binarize(gray: np.ndarray) -> np.ndarray:
    """
    Adaptive Gaussian threshold -> robust to uneven illumination that
    survives normalisation, plus a light morphological clean-up to
    remove salt-and-pepper speckle from stains without breaking strokes.
    Output: uint8 mask where 255 = ink, 0 = background.
    """
    bw = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
        blockSize=31, C=10,
    )
    bw = cv2.medianBlur(bw, 3)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return bw


def remove_ruling_lines(binary_ink: np.ndarray) -> np.ndarray:
    """
    Strip long straight horizontal/vertical ruling lines (page borders,
    column rules, scanner crop guides) from the ink mask.

    These are common on both paper and palm-leaf manuscripts, are not
    part of any of the five target classes, and -- left in -- they chain
    connected-components across the whole page (a page border touches
    header, body and footer text alike), which silently merges every
    region into one giant block. Detected via morphological opening
    with kernels much longer than a normal glyph/stroke, so genuine
    text (which is not a single straight run of ink) survives.
    """
    h, w = binary_ink.shape
    h_len = max(15, w // 12)
    v_len = max(15, h // 12)

    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_len))

    h_lines = cv2.morphologyEx(binary_ink, cv2.MORPH_OPEN, h_kernel)
    v_lines = cv2.morphologyEx(binary_ink, cv2.MORPH_OPEN, v_kernel)

    lines = cv2.bitwise_or(h_lines, v_lines)
    # slight dilation so anti-aliased edges of the rule are fully removed
    lines = cv2.dilate(lines, np.ones((3, 3), np.uint8))

    cleaned = cv2.bitwise_and(binary_ink, cv2.bitwise_not(lines))
    return cleaned


def estimate_skew_angle(binary_ink: np.ndarray) -> float:
    """
    Estimate page skew (in degrees) from the orientation of ink pixels
    using minAreaRect over the text mask. Returns 0 if too little ink
    to make a confident estimate.
    """
    coords = np.column_stack(np.where(binary_ink > 0))
    if coords.shape[0] < 50:
        return 0.0
    rect = cv2.minAreaRect(coords.astype(np.float32))
    angle = rect[-1]
    # cv2 returns angle in [-90, 0); normalise to a small rotation
    if angle < -45:
        angle = 90 + angle
    # Only trust small corrective rotations -- large angles usually mean
    # the mask is dominated by a non-text blob, not real page skew.
    if abs(angle) > 15:
        return 0.0
    return float(angle)


def rotate_image(img: np.ndarray, angle: float, border_value=None) -> np.ndarray:
    """Rotate `img` by `angle` degrees around its center, keeping full canvas."""
    if abs(angle) < 0.1:
        return img
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    if border_value is None:
        border_value = 255 if img.ndim == 2 else (255, 255, 255)
    return cv2.warpAffine(
        img, M, (w, h), flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT, borderValue=border_value,
    )


class PreprocessResult:
    """Container bundling every intermediate representation a downstream
    stage might need, so we only run the pipeline once per image."""

    __slots__ = ("original", "page_bgr", "page_offset", "gray", "binary", "skew_angle")

    def __init__(self, original, page_bgr, page_offset, gray, binary, skew_angle):
        self.original = original
        self.page_bgr = page_bgr
        self.page_offset = page_offset
        self.gray = gray
        self.binary = binary
        self.skew_angle = skew_angle


def preprocess(path: str, correct_skew: bool = True) -> PreprocessResult:
    """Run the full preprocessing pipeline on an image path."""
    original = load_image(path)
    page_bgr, offset = localize_page(original)

    gray = cv2.cvtColor(page_bgr, cv2.COLOR_BGR2GRAY)
    gray = normalize_illumination(gray)
    gray = denoise(gray)
    gray = enhance_contrast(gray)

    binary = binarize(gray)
    binary = remove_ruling_lines(binary)
    angle = estimate_skew_angle(binary) if correct_skew else 0.0

    if abs(angle) >= 0.3:
        page_bgr = rotate_image(page_bgr, angle)
        gray = rotate_image(gray, angle)
        binary = rotate_image(binary, angle, border_value=0)

    return PreprocessResult(original, page_bgr, offset, gray, binary, angle)
