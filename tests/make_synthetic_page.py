"""
Generates a synthetic manuscript-like test page containing all five
target layout classes (header, footer, main_text, side_text, filler)
plus mild degradation (uneven lighting, blur, noise, a stain), so the
full pipeline can be smoke-tested end-to-end without needing more real
manuscript scans.

Not part of the shipped pipeline -- a test fixture generator only.
"""

import cv2
import numpy as np


def _put_scribble_line(img, x0, x1, y, thickness=2, jitter=3, seed=None):
    rng = np.random.default_rng(seed)
    n_glyphs = max(3, (x1 - x0) // 18)
    xs = np.linspace(x0, x1, n_glyphs).astype(int)
    for gx in xs:
        h = rng.integers(8, 16)
        w = rng.integers(6, 14)
        y0 = y - h // 2 + rng.integers(-jitter, jitter)
        cv2.ellipse(img, (gx, y0 + h // 2), (w // 2, h // 2),
                    angle=float(rng.integers(0, 180)), startAngle=0, endAngle=300,
                    color=0, thickness=thickness)


def make_page(path: str, w=1600, h=2200, seed=0):
    rng = np.random.default_rng(seed)
    page = np.full((h, w), 245, dtype=np.uint8)

    # --- header: running header + folio number ---------------------------
    _put_scribble_line(page, int(0.30 * w), int(0.70 * w), int(0.05 * h), thickness=2, seed=1)
    _put_scribble_line(page, int(0.86 * w), int(0.94 * w), int(0.05 * h), thickness=2, seed=2)  # folio no.

    # --- footer: catchword + page number ----------------------------------
    _put_scribble_line(page, int(0.10 * w), int(0.30 * w), int(0.95 * h), thickness=2, seed=3)
    _put_scribble_line(page, int(0.48 * w), int(0.52 * w), int(0.95 * h), thickness=2, seed=4)

    # --- main_text: dense multi-line body -----------------------------
    top, bottom = int(0.14 * h), int(0.88 * h)
    left, right = int(0.22 * w), int(0.80 * w)
    y = top
    while y < bottom:
        _put_scribble_line(page, left, right, y, thickness=2,
                            seed=int(rng.integers(0, 10_000)))
        y += 22

    # --- side_text: narrow marginal column, left margin ------------------
    y = int(0.20 * h)
    while y < int(0.75 * h):
        _put_scribble_line(page, int(0.02 * w), int(0.14 * w), y, thickness=1,
                            seed=int(rng.integers(0, 10_000)))
        y += 16

    # --- filler: a few isolated decorative marks / stray pencil marks ----
    for cx, cy in [(int(0.90 * w), int(0.45 * h)), (int(0.06 * w), int(0.88 * h)),
                   (int(0.55 * w), int(0.10 * h) + 40)]:
        cv2.circle(page, (cx, cy), 6, 0, 2)
        cv2.line(page, (cx - 10, cy + 10), (cx + 10, cy - 10), 0, 1)

    # --- degradation: uneven lighting, blur, noise, a stain --------------
    yy, xx = np.mgrid[0:h, 0:w]
    gradient = 20 * np.sin(xx / w * np.pi) * np.cos(yy / h * np.pi)
    page = np.clip(page.astype(np.float32) + gradient, 0, 255).astype(np.uint8)

    page = cv2.GaussianBlur(page, (3, 3), 0.6)
    noise = rng.normal(0, 4, page.shape).astype(np.float32)
    page = np.clip(page.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    stain_center = (int(0.65 * w), int(0.55 * h))
    overlay = page.copy()
    cv2.circle(overlay, stain_center, 90, 190, -1)
    page = cv2.addWeighted(overlay, 0.35, page, 0.65, 0)

    bgr = cv2.cvtColor(page, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(path, bgr)
    return path


if __name__ == "__main__":
    make_page("data/test_images/synthetic_degraded_sample.png")
    print("wrote data/test_images/synthetic_degraded_sample.png")
