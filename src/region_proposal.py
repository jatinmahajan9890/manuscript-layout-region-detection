"""
region_proposal.py
-------------------
Turns a binary ink mask into a set of candidate layout blocks (bounding
boxes), without yet deciding what label (header/footer/main_text/...)
each one gets -- that is `classifier.py`'s job.

Why projection profiles instead of plain connected components
----------------------------------------------------------------
Dense historical scripts (e.g. Devanagari-family manuscripts) very often
have ascenders/descenders/matras that visually *touch* the line above or
below. Running connected-component labelling straight on the ink mask
then chains many real text lines into one giant blob, which destroys
layout information. Projection profiles are far more robust to this: a
row of touching glyphs still shows a *relative* dip in ink density
compared to the row centers, which is enough to split lines even when
pixels are 8-connected.

Pipeline
--------
    1. Row (horizontal) projection profile -> split the page into
       line "bands" wherever ink density drops into a page-relative gap.
    2. Within each band, a column (vertical) projection profile splits
       it into horizontal segments -> separates side-columns / gutters
       from the main column on the same text line.
    3. Segments are grouped into blocks with a greedy, pitch-aware
       vertical + horizontal-overlap clustering pass (a lightweight
       XY-cut / RLSA-style merge) -> paragraphs / margins / columns.
    4. Degenerate / noise-sized blocks are dropped; everything left is
       clipped to the page.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class Region:
    """A candidate layout region prior to classification."""
    x: int
    y: int
    w: int
    h: int
    ink_pixels: int
    n_components: int
    extra: dict = field(default_factory=dict)

    @property
    def bbox(self):
        return (self.x, self.y, self.x + self.w, self.y + self.h)

    @property
    def area(self):
        return self.w * self.h

    @property
    def density(self):
        return self.ink_pixels / float(self.area) if self.area else 0.0


@dataclass
class _Segment:
    """One line-band x one column-split: a small rectangular strip of ink."""
    x0: int
    y0: int
    x1: int
    y1: int
    ink: int

    @property
    def w(self):
        return self.x1 - self.x0

    @property
    def h(self):
        return self.y1 - self.y0


def estimate_stroke_scale(binary_ink: np.ndarray) -> int:
    """Representative ink-stroke thickness in pixels, via distance transform."""
    if cv2.countNonZero(binary_ink) == 0:
        return 3
    dist = cv2.distanceTransform(binary_ink, cv2.DIST_L2, 5)
    vals = dist[binary_ink > 0]
    if vals.size == 0:
        return 3
    stroke = float(np.percentile(vals, 75)) * 2.0
    return int(np.clip(round(stroke), 2, 12))


def _smoothed_profile(counts: np.ndarray, window: int) -> np.ndarray:
    window = max(1, window)
    if window == 1:
        return counts.astype(np.float64)
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(counts.astype(np.float64), kernel, mode="same")


def _find_bands(profile: np.ndarray, min_gap: int, min_band: int) -> list[tuple[int, int]]:
    """
    Given a 1D smoothed density profile, return contiguous (start, end)
    index ranges that are "content" (above threshold), treating gap runs
    shorter than `min_gap` as still part of the surrounding content.
    """
    if profile.size == 0:
        return []
    threshold = max(1.0, 0.06 * profile.max())
    is_ink = profile > threshold

    n = len(is_ink)
    i = 0
    bridged = is_ink.copy()
    while i < n:
        if not is_ink[i]:
            j = i
            while j < n and not is_ink[j]:
                j += 1
            if (j - i) < min_gap:
                bridged[i:j] = True
            i = j
        else:
            i += 1

    bands = []
    i = 0
    while i < n:
        if bridged[i]:
            j = i
            while j < n and bridged[j]:
                j += 1
            if (j - i) >= min_band:
                bands.append((i, j))
            i = j
        else:
            i += 1
    return bands


def find_line_bands(binary_ink: np.ndarray, stroke: int) -> list[tuple[int, int]]:
    """Row-wise: split the page into horizontal line bands."""
    row_counts = np.count_nonzero(binary_ink, axis=1)
    profile = _smoothed_profile(row_counts, window=max(1, stroke // 2))
    min_gap = max(2, int(stroke * 0.9))
    min_band = max(2, int(stroke * 0.8))
    return _find_bands(profile, min_gap=min_gap, min_band=min_band)


def find_column_segments(binary_ink: np.ndarray, y0: int, y1: int, stroke: int,
                          page_w: int) -> list[tuple[int, int]]:
    """Column-wise, within one line band: split into horizontal segments
    (separates a side-margin gloss / a second column from the main text
    on the same row)."""
    band = binary_ink[y0:y1, :]
    col_counts = np.count_nonzero(band, axis=0)
    profile = _smoothed_profile(col_counts, window=max(1, stroke))
    min_gap = max(int(stroke * 5), int(0.012 * page_w))
    min_band = max(2, stroke)
    return _find_bands(profile, min_gap=min_gap, min_band=min_band)


def build_segments(binary_ink: np.ndarray) -> list[_Segment]:
    stroke = estimate_stroke_scale(binary_ink)
    h, w = binary_ink.shape
    bands = find_line_bands(binary_ink, stroke)

    segments: list[_Segment] = []
    for (y0, y1) in bands:
        cols = find_column_segments(binary_ink, y0, y1, stroke, w)
        for (x0, x1) in cols:
            ink = int(cv2.countNonZero(binary_ink[y0:y1, x0:x1]))
            if ink == 0:
                continue
            segments.append(_Segment(x0=x0, y0=y0, x1=x1, y1=y1, ink=ink))
    return segments


class _Block:
    __slots__ = ("segments", "x0", "y0", "x1", "y1")

    def __init__(self, seg: _Segment):
        self.segments = [seg]
        self.x0, self.y0, self.x1, self.y1 = seg.x0, seg.y0, seg.x1, seg.y1

    def last_x_range(self):
        s = self.segments[-1]
        return s.x0, s.x1

    def last_y1(self):
        return self.segments[-1].y1

    def add(self, seg: _Segment):
        self.segments.append(seg)
        self.x0 = min(self.x0, seg.x0)
        self.y0 = min(self.y0, seg.y0)
        self.x1 = max(self.x1, seg.x1)
        self.y1 = max(self.y1, seg.y1)


def _x_overlap_ratio(a0, a1, b0, b1) -> float:
    inter = max(0, min(a1, b1) - max(a0, b0))
    denom = min(a1 - a0, b1 - b0)
    return inter / denom if denom > 0 else 0.0


def cluster_segments(segments: list[_Segment], page_shape: tuple[int, int]) -> list[_Block]:
    """
    Greedy XY-cut-style clustering: walk segments top-to-bottom; attach
    each to the most recent compatible block that is (a) vertically close
    (gap <= the page's typical line pitch, scaled) and (b) horizontally
    overlapping (same column), else start a new block.
    """
    if not segments:
        return []
    h, w = page_shape
    segs = sorted(segments, key=lambda s: (s.y0, s.x0))

    y_starts = sorted(set(s.y0 for s in segs))
    if len(y_starts) >= 2:
        pitch = float(np.median(np.diff(y_starts)))
    else:
        pitch = 0.02 * h
    merge_gap = float(np.clip(pitch * 1.3, 4, 0.035 * h))

    blocks: list[_Block] = []
    for seg in segs:
        best = None
        best_gap = None
        for b in blocks:
            gap = seg.y0 - b.last_y1()
            if gap < -2 or gap > merge_gap:
                continue
            bx0, bx1 = b.last_x_range()
            overlap = _x_overlap_ratio(seg.x0, seg.x1, bx0, bx1)
            left_aligned = abs(seg.x0 - bx0) < 0.04 * w
            if overlap < 0.2 and not left_aligned:
                continue
            if best_gap is None or gap < best_gap:
                best, best_gap = b, gap
        if best is not None:
            best.add(seg)
        else:
            blocks.append(_Block(seg))
    return blocks


def clip_to_page(regions: list[Region], page_shape: tuple[int, int]) -> list[Region]:
    """Constraint from the assignment: bounding boxes must stay inside the page."""
    h, w = page_shape
    clipped = []
    for r in regions:
        x0, y0 = max(0, r.x), max(0, r.y)
        x1, y1 = min(w, r.x + r.w), min(h, r.y + r.h)
        if x1 <= x0 or y1 <= y0:
            continue
        r.x, r.y, r.w, r.h = x0, y0, x1 - x0, y1 - y0
        clipped.append(r)
    return clipped


def _count_distinct_lines(segments: list[_Segment]) -> int:
    return len(set(round(s.y0 / 4) for s in segments)) or 1


def find_residual_marks(binary_ink: np.ndarray, blocks: list[_Block],
                         min_abs_area: int) -> list["_Block"]:
    """
    The row/column projection-profile pass is tuned (via a relative
    density threshold) to find substantial text bands, so very sparse,
    isolated marks -- stray pen/pencil dots, small decorative glyphs --
    can fall below that threshold and never become a segment. This pass
    recovers them directly via connected components on whatever ink was
    NOT already claimed by an assigned block, so they still surface as
    small, low-density regions for the classifier (typically -> filler).
    """
    h, w = binary_ink.shape
    claimed = np.zeros((h, w), dtype=np.uint8)
    for b in blocks:
        cv2.rectangle(claimed, (b.x0, b.y0), (b.x1, b.y1), 255, -1)

    residual = cv2.bitwise_and(binary_ink, cv2.bitwise_not(claimed))
    # light dilation so a mark's strokes count as one connected blob
    residual = cv2.dilate(residual, np.ones((3, 3), np.uint8))

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(residual, connectivity=8)
    extra_blocks = []
    for i in range(1, n_labels):
        x, y, cw, ch, area = stats[i]
        if area < min_abs_area:
            continue
        ink = int(cv2.countNonZero(binary_ink[y:y + ch, x:x + cw]))
        if ink == 0:
            continue
        seg = _Segment(x0=int(x), y0=int(y), x1=int(x + cw), y1=int(y + ch), ink=ink)
        extra_blocks.append(_Block(seg))
    return extra_blocks


def propose_regions(binary_ink: np.ndarray, min_region_area_ratio: float = 0.00008) -> list[Region]:
    """Full proposal pipeline: ink mask -> line bands -> column segments -> blocks."""
    h, w = binary_ink.shape
    segments = build_segments(binary_ink)
    blocks = cluster_segments(segments, (h, w))
    n_primary_blocks = len(blocks)

    stroke = estimate_stroke_scale(binary_ink)
    min_abs_area = max(20, int((stroke * 2.5) ** 2))
    blocks += find_residual_marks(binary_ink, blocks, min_abs_area)

    regions = []
    for idx, b in enumerate(blocks):
        ink = sum(s.ink for s in b.segments)
        line_heights = [s.h for s in b.segments]
        n_lines = _count_distinct_lines(b.segments)
        is_residual_mark = idx >= n_primary_blocks
        r = Region(x=b.x0, y=b.y0, w=b.x1 - b.x0, h=b.y1 - b.y0,
                   ink_pixels=ink, n_components=n_lines,
                   extra={"line_heights": line_heights, "is_residual_mark": is_residual_mark})
        regions.append(r)

    regions = clip_to_page(regions, (h, w))
    min_area = min_region_area_ratio * h * w
    # Residual marks were already area-filtered (at a finer, stroke-relative
    # threshold) when detected -- don't re-filter them against the coarser
    # page-relative threshold meant for paragraph/margin blocks.
    regions = [r for r in regions if r.area >= min_area or r.extra.get("is_residual_mark")]
    regions.sort(key=lambda r: (r.y, r.x))
    return regions
