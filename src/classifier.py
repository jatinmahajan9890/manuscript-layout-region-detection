"""
classifier.py
-------------
Assigns one of the five target labels to each candidate region proposed
by `region_proposal.py`:

    header, footer, main_text, side_text, filler

This is a transparent, rule-based / heuristic classifier (no training
data or API required), built on layout cues that are stable across
manuscript formats:

  * position on the page (top/bottom margin bands vs. body)
  * horizontal placement relative to the body's column span
    (far left/right -> side_text / marginalia)
  * size relative to the page and relative to the largest block
    (main_text is normally the dominant block(s))
  * ink density and shape regularity (very sparse / irregular /
    small blobs -> filler: decorations, stray pencil marks, isolated
    English annotations, etc.)
  * number of text lines contained (a single short line in the top
    band reads as a header/folio number; a tall multi-line block in
    the body reads as main_text or side_text depending on width)

Each rule contributes a weighted vote; the label with the highest
combined score wins, and that normalised score becomes the
`confidence` reported in the output JSON. Thresholds live in
`config.yaml` so they can be tuned per collection without touching code.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .region_proposal import Region

LABELS = ("header", "footer", "main_text", "side_text", "filler")


@dataclass
class ClassifiedRegion:
    label: str
    confidence: float
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2 (clipped to page)
    scores: dict  # per-label raw scores, for debugging / audit


def _band_of(y_center_ratio: float, cfg: dict) -> str:
    if y_center_ratio <= cfg["header_band"]:
        return "top"
    if y_center_ratio >= 1.0 - cfg["footer_band"]:
        return "bottom"
    return "body"


def classify_regions(regions: list[Region], page_shape: tuple[int, int], cfg: dict) -> list[ClassifiedRegion]:
    if not regions:
        return []

    h, w = page_shape
    page_area = float(h * w)

    # Body column span: use the union x-extent of WIDE, large blocks as a
    # proxy for where the main writing block(s) sit, so side margins can
    # be judged relative to the actual page content, not just raw x.
    # Restricting to "wide" blocks (not just "large" ones) prevents a
    # tall-but-narrow side column from folding itself into the body span
    # it's supposed to be judged against.
    large = [r for r in regions
             if r.area >= cfg["min_area_ratio_large"] * page_area
             and r.w >= cfg["body_min_width_ratio"] * w]
    if large:
        body_x0 = min(r.x for r in large)
        body_x1 = max(r.x + r.w for r in large)
    else:
        body_x0, body_x1 = 0, w
    body_width = max(1, body_x1 - body_x0)

    max_area = max(r.area for r in regions)

    results = []
    for r in regions:
        scores = {lab: 0.0 for lab in LABELS}

        y_center_ratio = (r.y + r.h / 2.0) / h
        x_center_ratio = (r.x + r.w / 2.0) / w
        area_ratio = r.area / page_area
        rel_area = r.area / max_area
        avg_line_h = float(np.mean(r.extra.get("line_heights", [r.h])))
        n_lines = r.n_components
        density = r.density

        band = _band_of(y_center_ratio, cfg)

        # An "isolated mark" is small in BOTH dimensions -- too small to be
        # a real running header/footer line (which spans multiple glyphs
        # over a meaningful width) or a marginal annotation column (which
        # runs tall along the margin). Structured labels require the
        # region to look, well, structured; single small blobs should
        # default toward filler regardless of which page band they land in.
        isolated_mark = (r.w < cfg["isolated_mark_max_width_ratio"] * w and
                          r.h < cfg["isolated_mark_max_height_ratio"] * h)

        # ---- main_text ------------------------------------------------
        # Large, multi-line, centered-ish over the body column span.
        overlap = max(0, min(r.x + r.w, body_x1) - max(r.x, body_x0))
        body_overlap_ratio = overlap / max(1, r.w)
        scores["main_text"] += 3.0 * rel_area
        scores["main_text"] += 1.5 if n_lines >= cfg["main_text_min_lines"] else 0.0
        scores["main_text"] += 1.0 * body_overlap_ratio
        scores["main_text"] += 1.0 if band == "body" else -1.0
        scores["main_text"] -= 2.0 if r.w < cfg["body_min_width_ratio"] * w else 0.0

        # ---- header -----------------------------------------------------
        scores["header"] += 2.5 if band == "top" else -2.0
        scores["header"] += 1.0 if n_lines <= cfg["margin_max_lines"] else -0.5
        scores["header"] += 0.5 if r.h <= cfg["margin_max_height_ratio"] * h else -0.5
        scores["header"] += 0.5 * (1.0 - area_ratio)  # margins tend to be small
        scores["header"] -= 3.0 if isolated_mark else 0.0

        # ---- footer -------------------------------------------------
        scores["footer"] += 2.5 if band == "bottom" else -2.0
        scores["footer"] += 1.0 if n_lines <= cfg["margin_max_lines"] else -0.5
        scores["footer"] += 0.5 if r.h <= cfg["margin_max_height_ratio"] * h else -0.5
        scores["footer"] += 0.5 * (1.0 - area_ratio)
        scores["footer"] -= 3.0 if isolated_mark else 0.0

        # ---- side_text ------------------------------------------------
        # Sits left/right of the main body column, tall relative to width,
        # not part of the dominant block(s).
        is_left = (r.x + r.w) <= body_x0 + cfg["side_text_slack"] * body_width
        is_right = r.x >= body_x1 - cfg["side_text_slack"] * body_width
        outside_body = (is_left or is_right) and body_overlap_ratio < 0.5
        scores["side_text"] += 2.0 if outside_body else -1.5
        scores["side_text"] += 1.0 if band == "body" else 0.0
        scores["side_text"] += 0.5 if (r.h / max(1, r.w)) > 1.0 else 0.0
        scores["side_text"] -= 1.0 if rel_area > 0.5 else 0.0
        scores["side_text"] -= 2.5 if isolated_mark else 0.0

        # ---- filler -----------------------------------------------------
        # Small, sparse, irregular, or isolated marks that don't read as
        # structured text: decorative glyphs, stray pencil/pen marks,
        # single stray words far from any text band.
        scores["filler"] += 1.5 if area_ratio < cfg["filler_max_area_ratio"] else -1.0
        scores["filler"] += 1.0 if density < cfg["filler_max_density"] else -0.5
        scores["filler"] += 0.5 if n_lines == 1 and avg_line_h < cfg["filler_max_line_height_ratio"] * h else 0.0
        scores["filler"] += 0.75 if not outside_body and band == "body" and rel_area < 0.05 else 0.0
        scores["filler"] += 3.0 if isolated_mark else 0.0

        label = max(scores, key=scores.get)
        raw = np.array(list(scores.values()), dtype=np.float64)
        # softmax -> confidence in [0, 1], comparable across regions
        exp = np.exp(raw - raw.max())
        probs = exp / exp.sum()
        confidence = float(probs[LABELS.index(label)])

        results.append(ClassifiedRegion(
            label=label,
            confidence=round(confidence, 4),
            bbox=(r.x, r.y, r.x + r.w, r.y + r.h),
            scores={k: round(v, 3) for k, v in scores.items()},
        ))

    return results
