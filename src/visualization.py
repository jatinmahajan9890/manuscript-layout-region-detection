"""
visualization.py
-----------------
Draws the final classified regions onto a copy of the (preprocessed,
de-skewed) page image for human QA, and never touches the original
input file.
"""

from __future__ import annotations

import cv2
import numpy as np

COLORS = {
    "header": (66, 133, 244),      # blue
    "footer": (52, 168, 83),       # green
    "main_text": (219, 68, 55),    # red
    "side_text": (244, 160, 0),    # orange
    "filler": (155, 89, 182),      # purple
}


def draw_regions(image_bgr: np.ndarray, classified_regions: list, thickness: int = 2) -> np.ndarray:
    """Return a NEW annotated image; `image_bgr` is not modified in place."""
    out = image_bgr.copy()
    for reg in classified_regions:
        x1, y1, x2, y2 = reg.bbox
        color = COLORS.get(reg.label, (200, 200, 200))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)

        label_text = f"{reg.label} {reg.confidence:.2f}"
        (tw, th), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty1 = max(0, y1 - th - baseline - 4)
        cv2.rectangle(out, (x1, ty1), (x1 + tw + 4, ty1 + th + baseline + 4), color, -1)
        text_color = (255, 255, 255)
        cv2.putText(out, label_text, (x1 + 2, ty1 + th + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)
    return out


def draw_legend(image_bgr: np.ndarray) -> np.ndarray:
    """Overlay a small color legend in the top-right corner (used for demo figures)."""
    out = image_bgr.copy()
    x0, y0 = out.shape[1] - 170, 10
    for i, (label, color) in enumerate(COLORS.items()):
        y = y0 + i * 22
        cv2.rectangle(out, (x0, y), (x0 + 16, y + 16), color, -1)
        cv2.putText(out, label, (x0 + 22, y + 13), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(out, label, (x0 + 22, y + 13), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return out
