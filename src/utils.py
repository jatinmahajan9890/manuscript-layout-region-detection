"""
utils.py
--------
Small shared helpers: config loading, image discovery for batch mode,
and JSON metadata serialization.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import yaml

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

DEFAULT_CONFIG = {
    "header_band": 0.14,             # top fraction of page considered header band
    "footer_band": 0.14,             # bottom fraction of page considered footer band
    "margin_max_lines": 2,           # max lines typical of a header/footer block
    "margin_max_height_ratio": 0.10, # max block height (fraction of page h) for header/footer
    "min_area_ratio_large": 0.03,    # area ratio above which a block counts as "large" (defines body column)
    "body_min_width_ratio": 0.35,    # a block must span at least this fraction of page width to help define / count as the body column
    "main_text_min_lines": 3,        # min lines typical of a main_text block
    "side_text_slack": 0.06,         # extra fraction of body width allowed when testing side placement
    "filler_max_area_ratio": 0.015,  # area ratio below which a block leans "filler"
    "filler_max_density": 0.12,      # ink density below which a block leans "filler"
    "filler_max_line_height_ratio": 0.02,
    "isolated_mark_max_width_ratio": 0.035,   # marks smaller than this (both dims) can't be a real header/footer/side_text block
    "isolated_mark_max_height_ratio": 0.035,
    "min_region_area_ratio": 0.00008,  # drop proposals smaller than this (page-relative) as noise
}


def load_config(path: str | None) -> dict:
    cfg = dict(DEFAULT_CONFIG)
    if path and os.path.isfile(path):
        with open(path, "r") as f:
            user_cfg = yaml.safe_load(f) or {}
        cfg.update(user_cfg)
    return cfg


def discover_images(input_path: str) -> list[str]:
    """Return a sorted list of image file paths for a file or a directory."""
    p = Path(input_path)
    if p.is_file():
        return [str(p)] if p.suffix.lower() in IMAGE_EXTENSIONS else []
    if p.is_dir():
        return sorted(
            str(f) for f in p.rglob("*")
            if f.suffix.lower() in IMAGE_EXTENSIONS
        )
    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def regions_to_json(image_path: str, page_shape: tuple[int, int], skew_angle: float,
                     classified_regions: list, elapsed_s: float) -> dict:
    h, w = page_shape
    return {
        "image": os.path.basename(image_path),
        "source_path": str(image_path),
        "page_width": w,
        "page_height": h,
        "estimated_skew_deg": round(skew_angle, 3),
        "processing_time_s": round(elapsed_s, 3),
        "num_regions": len(classified_regions),
        "regions": [
            {
                "label": r.label,
                "confidence": r.confidence,
                "bbox": {
                    "x1": int(r.bbox[0]), "y1": int(r.bbox[1]),
                    "x2": int(r.bbox[2]), "y2": int(r.bbox[3]),
                },
                "scores": r.scores,
            }
            for r in classified_regions
        ],
    }


def save_json(data: dict, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)


def timer():
    return time.perf_counter()
