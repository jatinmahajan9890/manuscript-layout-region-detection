#!/usr/bin/env python3
"""
inference.py
-------------
CLI entry point for the manuscript layout region detection pipeline.

Usage
-----
    python inference.py --input ./data/test_images --output ./results
    python inference.py --input ./data/test_images/page01.png --output ./results
    python inference.py --input ./data/test_images --output ./results \
        --config config.yaml --no-skew-correction --no-visualize

Behavior
--------
- `--input` may be a single image or a folder (recursed) -> batch mode.
- For every image `page.ext` it writes:
      <output>/annotated/page.png   (bounding boxes + labels + confidence)
      <output>/predictions/page.json
- The original input images and any pre-existing metadata are never
  modified -- everything is written under `--output`.
- All paths are resolved relative to the current working directory, so
  the exact CLI example from the assignment works unmodified from the
  repo root.
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback

import cv2

from src import classifier, preprocessing, region_proposal, utils, visualization


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect and classify manuscript layout regions "
                    "(header, footer, main_text, side_text, filler)."
    )
    parser.add_argument("--input", required=True,
                         help="Path to a single manuscript image or a folder of images.")
    parser.add_argument("--output", required=True,
                         help="Output folder for annotated images and JSON predictions.")
    parser.add_argument("--config", default="config.yaml",
                         help="Path to a YAML file overriding default thresholds (optional).")
    parser.add_argument("--no-skew-correction", action="store_true",
                         help="Disable automatic deskewing.")
    parser.add_argument("--no-visualize", action="store_true",
                         help="Skip writing annotated images (JSON predictions still written).")
    parser.add_argument("--legend", action="store_true",
                         help="Draw a color legend on annotated images.")
    return parser.parse_args()


def process_image(image_path: str, output_dir: str, cfg: dict,
                   correct_skew: bool, visualize: bool, legend: bool) -> dict | None:
    t0 = utils.timer()
    try:
        pre = preprocessing.preprocess(image_path, correct_skew=correct_skew)
    except Exception as exc:  # noqa: BLE001 - report and continue batch
        print(f"  [ERROR] preprocessing failed for {image_path}: {exc}", file=sys.stderr)
        traceback.print_exc()
        return None

    page_shape = pre.binary.shape  # (h, w)

    proposals = region_proposal.propose_regions(
        pre.binary, min_region_area_ratio=cfg["min_region_area_ratio"]
    )
    classified = classifier.classify_regions(proposals, page_shape, cfg)

    elapsed = utils.timer() - t0

    stem = os.path.splitext(os.path.basename(image_path))[0]

    if visualize:
        annotated = visualization.draw_regions(pre.page_bgr, classified)
        if legend:
            annotated = visualization.draw_legend(annotated)
        ann_dir = os.path.join(output_dir, "annotated")
        os.makedirs(ann_dir, exist_ok=True)
        cv2.imwrite(os.path.join(ann_dir, f"{stem}.png"), annotated)

    meta = utils.regions_to_json(image_path, page_shape, pre.skew_angle, classified, elapsed)
    pred_dir = os.path.join(output_dir, "predictions")
    os.makedirs(pred_dir, exist_ok=True)
    utils.save_json(meta, os.path.join(pred_dir, f"{stem}.json"))

    return meta


def main() -> int:
    args = parse_args()
    cfg = utils.load_config(args.config)

    try:
        images = utils.discover_images(args.input)
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    if not images:
        print(f"[ERROR] No supported images found under: {args.input}", file=sys.stderr)
        return 1

    os.makedirs(args.output, exist_ok=True)

    print(f"Found {len(images)} image(s). Writing results to: {args.output}\n")

    summary = {"processed": 0, "failed": 0, "total_regions": 0}
    for i, img_path in enumerate(images, 1):
        print(f"[{i}/{len(images)}] {img_path}")
        meta = process_image(
            img_path, args.output, cfg,
            correct_skew=not args.no_skew_correction,
            visualize=not args.no_visualize,
            legend=args.legend,
        )
        if meta is None:
            summary["failed"] += 1
            continue
        summary["processed"] += 1
        summary["total_regions"] += meta["num_regions"]
        counts = {}
        for r in meta["regions"]:
            counts[r["label"]] = counts.get(r["label"], 0) + 1
        print(f"    -> {meta['num_regions']} regions: {counts}")

    print("\n--- Summary ---")
    print(f"Processed: {summary['processed']}  Failed: {summary['failed']}  "
          f"Total regions detected: {summary['total_regions']}")
    return 0 if summary["failed"] == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
