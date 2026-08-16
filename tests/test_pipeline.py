"""
test_pipeline.py
-----------------
Lightweight unit / integration tests for the manuscript layout
detection pipeline. No test framework dependency beyond `pytest`
(already ubiquitous); run with:

    pytest tests/ -v

Covers:
  * preprocessing produces sane, correctly-shaped outputs
  * region proposals are valid (non-degenerate, inside the page)
  * the assignment's hard constraint -- bounding boxes must stay
    within page boundaries -- holds after classification
  * the classifier recovers all five target classes on a synthetic
    page built specifically to contain them
  * batch discovery + the CLI's JSON schema are correct
"""

import json
import os
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import classifier, preprocessing, region_proposal, utils  # noqa: E402
from tests.make_synthetic_page import make_page  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


@pytest.fixture(scope="module")
def synthetic_page(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("synthetic")
    path = os.path.join(out_dir, "synthetic.png")
    make_page(path, w=1200, h=1600, seed=42)
    return path


@pytest.fixture(scope="module")
def real_samples():
    d = os.path.join(ROOT, "data", "test_images")
    return [os.path.join(d, f) for f in os.listdir(d)
            if f.lower().endswith((".png", ".jpg")) and "synthetic" not in f]


# --------------------------------------------------------------------------
# preprocessing
# --------------------------------------------------------------------------

def test_preprocess_shapes(synthetic_page):
    result = preprocessing.preprocess(synthetic_page)
    assert result.gray.ndim == 2
    assert result.binary.ndim == 2
    assert result.binary.dtype == np.uint8
    assert set(np.unique(result.binary)).issubset({0, 255})
    assert result.page_bgr.shape[:2] == result.gray.shape


def test_preprocess_handles_all_real_samples(real_samples):
    assert real_samples, "expected at least one sample image in data/test_images"
    for path in real_samples:
        result = preprocessing.preprocess(path)
        assert result.binary.size > 0


def test_skew_correction_is_optional(synthetic_page):
    with_skew = preprocessing.preprocess(synthetic_page, correct_skew=True)
    without_skew = preprocessing.preprocess(synthetic_page, correct_skew=False)
    assert without_skew.skew_angle == 0.0
    assert with_skew.gray.shape == without_skew.gray.shape


def test_rotate_image_identity_for_zero_angle():
    img = np.zeros((10, 10), dtype=np.uint8)
    out = preprocessing.rotate_image(img, 0.0)
    assert np.array_equal(img, out)


# --------------------------------------------------------------------------
# region proposal
# --------------------------------------------------------------------------

def test_region_proposals_are_within_page(synthetic_page):
    pre = preprocessing.preprocess(synthetic_page)
    h, w = pre.binary.shape
    regions = region_proposal.propose_regions(pre.binary)
    assert len(regions) > 0
    for r in regions:
        assert r.x >= 0 and r.y >= 0
        assert r.x + r.w <= w
        assert r.y + r.h <= h
        assert r.w > 0 and r.h > 0


def test_region_proposals_nondegenerate_on_blank_page():
    blank = np.zeros((500, 400), dtype=np.uint8)  # no ink at all
    regions = region_proposal.propose_regions(blank)
    assert regions == []


def test_stroke_scale_reasonable(synthetic_page):
    pre = preprocessing.preprocess(synthetic_page)
    stroke = region_proposal.estimate_stroke_scale(pre.binary)
    assert 1 <= stroke <= 20


# --------------------------------------------------------------------------
# classifier
# --------------------------------------------------------------------------

def test_classifier_all_labels_have_valid_confidence(synthetic_page):
    pre = preprocessing.preprocess(synthetic_page)
    regions = region_proposal.propose_regions(pre.binary)
    cfg = utils.load_config(None)
    classified = classifier.classify_regions(regions, pre.binary.shape, cfg)
    assert len(classified) == len(regions)
    for c in classified:
        assert c.label in classifier.LABELS
        assert 0.0 <= c.confidence <= 1.0
        x1, y1, x2, y2 = c.bbox
        assert x2 > x1 and y2 > y1


def test_classifier_recovers_all_five_classes_on_synthetic_page(synthetic_page):
    """
    The synthetic fixture is constructed with an unambiguous header,
    footer, main_text, side_text, and several filler marks. A reasonable
    pipeline should recover all five labels somewhere in its output.
    """
    pre = preprocessing.preprocess(synthetic_page)
    regions = region_proposal.propose_regions(pre.binary)
    cfg = utils.load_config(None)
    classified = classifier.classify_regions(regions, pre.binary.shape, cfg)
    found_labels = {c.label for c in classified}
    missing = set(classifier.LABELS) - found_labels
    assert not missing, f"classifier failed to recover labels: {missing}"


def test_classifier_empty_input_returns_empty():
    cfg = utils.load_config(None)
    assert classifier.classify_regions([], (100, 100), cfg) == []


# --------------------------------------------------------------------------
# utils
# --------------------------------------------------------------------------

def test_discover_images_finds_all_samples(real_samples):
    found = utils.discover_images(os.path.join(ROOT, "data", "test_images"))
    assert len(found) >= len(real_samples)


def test_discover_images_single_file(real_samples):
    found = utils.discover_images(real_samples[0])
    assert found == [real_samples[0]]


def test_discover_images_missing_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        utils.discover_images(str(tmp_path / "does_not_exist"))


def test_config_defaults_present():
    cfg = utils.load_config(None)
    for key in ("header_band", "footer_band", "filler_max_area_ratio"):
        assert key in cfg


# --------------------------------------------------------------------------
# end-to-end CLI
# --------------------------------------------------------------------------

def test_cli_end_to_end(tmp_path, synthetic_page):
    out_dir = tmp_path / "results"
    cmd = [
        sys.executable, os.path.join(ROOT, "inference.py"),
        "--input", synthetic_page,
        "--output", str(out_dir),
    ]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr

    stem = os.path.splitext(os.path.basename(synthetic_page))[0]
    json_path = out_dir / "predictions" / f"{stem}.json"
    png_path = out_dir / "annotated" / f"{stem}.png"
    assert json_path.exists()
    assert png_path.exists()

    with open(json_path) as f:
        meta = json.load(f)
    assert meta["num_regions"] == len(meta["regions"])
    for r in meta["regions"]:
        assert r["label"] in classifier.LABELS
        assert 0 <= r["bbox"]["x1"] < r["bbox"]["x2"] <= meta["page_width"]
        assert 0 <= r["bbox"]["y1"] < r["bbox"]["y2"] <= meta["page_height"]


def test_cli_rejects_missing_input(tmp_path):
    cmd = [
        sys.executable, os.path.join(ROOT, "inference.py"),
        "--input", str(tmp_path / "nope"),
        "--output", str(tmp_path / "out"),
    ]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode != 0
