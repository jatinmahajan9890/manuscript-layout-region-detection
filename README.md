# Manuscript Layout Region Detection

An end-to-end, fully offline Python pipeline that analyzes historical
manuscript images (palm-leaf and paper, any script) and automatically
detects and classifies five non-body/body layout region types:

| Class | Definition (per spec) |
|---|---|
| `header` | Top margin text, running headers, section titles, folio numbers |
| `footer` | Bottom margin text, catchwords, page numbers, signatures |
| `main_text` | The main body of the manuscript |
| `side_text` | Marginalia, annotations, or commentary along page margins |
| `filler` | Decorative elements, stray English text, pencil marks |

No paid/cloud API, API key, or internet access is used anywhere in this
pipeline. Everything runs locally with classical computer vision
(OpenCV / NumPy / scikit-image).

---

## 1. Why a classical CV approach (model rationale)

There is no labeled training set for this task (no ground-truth
bounding boxes for header/footer/main_text/side_text/filler across
manuscript collections), and the brief explicitly rules out any
API-based model. Training a supervised detector (YOLO/Detectron2/etc.)
from scratch without labels is not viable in a 2-day window, and would
not generalize better than a well-tuned classical pipeline on a *single*
new collection anyway.

Instead this project uses a **transparent, deterministic, unsupervised**
pipeline:

1. **Preprocessing** normalizes away scan artifacts (uneven lighting,
   noise, blur, skew) so the *content* — not the *acquisition
   condition* — drives every later decision.
2. **Region proposal** uses row/column ink-density projection profiles
   (not naive connected components) to find text lines and merge them
   into blocks. This is specifically chosen because dense historical
   scripts (e.g. Devanagari-family manuscripts) have ascenders/matras
   that visually touch the line above or below — plain connected
   components chain those into one giant blob and destroy the layout.
   Projection profiles are much more robust to this because a line
   boundary still shows up as a *relative* dip in ink density even when
   individual pixels are 8-connected.
3. **Classification** is a rule-based scorer over interpretable,
   script-agnostic layout cues: position on the page, size, aspect
   ratio, ink density, and how many text lines a block contains. Every
   rule and threshold lives in `config.yaml`, so a new manuscript
   collection with different margins/proportions can be retargeted by
   editing numbers, not code.

This design is:
- **Format-flexible** — no assumption about script, language, column
  count, or substrate (palm-leaf vs. paper); every threshold is
  page-relative or ink-stroke-relative, not a fixed pixel count.
- **Fully reproducible** — deterministic, no random initialization or
  GPU, same input always gives the same output.
- **Auditable** — the JSON output includes the *raw per-label scores*
  for every region, so a reviewer can see exactly why a block was
  classified as `header` vs `filler` rather than trusting a black box.
- **Extensible** — see [§7](#7-extending-this-project) for how to swap
  in a trained detector later without touching the CLI or output format.

---

## 2. Pipeline architecture

```
inference.py                     CLI entry point (batch or single image)
 └─ src/
     ├─ preprocessing.py         page crop, denoise, contrast, binarize, deskew
     ├─ region_proposal.py       ink mask -> line bands -> blocks (candidate regions)
     ├─ classifier.py            rule-based label + confidence per region
     ├─ visualization.py         draws annotated boxes/labels/scores
     └─ utils.py                 config, batch discovery, JSON export
config.yaml                      tunable thresholds (no code changes needed)
data/test_images/                sample manuscript pages (bundled)
results/                         default output folder (git-ignored)
tests/
 ├─ test_pipeline.py             pytest unit + integration tests
 └─ make_synthetic_page.py       generates a synthetic 5-class test page
```

### Preprocessing stages (`src/preprocessing.py`)
1. **Page localization** — crops the manuscript out of a photographed
   scanner backdrop (e.g. blue-grey background behind a palm-leaf
   photo), when one is detected via color-uniformity/contrast checks.
   Scans that are already page-only (no separate backdrop) are left
   untouched — this is verified, not assumed.
2. **Illumination normalization** — divides out a large-kernel median
   background estimate, correcting uneven lighting and mild
   bleed-through gradients.
3. **Denoising** — edge-preserving `fastNlMeansDenoising` for scan
   noise, stains, and mild blur.
4. **Contrast enhancement** — CLAHE local contrast, to recover faded
   ink.
5. **Binarization** — adaptive Gaussian thresholding (robust to
   residual uneven lighting) + light morphological clean-up.
6. **Ruling-line removal** — strips long straight horizontal/vertical
   lines (page borders, column rules, scanner crop guides). Left in,
   these silently chain every region on the page into one connected
   component.
7. **Skew estimation & correction** — `minAreaRect` over the ink mask;
   only trusts small corrective angles (≤15°) to avoid being fooled by
   a single non-text blob.

### Region proposal (`src/region_proposal.py`)
1. Row-wise ink-density projection profile → line "bands".
2. Column-wise projection *within* each band → splits a band into
   horizontal segments (separates a marginal column from the main text
   on the same row, or a multi-column layout).
3. Segments are grouped into blocks with a **pitch-aware greedy
   clustering** pass (a lightweight XY-cut): a segment joins the most
   recent block that is vertically close (gap ≤ the page's own median
   line pitch, scaled) *and* horizontally overlapping — i.e. same
   column. This is what turns individual lines into paragraphs/margins
   without also fusing unrelated blocks that happen to be near each
   other vertically.
4. Degenerate/noise-sized proposals are dropped; everything else is
   clipped to the page.

### Classification (`src/classifier.py`)
Each candidate block is scored against every one of the five labels
using weighted, interpretable rules (page-band position, size relative
to the page and to the largest block, horizontal placement relative to
the body column span, ink density, line count). The highest-scoring
label wins; a softmax over all five raw scores produces the reported
`confidence` (so confidence reflects how decisively one label beat the
others, not an arbitrary constant).

---

## 3. Setup

```bash
git clone <this-repo-url>
cd manuscript_layout_detection
python3 -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt
```

Requirements: Python 3.10+. All dependencies are open-source and
installed from PyPI — no API keys, no license keys, no network access
required at *run time* (only at `pip install` time).

---

## 4. Usage

### Single image
```bash
python inference.py --input ./data/test_images/palm_leaf_sample.png --output ./results
```

### Batch (folder, recursive)
```bash
python inference.py --input ./data/test_images --output ./results
```
This is the exact CLI shape specified in the assignment brief and works
unmodified from the repo root using relative paths.

### Options
```
--input               Path to a single image or a folder of images (required)
--output              Output folder for annotated images + JSON (required)
--config              Path to a YAML file overriding default thresholds (default: config.yaml)
--no-skew-correction  Disable automatic deskewing
--no-visualize        Skip writing annotated images (JSON predictions still written)
--legend              Draw a color legend on annotated images
```

### Output layout
```
results/
├─ annotated/
│   ├─ palm_leaf_sample.png     # original page + colored boxes + label + confidence
│   └─ paper_sample.png
└─ predictions/
    ├─ palm_leaf_sample.json
    └─ paper_sample.json
```

Example `predictions/*.json` record:
```json
{
  "image": "paper_sample.png",
  "page_width": 2047,
  "page_height": 774,
  "estimated_skew_deg": -1.06,
  "processing_time_s": 1.28,
  "num_regions": 9,
  "regions": [
    {
      "label": "header",
      "confidence": 0.9472,
      "bbox": {"x1": 590, "y1": 27, "x2": 1459, "y2": 50},
      "scores": {"header": 4.49, "footer": -0.01, "main_text": 0.08, "side_text": -1.5, "filler": -1.5}
    }
  ]
}
```

Original input images and any pre-existing metadata are **never
modified** — every output is written under `--output`.

---

## 5. Testing

```bash
pip install pytest
pytest tests/ -v
```

`tests/test_pipeline.py` covers:
- preprocessing output shapes/dtypes on both bundled real samples
- region proposals stay within page bounds and are non-degenerate
- the classifier recovers **all five** target labels on a synthetic
  page (`tests/make_synthetic_page.py`) built specifically to exercise
  header/footer/main_text/side_text/filler simultaneously, under
  injected blur/noise/uneven-lighting/a stain
- batch discovery and CLI JSON schema, end-to-end via subprocess

Regenerate the synthetic 5-class test page any time with:
```bash
python tests/make_synthetic_page.py
```

---

## 6. Robustness — how each degraded condition is handled

| Condition | Handling |
|---|---|
| Faded ink | CLAHE local contrast enhancement before binarization |
| Bleed-through | Illumination normalization (background division) suppresses low-frequency gradients from the reverse side |
| Stains | Edge-preserving denoise + adaptive (local, not global) threshold, so a stain's average brightness doesn't shift the whole page's threshold |
| Blur | `fastNlMeansDenoising` + adaptive threshold tolerate moderate blur; heavy blur will reduce line/column separation quality (see §8) |
| Skew | `minAreaRect`-based estimation + affine correction, applied before region proposal |
| Uneven lighting | Median-background division (step 2) + adaptive thresholding (step 5), which is local by construction |
| Page damage / holes | Handled implicitly — a hole simply produces no ink in that area, which the projection-profile approach tolerates far better than a single fragile global text-region contour |

---

## 7. Extending this project

The architecture is intentionally modular so a trained model can later
replace the heuristic stages **without changing the CLI, JSON schema,
or visualization code**:

- Swap `region_proposal.propose_regions()` for a trained detector's
  region proposals (e.g. a fine-tuned YOLO/Detectron2 model once
  labeled data exists) — it only needs to return `Region` objects.
- Swap `classifier.classify_regions()` for a trained classifier head
  over the same proposals — it only needs to return `ClassifiedRegion`
  objects with `label`, `confidence`, and `bbox`.
- `config.yaml` thresholds can be tuned per collection (e.g. a
  collection with unusually wide margins might need a larger
  `header_band`/`footer_band`).

---

## 8. Known limitations

- The classifier is heuristic/geometric, not semantic — it cannot
  distinguish a printed archival caption from a genuine manuscript
  header/footer purely by position when the two sit only a few pixels
  apart; on the bundled `paper_sample.png`, closely-spaced captions can
  be absorbed into `main_text`. This can be mitigated by widening
  `header_band`/`footer_band` per collection or narrowing the block
  merge tolerance in `region_proposal.cluster_segments`.
- Very heavy blur or extremely faint ink can reduce line-band
  separation quality, since the row/column projection profile depends
  on there being a genuine measurable dip in ink density between
  regions.
- Ruling-line removal assumes lines are close to horizontal/vertical;
  a heavily skewed photo (before correction) can leave line remnants
  that get picked up as thin, low-confidence `side_text`/`filler`
  proposals.

---

## 9. Constraints checklist (per assignment §5)

- [x] Bounding boxes are clipped to stay within page boundaries (`region_proposal.clip_to_page`, enforced again implicitly since detection only occurs inside the loaded page).
- [x] Modular, readable, reproducible (single-responsibility modules under `src/`, deterministic classical CV, no randomness in inference).
- [x] Original input images and metadata are never modified (all outputs written under `--output`).
- [x] Batch execution across multiple manuscript pages (`--input` accepts a folder, recursed).
