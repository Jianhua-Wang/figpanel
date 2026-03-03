# figpanel

**Detect and extract individual panels from scientific figures using YOLOv12.**

[![PyPI](https://img.shields.io/pypi/v/figpanel?color=blue)](https://pypi.org/project/figpanel/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Model-yellow)](https://huggingface.co/plottie/figpanel-yolov12)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-green)](https://www.gnu.org/licenses/agpl-3.0)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Plottie/figpanel/blob/main/examples/demo.ipynb)
[![Plottie](https://img.shields.io/badge/Built%20by-Plottie-purple)](https://plottie.art)

Scientific papers often combine multiple plots into a single composite figure (Figure 1a, 1b, 1c...). **figpanel** automatically detects these sub-panels, reads their labels via OCR, and extracts each one as a separate image.

## Quick Start

```bash
pip install figpanel
```

```python
import figpanel

# Detect subplots and captions (bounding boxes + confidence scores)
results = figpanel.detect("figure.png")
# {'subplots': [(x1,y1,x2,y2,conf), ...], 'captions': [...]}

# Visualize detections on the image
figpanel.visualize("figure.png", save="annotated.png")
```

For the full pipeline (detect + OCR + match + crop + deduplicate):

```bash
pip install figpanel[full]
```

```python
# Extract individual panels
panels = figpanel.extract("figure.png", "output/")

for panel in panels:
    print(f"Panel {panel.label}: bbox={panel.bbox}, confidence={panel.confidence:.2f}")
    panel.image.show()
```

## Examples

| Genetics | Potency | Multi-omics |
|:--------:|:-------:|:-----------:|
| ![](https://raw.githubusercontent.com/Jianhua-Wang/figpanel/main/examples/images/demo_genetics.png) | ![](https://raw.githubusercontent.com/Jianhua-Wang/figpanel/main/examples/images/demo_potency.png) | ![](https://raw.githubusercontent.com/Jianhua-Wang/figpanel/main/examples/images/demo_multiomics.png) |

> Blue boxes = detected subplots, Green boxes = detected captions. Images from open-access papers ([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)).

## Features

- **YOLOv12 Detection** - Trained on 5,000+ annotated scientific figures to detect subplots and caption labels
- **Automatic Model Download** - Model weights are hosted on [HuggingFace](https://huggingface.co/plottie/figpanel-yolov12) and downloaded automatically on first use
- **OCR Caption Reading** - Reads single-character panel labels (a, b, c...) using Tesseract
- **Smart Matching** - Greedy nearest-neighbor algorithm matches captions to their corresponding subplots
- **Deduplication** - Removes duplicate panels using perceptual hashing (pHash) and ORB feature matching
- **Two-Tier Install** - Base install for detection only; `[full]` adds OCR, matching, and deduplication

## Model Details

| Property | Value |
|---|---|
| Architecture | YOLOv12 |
| Classes | `subplot` (panel region), `caption` (label character) |
| Training Data | 5,000+ annotated figures from open-access journals |
| Model Size | ~18 MB |
| Input | Any image (PNG, JPEG, etc.) |
| Hosted On | [HuggingFace](https://huggingface.co/plottie/figpanel-yolov12) |

## Installation Options

```bash
# Detection only (ultralytics + huggingface_hub)
pip install figpanel

# Full pipeline (+ pytesseract, imagehash, opencv)
pip install figpanel[full]
```

> **Note:** The `[full]` install requires [Tesseract OCR](https://tesseract-ocr.github.io/tessdoc/Installation.html) to be installed on your system.

## API Reference

### `figpanel.detect(image_path, conf=0.25, iou=0.5)`

Run YOLO detection on a scientific figure.

**Returns:** `dict` with `"subplots"` and `"captions"` keys, each containing a list of `(x1, y1, x2, y2, confidence)` tuples.

### `figpanel.extract(image_path, output_dir=None, *, conf=0.25, iou=0.5, padding=4, dedup_thresh=0.3, prefix=None)`

Full pipeline: detect, OCR, match, crop, and deduplicate panels.

**Returns:** `list[Panel]` where each `Panel` has `.label`, `.bbox`, `.confidence`, and `.image` attributes.

**Requires:** `pip install figpanel[full]`

### `figpanel.visualize(image_path, save=None, *, conf=0.25, iou=0.5, line_width=3)`

Draw detection boxes on the image. Subplots in blue, captions in green.

**Returns:** Annotated `PIL.Image`.

## Use Cases

- **Literature Mining** - Automatically decompose composite figures for indexing and search
- **Dataset Creation** - Build plot-type datasets from scientific papers at scale
- **Accessibility** - Extract individual panels for screen readers and alternative text
- **Research Tools** - Integrate panel detection into paper analysis pipelines
- **Quality Control** - Validate figure composition in manuscript preparation

## About Plottie

**figpanel** is built and maintained by [Plottie](https://plottie.art) - the scientific plot discovery platform. Plottie helps researchers explore, collect, and find inspiration from high-quality scientific plots across open-access literature.

Visit [plottie.art](https://plottie.art) to browse thousands of curated scientific figures.

## Citation

If you use figpanel in your research, please cite:

```bibtex
@software{figpanel,
  title = {figpanel: Scientific Figure Panel Detector},
  author = {Plottie},
  url = {https://github.com/Plottie/figpanel},
  version = {0.1.0},
  year = {2026},
  note = {Built by Plottie (https://plottie.art)}
}
```

## License

This project is licensed under the [AGPL-3.0](LICENSE) license. The YOLOv12 model weights inherit this license from [Ultralytics](https://github.com/ultralytics/ultralytics).

Academic and research use is unaffected. If you integrate figpanel into a network service, you must make your source code available under AGPL-3.0.
