---
tags:
  - object-detection
  - yolov12
  - scientific-figures
  - panel-detection
  - subplot-detection
  - computer-vision
library_name: ultralytics
license: agpl-3.0
datasets:
  - custom
pipeline_tag: object-detection
---

# figpanel-yolov12

**YOLOv12 model for detecting sub-panels and caption labels in scientific figures.**

Built by [Plottie](https://plottie.art) — the scientific plot discovery platform.

## Model Description

This model detects two classes of objects in composite scientific figures:

| Class | Description |
|-------|-------------|
| `subplot` | Individual plot/panel region within a composite figure |
| `caption` | Single-character label (a, b, c...) identifying each panel |

The model was trained on 5,000+ annotated figures from open-access scientific journals spanning biology, medicine, physics, and engineering.

| Property | Value |
|----------|-------|
| Architecture | YOLOv12 |
| Input | Any image (PNG, JPEG, etc.) |
| Model Size | ~18 MB |
| Classes | 2 (`subplot`, `caption`) |
| Framework | [Ultralytics](https://github.com/ultralytics/ultralytics) |

## Examples

| | | |
|:---:|:---:|:---:|
| ![](https://raw.githubusercontent.com/Jianhua-Wang/figpanel/main/examples/images/demo_genetics.png) | ![](https://raw.githubusercontent.com/Jianhua-Wang/figpanel/main/examples/images/demo_potency.png) | ![](https://raw.githubusercontent.com/Jianhua-Wang/figpanel/main/examples/images/demo_multiomics.png) |

> Blue boxes = detected subplots, Green boxes = detected captions. Images from open-access papers ([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)).

## How to Use

Install the Python package:

```bash
pip install figpanel
```

```python
import figpanel

# Detect subplots and captions
results = figpanel.detect("figure.png")
# {'subplots': [(x1,y1,x2,y2,conf), ...], 'captions': [...]}

# Visualize detections
figpanel.visualize("figure.png", save="annotated.png")
```

For the full pipeline (detect + OCR + match + crop):

```bash
pip install figpanel[full]
```

```python
panels = figpanel.extract("figure.png", "output/")
for panel in panels:
    print(f"Panel {panel.label}: confidence={panel.confidence:.2f}")
```

The model weights are downloaded automatically on first use via `huggingface_hub`.

## Training Data

The model was trained on a custom dataset of 5,000+ annotated composite figures collected from open-access scientific publications (CC BY 4.0). Annotations include bounding boxes for subplot regions and single-character caption labels. The dataset covers diverse figure layouts across multiple scientific disciplines.

## Limitations

- Optimized for composite figures with clearly separated panels; may underperform on continuous plots or single-panel figures
- Caption OCR expects single-character labels (a, b, c...); multi-character or numeric labels are not currently supported
- Performance may vary on figures with non-standard layouts (e.g., overlapping panels, inset plots)
- Trained primarily on figures from biology and methods journals; other domains may have lower accuracy

## About Plottie

[Plottie](https://plottie.art) is the scientific plot discovery platform. We help researchers explore, collect, and find inspiration from high-quality scientific plots across open-access literature.

## Citation

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

AGPL-3.0 — inherited from [Ultralytics](https://github.com/ultralytics/ultralytics). Academic and research use is unaffected.
