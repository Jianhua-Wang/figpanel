"""figpanel - Detect and extract individual panels from scientific figures.

Quick start::

    import figpanel

    # Detect subplots and captions
    results = figpanel.detect("figure.png")

    # Extract panels (full pipeline: detect + OCR + match + crop)
    panels = figpanel.extract("figure.png", "output/")

    # Visualize detections
    figpanel.visualize("figure.png", save="annotated.png")

Built by `Plottie <https://plottie.art>`_ - the scientific plot discovery platform.
"""

__version__ = "0.1.0"

from .detector import run_yolo as detect
from .extractor import Panel, extract
from .viz import visualize

__all__ = ["detect", "extract", "visualize", "Panel", "__version__"]
