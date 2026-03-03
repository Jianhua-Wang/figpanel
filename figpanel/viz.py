"""Visualization utilities for detection results."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .detector import run_yolo

logger = logging.getLogger("figpanel")

ImagePath = Union[str, Path]

# Colors (RGB)
_SUBPLOT_COLOR = (66, 133, 244)  # Blue
_CAPTION_COLOR = (52, 168, 83)  # Green
_LABEL_COLOR = (234, 67, 53)  # Red


def visualize(
    image_path: ImagePath,
    save: Optional[Union[str, Path]] = None,
    *,
    conf: float = 0.25,
    iou: float = 0.5,
    line_width: int = 3,
) -> Image.Image:
    """Draw detection boxes on the image and optionally save.

    Subplots are drawn in blue, captions in green.

    Args:
        image_path: Path to the input image.
        save: If provided, save the annotated image to this path.
        conf: YOLO confidence threshold. Default 0.25.
        iou: YOLO NMS IoU threshold. Default 0.5.
        line_width: Width of bounding box lines. Default 3.

    Returns:
        Annotated PIL Image.
    """
    yolo_out = run_yolo(image_path, conf=conf, iou=iou)

    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    # Draw subplot boxes in blue
    for x1, y1, x2, y2, c in yolo_out["subplots"]:
        draw.rectangle([x1, y1, x2, y2], outline=_SUBPLOT_COLOR, width=line_width)
        draw.text((x1, y1 - 12), f"subplot {c:.2f}", fill=_SUBPLOT_COLOR)

    # Draw caption boxes in green
    for x1, y1, x2, y2, c in yolo_out["captions"]:
        draw.rectangle([x1, y1, x2, y2], outline=_CAPTION_COLOR, width=line_width)
        draw.text((x1, y1 - 12), f"caption {c:.2f}", fill=_CAPTION_COLOR)

    if save:
        save_path = Path(save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(save_path)
        logger.info("Saved annotated image to %s", save_path)

    return img
