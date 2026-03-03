"""YOLO-based detection of subplots and captions in scientific figures."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import numpy as np

from .model import load_model

logger = logging.getLogger("figpanel")

ImagePath = Union[str, Path]


def run_yolo(image_path: ImagePath, conf: float = 0.25, iou: float = 0.5) -> dict:
    """Run YOLO detection on a scientific figure image.

    Args:
        image_path: Path to the input image.
        conf: Confidence threshold. Default 0.25.
        iou: NMS IoU threshold. Default 0.5.

    Returns:
        Dictionary with keys ``"subplots"`` and ``"captions"``, each containing
        a list of tuples ``(x1, y1, x2, y2, confidence)``.
    """
    model = load_model(conf=conf, iou=iou)

    logger.info("Running YOLO detection on %s", image_path)
    res = model(str(image_path), verbose=False)[0]

    boxes = res.boxes.xyxy.cpu().numpy()
    confs = res.boxes.conf.cpu().numpy()
    labels = res.boxes.cls.cpu().numpy().astype(int)

    out: dict[str, list] = {"subplots": [], "captions": []}
    for (x1, y1, x2, y2), c, lab in zip(boxes, confs, labels):
        entry = (int(x1), int(y1), int(x2), int(y2), float(c))
        # label 1 = subplot, label 0 = caption
        (out["subplots"] if lab == 1 else out["captions"]).append(entry)

    logger.info(
        "Detected %d subplots and %d captions",
        len(out["subplots"]),
        len(out["captions"]),
    )
    return out
