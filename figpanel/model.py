"""Model loading with automatic download from HuggingFace."""

import logging

from huggingface_hub import hf_hub_download
from ultralytics import YOLO

logger = logging.getLogger("figpanel")

_MODEL_REPO = "mermermer/figpanel-yolov12"
_MODEL_FILE = "v4.pt"

_model = None


def load_model(conf: float = 0.25, iou: float = 0.5) -> YOLO:
    """Load the figpanel YOLOv12 model, downloading from HuggingFace if needed.

    Args:
        conf: Confidence threshold for detections. Default 0.25.
        iou: IoU threshold for NMS. Default 0.5.

    Returns:
        Configured YOLO model instance.
    """
    global _model
    if _model is not None:
        return _model

    logger.info("Downloading model from HuggingFace: %s", _MODEL_REPO)
    path = hf_hub_download(repo_id=_MODEL_REPO, filename=_MODEL_FILE)

    model = YOLO(path)
    model.conf = conf  # type: ignore[attr-defined]
    model.iou = iou  # type: ignore[attr-defined]

    _model = model
    return model
