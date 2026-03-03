"""Complete pipeline: detect, OCR, match, crop, deduplicate, and export panels."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from PIL import Image

from .detector import run_yolo
from .matcher import assign_captions_to_subplots
from .ocr import ocr_single_char

logger = logging.getLogger("figpanel")

ImagePath = Union[str, Path]


@dataclass
class Panel:
    """A detected panel from a scientific figure.

    Attributes:
        label: Panel label (e.g. ``"a"``, ``"b"``, ``"c"``).
        bbox: Bounding box as ``(x1, y1, x2, y2)``.
        confidence: Detection confidence score.
        image: Cropped panel as a PIL Image.
    """

    label: str
    bbox: tuple[int, int, int, int]
    confidence: float
    image: Image.Image


def _detect_and_read(
    image_path: ImagePath, conf: float = 0.25, iou: float = 0.5, padding: int = 4
) -> tuple[list[dict], dict]:
    """Detect captions and read them with OCR.

    Returns:
        Tuple of (ocr_results, yolo_output).
    """
    img = Image.open(image_path).convert("RGB")
    yolo_out = run_yolo(image_path, conf=conf, iou=iou)

    results = []
    for x1, y1, x2, y2, c in yolo_out["captions"]:
        x1p = max(0, x1 - padding)
        y1p = max(0, y1 - padding)
        x2p = min(img.width, x2 + padding)
        y2p = min(img.height, y2 + padding)
        crop = img.crop((x1p, y1p, x2p, y2p))

        letter = ocr_single_char(crop)
        if letter:
            results.append({"caption": letter, "bbox": (x1, y1, x2, y2, c)})

    logger.info("Read %d caption labels via OCR", len(results))
    return results, yolo_out


def _crop_panels(image_path: ImagePath, pairs: list[dict], pad: int = 4) -> list[dict]:
    """Crop merged bounding boxes from the original image."""
    img = Image.open(image_path).convert("RGB")

    for p in pairs:
        x1, y1, x2, y2 = p["merged_bbox"]
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(img.width, x2 + pad), min(img.height, y2 + pad)
        p["crop"] = img.crop((x1, y1, x2, y2))

    return pairs


def extract(
    image_path: ImagePath,
    output_dir: Optional[Union[str, Path]] = None,
    *,
    conf: float = 0.25,
    iou: float = 0.5,
    padding: int = 4,
    dedup_thresh: float = 0.3,
    prefix: Optional[str] = None,
) -> list[Panel]:
    """Extract individual panels from a scientific figure.

    This runs the full pipeline: YOLO detection, OCR caption reading,
    caption-to-subplot matching, cropping, and deduplication.

    Requires ``figpanel[full]`` for OCR and deduplication features.

    Args:
        image_path: Path to the input figure image.
        output_dir: If provided, save cropped panels as JPEG files here.
        conf: YOLO confidence threshold. Default 0.25.
        iou: YOLO NMS IoU threshold. Default 0.5.
        padding: Pixel padding around cropped panels. Default 4.
        dedup_thresh: ORB similarity threshold for deduplication. Default 0.3.
        prefix: Filename prefix for saved panels. Defaults to the image stem.

    Returns:
        List of :class:`Panel` objects.
    """
    from .dedup import compute_phash, deduplicate

    image_path = Path(image_path)
    logger.info("Extracting panels from %s", image_path)

    ocr_res, yolo_out = _detect_and_read(image_path, conf=conf, iou=iou, padding=padding)

    # Special case: single subplot with no captions
    if len(yolo_out["subplots"]) == 1 and len(yolo_out["captions"]) == 0:
        logger.info("Single subplot with no captions - treating whole figure as panel 'a'")
        img = Image.open(image_path).convert("RGB")
        sub = yolo_out["subplots"][0]
        panel = Panel(label="a", bbox=(sub[0], sub[1], sub[2], sub[3]), confidence=sub[4], image=img)

        if output_dir:
            out_dir = Path(output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            pfx = prefix or image_path.stem
            img.save(out_dir / f"{pfx}_a.jpg", "JPEG", optimize=True)

        return [panel]

    pairs = assign_captions_to_subplots(ocr_res, yolo_out["subplots"])
    pairs = _crop_panels(image_path, pairs, pad=padding)

    # Add phash for deduplication
    for p in pairs:
        p["phash"] = compute_phash(p["crop"])

    pairs = deduplicate(pairs, thresh=dedup_thresh)

    # Build Panel objects
    panels = []
    for p in pairs:
        sub = p["sub_bbox"]
        panel = Panel(
            label=p["caption"],
            bbox=p["merged_bbox"],
            confidence=sub[4],
            image=p["crop"],
        )
        panels.append(panel)

    # Save if output_dir provided
    if output_dir:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        pfx = prefix or image_path.stem
        for panel in panels:
            out_path = out_dir / f"{pfx}_{panel.label}.jpg"
            panel.image.save(out_path, "JPEG", optimize=True)
            logger.info("Saved panel to %s", out_path)

    logger.info("Extracted %d panels", len(panels))
    return panels
