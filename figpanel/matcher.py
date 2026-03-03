"""Caption-to-subplot matching using greedy nearest-neighbor assignment."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger("figpanel")


def assign_captions_to_subplots(
    captions: list[dict],
    subplots: list[tuple],
    distance_k: float = 0.25,
) -> list[dict]:
    """Match captions to subplots using greedy one-to-one nearest-neighbor.

    Each caption is matched to the closest unmatched subplot, provided the
    distance is within ``distance_k * min(subplot_width, subplot_height)``.

    Args:
        captions: List of dicts with ``"caption"`` (str) and ``"bbox"`` (tuple).
        subplots: List of ``(x1, y1, x2, y2, confidence)`` tuples.
        distance_k: Distance threshold multiplier. Default 0.25.

    Returns:
        List of dicts with keys ``"caption"``, ``"cap_bbox"``, ``"sub_bbox"``,
        and ``"merged_bbox"``.
    """
    logger.info("Matching %d captions to %d subplots", len(captions), len(subplots))
    remaining = list(subplots)
    pairs = []

    for cap in captions:
        cx1, cy1, cx2, cy2, _ = cap["bbox"]
        best = None  # (idx, dist, sub_box)

        for idx, sub in enumerate(remaining):
            sx1, sy1, sx2, sy2, _ = sub

            # Caption inside subplot counts as distance 0
            inside = cx1 >= sx1 and cy1 >= sy1 and cx2 <= sx2 and cy2 <= sy2
            dist = 0.0 if inside else float(np.hypot(cx1 - sx1, cy1 - sy1))

            if best is None or dist < best[1]:
                best = (idx, dist, sub)

        if best is None:
            continue

        idx, dist, sub = best
        sx1, sy1, sx2, sy2, _ = sub
        thresh = distance_k * min(sx2 - sx1, sy2 - sy1)

        if dist <= thresh:
            merged = (min(cx1, sx1), min(cy1, sy1), max(cx2, sx2), max(cy2, sy2))
            pairs.append({
                "caption": cap["caption"],
                "cap_bbox": cap["bbox"],
                "sub_bbox": sub,
                "merged_bbox": merged,
            })
            remaining.pop(idx)

    logger.info("Matched %d caption-subplot pairs", len(pairs))
    return pairs
