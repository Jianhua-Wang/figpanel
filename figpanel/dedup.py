"""Panel deduplication using perceptual hashing and ORB feature matching."""

from __future__ import annotations

import logging
from collections import OrderedDict, defaultdict

import numpy as np

logger = logging.getLogger("figpanel")


def _check_deps() -> None:
    """Raise ImportError if optional dependencies are missing."""
    try:
        import cv2  # noqa: F401
        import imagehash  # noqa: F401
    except ImportError:
        raise ImportError(
            "opencv-python and imagehash are required for deduplication. "
            "Install them with: pip install figpanel[full]"
        )


def compute_phash(image):
    """Compute perceptual hash of a PIL Image.

    Args:
        image: PIL Image.

    Returns:
        imagehash.ImageHash instance.
    """
    import imagehash

    return imagehash.phash(image)


def orb_similarity(img1, img2) -> float:
    """Calculate similarity between two images using ORB features.

    Args:
        img1: PIL Image.
        img2: PIL Image.

    Returns:
        Similarity score between 0 and 1.
    """
    import cv2

    arr1 = np.array(img1)
    arr2 = np.array(img2)

    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(arr1, None)
    kp2, des2 = orb.detectAndCompute(arr2, None)

    if des1 is None or des2 is None:
        return 0.0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    return len(matches) / max(len(des1), len(des2))


def deduplicate(pairs: list[dict], thresh: float = 0.3) -> list[dict]:
    """Remove duplicate panels based on image similarity.

    Uses perceptual hashing for bucketing and ORB feature matching for
    fine-grained similarity comparison.

    Args:
        pairs: List of panel dicts with ``"crop"`` (PIL Image) and ``"phash"``.
        thresh: ORB similarity threshold for considering duplicates. Default 0.3.

    Returns:
        Deduplicated list of panel dicts.
    """
    _check_deps()

    logger.info("Deduplicating %d panels (threshold=%.2f)", len(pairs), thresh)
    buckets: dict[int, list] = defaultdict(list)
    for p in pairs:
        buckets[int(str(p["phash"]), 16)].append(p)

    unique: OrderedDict[str, dict] = OrderedDict()
    for bucket in buckets.values():
        for p in bucket:
            if p["caption"] in unique:
                continue
            is_dup = False
            for q in unique.values():
                if orb_similarity(p["crop"], q["crop"]) >= thresh:
                    is_dup = True
                    break
            if not is_dup:
                unique[p["caption"]] = p

    logger.info("Found %d unique panels after deduplication", len(unique))
    return list(unique.values())
