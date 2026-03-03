"""OCR for reading single-character panel labels (a, b, c, ...)."""

from __future__ import annotations

import logging

from PIL import Image

logger = logging.getLogger("figpanel")


def _check_tesseract() -> None:
    """Raise ImportError with install instructions if pytesseract is missing."""
    try:
        import pytesseract  # noqa: F401
    except ImportError:
        raise ImportError(
            "pytesseract is required for OCR. "
            "Install it with: pip install figpanel[full]\n"
            "You also need Tesseract installed on your system: "
            "https://tesseract-ocr.github.io/tessdoc/Installation.html"
        )


def ocr_single_char(image_crop: Image.Image) -> str | None:
    """Perform OCR on a cropped image to detect a single character.

    Args:
        image_crop: Cropped PIL image containing a single character.

    Returns:
        Detected character in lowercase, or None if OCR fails.
    """
    _check_tesseract()
    import pytesseract

    try:
        config = r"--psm 10 -c tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        text = pytesseract.image_to_string(image_crop, config=config)
        cleaned = text.strip()

        if len(cleaned) == 1 and cleaned.isalpha():
            return cleaned.lower()

        logger.debug("OCR result '%s' is not a single letter", cleaned)
        return None

    except pytesseract.TesseractNotFoundError:
        logger.error(
            "Tesseract is not installed or not in PATH. "
            "See https://tesseract-ocr.github.io/tessdoc/Installation.html"
        )
        return None
    except Exception as e:
        logger.error("OCR error: %s", e)
        return None
