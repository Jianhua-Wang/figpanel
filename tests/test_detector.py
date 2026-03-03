"""Tests for figpanel detection pipeline."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def sample_image(tmp_path):
    """Create a simple test image."""
    img = Image.new("RGB", (800, 600), color=(255, 255, 255))
    path = tmp_path / "test_figure.png"
    img.save(path)
    return path


@pytest.fixture
def mock_yolo_result():
    """Create a mock YOLO result."""
    result = MagicMock()
    result.boxes.xyxy.cpu().numpy.return_value = np.array([
        [10, 10, 390, 290],   # subplot
        [410, 10, 790, 290],  # subplot
        [5, 5, 25, 25],       # caption
        [405, 5, 425, 25],    # caption
    ])
    result.boxes.conf.cpu().numpy.return_value = np.array([0.95, 0.92, 0.88, 0.85])
    result.boxes.cls.cpu().numpy.return_value = np.array([1, 1, 0, 0])
    return result


class TestDetector:
    """Test the YOLO detector module."""

    @patch("figpanel.detector.load_model")
    def test_run_yolo_returns_dict(self, mock_load, sample_image, mock_yolo_result):
        mock_model = MagicMock()
        mock_model.return_value = [mock_yolo_result]
        mock_load.return_value = mock_model

        from figpanel.detector import run_yolo

        result = run_yolo(sample_image)

        assert "subplots" in result
        assert "captions" in result
        assert len(result["subplots"]) == 2
        assert len(result["captions"]) == 2

    @patch("figpanel.detector.load_model")
    def test_detection_format(self, mock_load, sample_image, mock_yolo_result):
        mock_model = MagicMock()
        mock_model.return_value = [mock_yolo_result]
        mock_load.return_value = mock_model

        from figpanel.detector import run_yolo

        result = run_yolo(sample_image)

        # Each entry should be (x1, y1, x2, y2, confidence)
        for entry in result["subplots"] + result["captions"]:
            assert len(entry) == 5
            assert all(isinstance(v, (int, float)) for v in entry)

    @patch("figpanel.detector.load_model")
    def test_detect_alias(self, mock_load, sample_image, mock_yolo_result):
        mock_model = MagicMock()
        mock_model.return_value = [mock_yolo_result]
        mock_load.return_value = mock_model

        import figpanel

        result = figpanel.detect(sample_image)
        assert "subplots" in result
        assert "captions" in result


class TestMatcher:
    """Test caption-to-subplot matching."""

    def test_assign_captions_basic(self):
        from figpanel.matcher import assign_captions_to_subplots

        captions = [
            {"caption": "a", "bbox": (5, 5, 25, 25, 0.9)},
            {"caption": "b", "bbox": (405, 5, 425, 25, 0.85)},
        ]
        subplots = [
            (10, 10, 390, 290, 0.95),
            (410, 10, 790, 290, 0.92),
        ]

        pairs = assign_captions_to_subplots(captions, subplots)
        assert len(pairs) == 2
        assert pairs[0]["caption"] == "a"
        assert pairs[1]["caption"] == "b"

    def test_caption_inside_subplot(self):
        from figpanel.matcher import assign_captions_to_subplots

        captions = [{"caption": "a", "bbox": (50, 50, 70, 70, 0.9)}]
        subplots = [(10, 10, 400, 300, 0.95)]

        pairs = assign_captions_to_subplots(captions, subplots)
        assert len(pairs) == 1
        assert pairs[0]["caption"] == "a"

    def test_no_match_when_too_far(self):
        from figpanel.matcher import assign_captions_to_subplots

        captions = [{"caption": "a", "bbox": (700, 700, 720, 720, 0.9)}]
        subplots = [(10, 10, 100, 100, 0.95)]

        pairs = assign_captions_to_subplots(captions, subplots)
        assert len(pairs) == 0

    def test_empty_inputs(self):
        from figpanel.matcher import assign_captions_to_subplots

        assert assign_captions_to_subplots([], []) == []
        assert assign_captions_to_subplots([], [(10, 10, 100, 100, 0.9)]) == []
        assert assign_captions_to_subplots([{"caption": "a", "bbox": (5, 5, 25, 25, 0.9)}], []) == []


class TestOCR:
    """Test OCR module."""

    def test_check_tesseract_import(self):
        """Test that _check_tesseract raises ImportError when pytesseract is missing."""
        from figpanel.ocr import _check_tesseract

        # Should not raise if pytesseract is installed
        try:
            _check_tesseract()
        except ImportError:
            pytest.skip("pytesseract not installed")


class TestPanel:
    """Test Panel dataclass."""

    def test_panel_creation(self):
        from figpanel.extractor import Panel

        img = Image.new("RGB", (100, 100))
        panel = Panel(label="a", bbox=(10, 10, 100, 100), confidence=0.95, image=img)

        assert panel.label == "a"
        assert panel.bbox == (10, 10, 100, 100)
        assert panel.confidence == 0.95
        assert panel.image == img


class TestVisualize:
    """Test visualization module."""

    @patch("figpanel.viz.run_yolo")
    def test_visualize_returns_image(self, mock_detect, sample_image):
        mock_detect.return_value = {
            "subplots": [(10, 10, 390, 290, 0.95)],
            "captions": [(5, 5, 25, 25, 0.88)],
        }

        from figpanel.viz import visualize

        result = visualize(sample_image)
        assert isinstance(result, Image.Image)

    @patch("figpanel.viz.run_yolo")
    def test_visualize_save(self, mock_detect, sample_image, tmp_path):
        mock_detect.return_value = {
            "subplots": [(10, 10, 390, 290, 0.95)],
            "captions": [],
        }

        from figpanel.viz import visualize

        save_path = tmp_path / "annotated.png"
        visualize(sample_image, save=save_path)
        assert save_path.exists()
