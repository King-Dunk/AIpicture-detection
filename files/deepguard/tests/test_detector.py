"""
Tests for DeepGuard detector.
Run with: pytest tests/
"""

import numpy as np
import pytest
from PIL import Image
import tempfile
import os

from deepguard.detector import DeepfakeDetector, DetectionResult


def make_synthetic_image(mode="natural", size=(256, 256)) -> str:
    """Create a temporary test image and return its path."""
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)

    if mode == "natural":
        # Simulate natural photo: structured noise + gradients
        rng = np.random.default_rng(42)
        img = rng.normal(128, 30, (*size, 3)).astype(np.uint8)
        # Add gradient (natural images have structure)
        gradient = np.linspace(0, 50, size[1]).astype(np.uint8)
        img[:, :, 0] = np.clip(img[:, :, 0] + gradient[np.newaxis, :], 0, 255)

    elif mode == "uniform":
        # Simulate AI image: very uniform noise
        rng = np.random.default_rng(0)
        img = np.full((*size, 3), 128, dtype=np.uint8)
        img += rng.integers(-5, 5, img.shape, dtype=np.int8).astype(np.uint8)

    Image.fromarray(img.astype(np.uint8)).save(tmp.name)
    return tmp.name


class TestDetector:

    def setup_method(self):
        self.detector = DeepfakeDetector(threshold=0.5)

    def test_returns_detection_result(self):
        path = make_synthetic_image()
        try:
            result = self.detector.detect(path)
            assert isinstance(result, DetectionResult)
        finally:
            os.unlink(path)

    def test_confidence_in_range(self):
        path = make_synthetic_image()
        try:
            result = self.detector.detect(path)
            assert 0.0 <= result.confidence <= 1.0
        finally:
            os.unlink(path)

    def test_risk_level_values(self):
        path = make_synthetic_image()
        try:
            result = self.detector.detect(path)
            assert result.risk_level in ("LOW", "MEDIUM", "HIGH")
        finally:
            os.unlink(path)

    def test_all_signals_present(self):
        path = make_synthetic_image()
        try:
            result = self.detector.detect(path)
            expected_signals = {
                "frequency_anomaly", "noise_residual",
                "texture_score", "channel_mismatch", "edge_artifact"
            }
            assert set(result.signals.keys()) == expected_signals
        finally:
            os.unlink(path)

    def test_all_signals_in_range(self):
        path = make_synthetic_image()
        try:
            result = self.detector.detect(path)
            for name, score in result.signals.items():
                assert 0.0 <= score <= 1.0, f"Signal {name} out of range: {score}"
        finally:
            os.unlink(path)

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            self.detector.detect("/nonexistent/path/fake.jpg")

    def test_heatmap_saved(self):
        path = make_synthetic_image()
        try:
            result = self.detector.detect(path, save_heatmap=True)
            assert result.heatmap_path is not None
            assert os.path.exists(result.heatmap_path)
            os.unlink(result.heatmap_path)
        finally:
            os.unlink(path)

    def test_custom_threshold(self):
        # With threshold=0, everything should be flagged
        detector_low = DeepfakeDetector(threshold=0.0)
        path = make_synthetic_image()
        try:
            result = detector_low.detect(path)
            assert result.is_ai_generated is True
        finally:
            os.unlink(path)

        # With threshold=1.0, nothing should be flagged
        detector_high = DeepfakeDetector(threshold=1.0)
        path = make_synthetic_image()
        try:
            result = detector_high.detect(path)
            assert result.is_ai_generated is False
        finally:
            os.unlink(path)

    def test_str_representation(self):
        path = make_synthetic_image()
        try:
            result = self.detector.detect(path)
            s = str(result)
            assert "Verdict" in s
            assert "Confidence" in s
        finally:
            os.unlink(path)
