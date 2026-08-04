import sys
sys.path.append("src")
import numpy as np
from gradcam import overlay_heatmap


def test_overlay_heatmap_output_shape():
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    cam = np.random.rand(224, 224).astype(np.float32)
    result = overlay_heatmap(img, cam)
    assert result.shape == (224, 224, 3)
    assert result.dtype == np.uint8


def test_overlay_heatmap_values_in_range():
    img = np.full((10, 10, 3), 255, dtype=np.uint8)
    cam = np.zeros((10, 10), dtype=np.float32)
    result = overlay_heatmap(img, cam)
    assert result.min() >= 0
    assert result.max() <= 255