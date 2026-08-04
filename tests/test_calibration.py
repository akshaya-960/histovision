import sys
sys.path.append("src")
import numpy as np
from calibration import reliability_diagram


def test_perfect_calibration_gives_zero_ece():
    # If every bin's average confidence exactly equals its accuracy, ECE should be ~0.
    confidences = np.array([0.9] * 10 + [0.5] * 10)
    correct = np.array([1] * 9 + [0] * 1 + [1] * 5 + [0] * 5)  # 90% acc at 0.9, 50% acc at 0.5
    result = reliability_diagram(confidences, correct, n_bins=10)
    assert result["ece"] < 0.05


def test_reliability_diagram_bin_count():
    confidences = np.random.uniform(0, 1, 50)
    correct = np.random.randint(0, 2, 50)
    result = reliability_diagram(confidences, correct, n_bins=10)
    assert len(result["bin_accs"]) == 10
    assert len(result["bin_confs"]) == 10