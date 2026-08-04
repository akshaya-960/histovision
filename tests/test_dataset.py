import sys
sys.path.append("src")
from dataset import CLASSES


def test_classes_defined():
    assert len(CLASSES) == 3
    assert set(CLASSES) == {"Normal", "Benign", "Malignant"}


def test_classes_are_strings():
    assert all(isinstance(c, str) for c in CLASSES)