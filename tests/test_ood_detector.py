
import sys
sys.path.append("src")
import numpy as np
import torch
import torch.nn as nn
from ood_detector import MahalanobisOOD


class DummyFeatureExtractor(nn.Module):
    """Returns a feature vector with some random variation per input -- lets us
    test the math with realistic non-zero variance, without loading real weights."""
    def forward(self, x):
        batch_size = x.shape[0]
        # Base signal + small random noise, applied on top of the input hash
        # so different images get different features but stay reproducible-ish
        torch.manual_seed(int(x.sum().item() * 1000) % (2**31))
        return torch.randn(batch_size, 8, 1, 1)


class DummyLoader:
    """Mimics a DataLoader yielding a few batches of fake images/labels."""
    def __iter__(self):
        for _ in range(3):
            yield torch.randn(4, 3, 224, 224), torch.randint(0, 3, (4,))


def test_ood_fit_produces_valid_threshold():
    ood = MahalanobisOOD(feature_extractor=DummyFeatureExtractor(), device=torch.device("cpu"))
    ood.fit(DummyLoader(), calibration_loader=DummyLoader(), percentile=99.0)
    assert ood.threshold is not None
    assert ood.threshold > 0
    assert ood.class_means.shape[0] == 3


def test_ood_score_returns_expected_types():
    ood = MahalanobisOOD(feature_extractor=DummyFeatureExtractor(), device=torch.device("cpu"))
    ood.fit(DummyLoader(), calibration_loader=DummyLoader())
    dist, is_ood, nearest_class = ood.score(torch.randn(1, 3, 224, 224))
    assert isinstance(dist, float)
    assert isinstance(is_ood, (bool, np.bool_))
    assert nearest_class in ("Normal", "Benign", "Malignant")