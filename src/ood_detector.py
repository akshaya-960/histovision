import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.covariance import LedoitWolf

from dataset import HistoDataset, CLASSES


class MahalanobisOOD:
    """
    OOD detector using Mahalanobis distance in ResNet18's penultimate
    feature space, following Lee et al. (2018). Fits a per-class Gaussian
    (shared covariance) on TRAINING features, then flags new images whose
    nearest-class distance exceeds a percentile-based threshold learned
    from training data itself.
    """

    def __init__(self, feature_extractor, device):
        self.feature_extractor = feature_extractor
        self.device = device
        self.class_means = None      # [num_classes, feat_dim]
        self.inv_covariance = None   # [feat_dim, feat_dim], shared across classes
        self.threshold = None

    @torch.no_grad()
    def _extract_features(self, loader):
        feats, labels = [], []
        for images, lbls in loader:
            images = images.to(self.device)
            f = self.feature_extractor(images).squeeze(-1).squeeze(-1)  # [B, 512]
            feats.append(f.cpu().numpy())
            labels.append(lbls.numpy())
        return np.concatenate(feats), np.concatenate(labels)

    def fit(self, train_loader, calibration_loader=None, percentile=99.0):
        features, labels = self._extract_features(train_loader)

        means = []
        for c in range(len(CLASSES)):
            class_feats = features[labels == c]
            means.append(class_feats.mean(axis=0))
        self.class_means = np.stack(means)

        centered = np.concatenate([
            features[labels == c] - self.class_means[c] for c in range(len(CLASSES))
        ])
        lw = LedoitWolf().fit(centered)
        self.inv_covariance = lw.precision_

        # Calibrate the threshold on a held-out set (validation), NOT training data.
        # Training images sit artificially close to their own class mean since they
        # defined it -- calibrating there underestimates normal in-distribution
        # distance and causes real (but unseen) in-distribution images to false-flag.
        if calibration_loader is not None:
            calib_features, calib_labels = self._extract_features(calibration_loader)
        else:
            calib_features, calib_labels = features, labels  # fallback, not recommended

        calib_distances = [
            self._mahalanobis(calib_features[i], self.class_means[calib_labels[i]])
            for i in range(len(calib_features))
        ]
        self.threshold = float(np.percentile(calib_distances, percentile))

        print(f"OOD detector fit complete.")
        print(f"Calibration distance stats: min={min(calib_distances):.2f}, "
              f"max={max(calib_distances):.2f}, mean={np.mean(calib_distances):.2f}")
        print(f"Threshold ({percentile}th percentile): {self.threshold:.2f}")
        return self

    def _mahalanobis(self, feat, mean):
        diff = feat - mean
        return float(np.sqrt(diff @ self.inv_covariance @ diff.T))

    @torch.no_grad()
    def score(self, input_tensor):
        """Returns (min_distance, is_ood, nearest_class)."""
        f = self.feature_extractor(input_tensor.to(self.device)).squeeze(-1).squeeze(-1)
        f = f.cpu().numpy()[0]

        distances = [self._mahalanobis(f, self.class_means[c]) for c in range(len(CLASSES))]
        min_dist = min(distances)
        nearest_class = CLASSES[int(np.argmin(distances))]
        is_ood = min_dist > self.threshold

        return min_dist, is_ood, nearest_class

    def save(self, path):
        np.savez(path,
                 class_means=self.class_means,
                 inv_covariance=self.inv_covariance,
                 threshold=self.threshold)

    def load(self, path):
        data = np.load(path)
        self.class_means = data["class_means"]
        self.inv_covariance = data["inv_covariance"]
        self.threshold = float(data["threshold"])
        return self