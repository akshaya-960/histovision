import sys
sys.path.append("src")

import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models

from dataset import CLASSES, get_transforms
from gradcam import GradCAM
from ood_detector import MahalanobisOOD
from vit_explain import ViTAttentionRollout

DEVICE = torch.device("cpu")


@st.cache_resource(show_spinner=False)
def load_resnet():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    model.load_state_dict(torch.load("models/resnet18_best.pt", map_location=DEVICE))
    model.to(DEVICE).eval()
    return model


@st.cache_resource(show_spinner=False)
def load_vit():
    model = models.vit_b_16(weights=None)
    model.heads.head = nn.Linear(model.hidden_dim, len(CLASSES))
    model.load_state_dict(torch.load("models/vit_b16_best.pt", map_location=DEVICE))
    model.to(DEVICE).eval()
    return model


@st.cache_resource(show_spinner=False)
def load_gradcam(_resnet_model):
    return GradCAM(_resnet_model, target_layer=_resnet_model.layer4[-1])


@st.cache_resource(show_spinner=False)
def load_vit_rollout(_vit_model):
    return ViTAttentionRollout(_vit_model)


@st.cache_resource(show_spinner=False)
def load_ood_detector(_resnet_model):
    """
    Reuses the already-loaded ResNet18 prediction model instead of loading a
    second copy from disk just for feature extraction. A forward hook captures
    the pooled features right before the final classification layer, avoiding
    an entire duplicate ~45MB weight load.
    """
    features = {}

    def hook(module, input, output):
        features["value"] = output

    _resnet_model.avgpool.register_forward_hook(hook)

    class ReusedFeatureExtractor(nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base_model = base_model

        def forward(self, x):
            self.base_model(x)
            return features["value"]

    extractor = ReusedFeatureExtractor(_resnet_model)
    ood = MahalanobisOOD(feature_extractor=extractor, device=DEVICE)
    ood.load("models/ood_stats.npz")
    return ood


@st.cache_resource(show_spinner=False)
def load_transform():
    return get_transforms(train=False)