import torch
import torch.nn.functional as F
import numpy as np
import cv2

class GradCAM:
    """
    Grad-CAM for CNNs: uses gradients flowing into the last conv layer
    to produce a heatmap of which regions influenced the prediction.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_class=None):
        self.model.eval()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        score = output[0, target_class]
        score.backward()

        # Global-average-pool the gradients across spatial dims -> one
        # importance weight per channel (this is the "Grad" in Grad-CAM)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)

        # Weighted combination of activation maps, then ReLU
        # (we only care about features that POSITIVELY influenced the class)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = cam.squeeze().cpu().numpy()
        cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)  # normalize 0-1

        return cam, target_class


def overlay_heatmap(img_rgb, cam, alpha=0.4):
    """Overlay a Grad-CAM heatmap onto the original RGB image."""
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (alpha * heatmap + (1 - alpha) * img_rgb).astype(np.uint8)
    return overlay