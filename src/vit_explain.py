import torch
import numpy as np
import cv2


class ViTAttentionRollout:
    """
    Attention rollout for Vision Transformers (Abnar & Zuidema, 2020).
    torchvision's ViT doesn't return attention weights by default (it uses an
    efficient fused attention kernel), so we monkey-patch each encoder block's
    self_attention.forward to force need_weights=True and capture the weights.
    Rolling out (multiplying) attention across layers -- with an added identity
    matrix per layer to account for residual connections -- gives a single map
    of how much the class token ultimately attended to each patch. This plays
    the same explanatory role Grad-CAM plays for the CNN.
    """

    def __init__(self, model):
        self.model = model
        self.attn_weights = []
        self._patch_layers()

    def _patch_layers(self):
        for layer in self.model.encoder.layers:
            mha = layer.self_attention

            # Guard against re-patching an already-patched layer -- without this,
            # every Streamlit rerun (which re-executes the whole script) would wrap
            # forward again on top of the previous wrap, since the cached vit_model
            # object persists across reruns.
            if getattr(mha, "_rollout_patched", False):
                continue

            orig_forward = mha.forward
            # NOTE: this "storage" reference is baked into the closure below and
            # will point at whatever list self.attn_weights was AT PATCH TIME
            # (i.e. right now, in __init__). generate() must mutate this same
            # list object (via .clear()), never reassign self.attn_weights to a
            # new list, or these closures go on writing to an orphaned list that
            # nothing else can see.
            storage = self.attn_weights

            def new_forward(query, key, value, orig_forward=orig_forward, storage=storage, **kwargs):
                kwargs["need_weights"] = True
                kwargs["average_attn_weights"] = True
                out, weights = orig_forward(query, key, value, **kwargs)
                storage.append(weights.detach())
                return out, weights

            mha.forward = new_forward
            mha._rollout_patched = True

    @torch.no_grad()
    def generate(self, input_tensor):
        # Must clear the EXISTING list in place, not reassign self.attn_weights
        # to a new list -- see the note in _patch_layers above.
        self.attn_weights.clear()
        self.model(input_tensor)

        if len(self.attn_weights) == 0:
            raise RuntimeError(
                "No attention weights captured — the forward-patching hook did not fire. "
                "This usually means the ViT model's architecture doesn't match what this "
                "class expects (e.g. a different torchvision version), or the encoder "
                "layers were never patched. Try fully restarting the Streamlit app."
            )

        num_tokens = self.attn_weights[0].size(-1)
        result = torch.eye(num_tokens)

        for attn in self.attn_weights:
            attn = attn[0]  # drop batch dim -> [tokens, tokens]
            attn = attn + torch.eye(num_tokens)            # residual connection
            attn = attn / attn.sum(dim=-1, keepdim=True)    # renormalize
            result = attn @ result

        # Row 0 = class token; columns 1: = attention received from each patch
        mask = result[0, 1:].numpy()
        grid_size = int(np.sqrt(mask.shape[0]))
        mask = mask.reshape(grid_size, grid_size)
        mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
        mask = cv2.resize(mask, (224, 224))
        return mask


def overlay_attention(img_rgb, mask, alpha=0.4):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (alpha * heatmap + (1 - alpha) * img_rgb).astype(np.uint8)
    return overlay