"""
Feature extraction from video models.

Takes rendered Pong videos (79 conditions) and extracts hidden states
from open-weight video models at specified layers.

Output format matches the eval battery's expected input:
  (N_units, 79, T) — matching the DMFC neural data format.
"""

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class FeatureExtractor:
    """
    Generic feature extractor using forward hooks.

    Hooks into intermediate layers to capture hidden states.
    Works for both per-frame (image) and video models.
    """

    def __init__(self, model: torch.nn.Module, layer_names: List[str],
                 device: Optional[str] = None):
        self.device = device or _get_device()
        self.model = model.to(self.device).eval()
        self.layer_names = layer_names
        self._features = {}
        self._hooks = []
        self._register_hooks()

    def _register_hooks(self):
        for name, module in self.model.named_modules():
            if name in self.layer_names:
                hook = module.register_forward_hook(self._make_hook(name))
                self._hooks.append(hook)

    def _make_hook(self, name: str):
        def hook_fn(module, input, output):
            if isinstance(output, torch.Tensor):
                self._features[name] = output.detach().cpu()
            elif isinstance(output, tuple):
                self._features[name] = output[0].detach().cpu()
        return hook_fn

    @torch.no_grad()
    def extract_frame(self, frame: torch.Tensor) -> Dict[str, np.ndarray]:
        """
        Extract features from a single frame.

        Args:
            frame: (1, 3, H, W) preprocessed tensor

        Returns:
            {layer_name: (D,) feature vector} — CLS token or spatial mean
        """
        self._features = {}
        frame = frame.to(self.device)
        self.model(frame)

        result = {}
        for name, feat in self._features.items():
            feat = feat.squeeze()
            if feat.ndim == 2:
                # (n_tokens, D) — take CLS token (index 0) or mean
                result[name] = feat[0].numpy()  # CLS token
            elif feat.ndim == 1:
                result[name] = feat.numpy()
            else:
                # Higher dimensional — global average pool
                result[name] = feat.mean(dim=tuple(range(feat.ndim - 1))).numpy()
        return result

    def extract_all_conditions(self, frames_list: List[np.ndarray],
                               transform_fn=None,
                               batch_size: int = 32
                               ) -> Dict[str, np.ndarray]:
        """
        Extract features for all 79 conditions, frame by frame.

        Args:
            frames_list: list of 79 arrays, each (T_i, H, W, 3) uint8
            transform_fn: function to convert (H, W, 3) uint8 → (1, 3, H, W) tensor
            batch_size: frames per forward pass

        Returns:
            {layer_name: (D, 79, max_T)} — matches DMFC format (N, C, T)
        """
        n_conds = len(frames_list)
        max_T = max(len(f) for f in frames_list)
        all_features = {}

        for i in range(n_conds):
            if i % 10 == 0:
                print(f"  Condition {i}/{n_conds}...")

            frames = frames_list[i]  # (T_i, H, W, 3)
            T_i = len(frames)

            # Process frames in batches
            cond_feats = {name: [] for name in self.layer_names}
            for t_start in range(0, T_i, batch_size):
                t_end = min(t_start + batch_size, T_i)
                batch_frames = frames[t_start:t_end]

                # Transform each frame
                batch_tensors = []
                for frame in batch_frames:
                    if transform_fn is not None:
                        tensor = transform_fn(frame)
                    else:
                        tensor = self._default_transform(frame)
                    batch_tensors.append(tensor)

                batch = torch.cat(batch_tensors, dim=0).to(self.device)  # (B, 3, H, W)

                self._features = {}
                self.model(batch)

                for name in self.layer_names:
                    if name in self._features:
                        feat = self._features[name].cpu()
                        if feat.ndim == 3:
                            # (B, n_tokens, D) — take CLS token
                            feat = feat[:, 0, :]  # (B, D)
                        elif feat.ndim == 2:
                            pass  # (B, D) already
                        cond_feats[name].append(feat.numpy())

            # Concatenate across batches
            for name in self.layer_names:
                if cond_feats[name]:
                    feat_arr = np.concatenate(cond_feats[name], axis=0)  # (T_i, D)
                    D = feat_arr.shape[1]

                    if name not in all_features:
                        all_features[name] = np.full((D, n_conds, max_T), np.nan)

                    all_features[name][:, i, :T_i] = feat_arr.T

        return all_features

    def _default_transform(self, frame: np.ndarray) -> torch.Tensor:
        """Default transform: resize to 224, normalize with ImageNet stats."""
        # (H, W, 3) uint8 → (1, 3, H, W) float
        tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        tensor = F.interpolate(tensor.unsqueeze(0), size=(224, 224),
                               mode='bilinear', align_corners=False)
        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        tensor = (tensor - mean) / std
        return tensor

    def cleanup(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks = []


def load_dinov2(variant: str = "dinov2_vitb14") -> Tuple:
    """
    Load DINOv2 as a per-frame feature extractor.

    Returns:
        (model, layer_names)
    """
    model = torch.hub.load("facebookresearch/dinov2", variant)
    n_blocks = len(model.blocks)
    # Sample 4 layers across depth
    layer_names = [
        f"blocks.{n_blocks // 4 - 1}",      # early
        f"blocks.{n_blocks // 2 - 1}",      # mid
        f"blocks.{3 * n_blocks // 4 - 1}",  # late-mid
        f"blocks.{n_blocks - 1}",            # final
    ]
    return model, layer_names


def load_videomae_v2() -> Tuple:
    """Load VideoMAE V2 from HuggingFace."""
    from transformers import VideoMAEModel
    model = VideoMAEModel.from_pretrained(
        "MCG-NJU/videomae-base-finetuned-kinetics"
    )
    layer_names = [
        "encoder.layer.3",
        "encoder.layer.6",
        "encoder.layer.9",
        "encoder.layer.11",
    ]
    return model, layer_names


def extract_and_save(model_name: str, frames_list: List[np.ndarray],
                     output_path: str, batch_size: int = 32):
    """
    End-to-end: load model, extract features, save.

    Args:
        model_name: 'dinov2' or 'videomae_v2'
        frames_list: list of 79 arrays, each (T_i, H, W, 3)
        output_path: path to save .npz file
        batch_size: frames per batch
    """
    if model_name == "dinov2":
        model, layer_names = load_dinov2()
    elif model_name == "videomae_v2":
        model, layer_names = load_videomae_v2()
    else:
        raise ValueError(f"Unknown model: {model_name}")

    print(f"Loaded {model_name}, layers: {layer_names}")
    extractor = FeatureExtractor(model, layer_names)

    print(f"Extracting features from {len(frames_list)} conditions...")
    features = extractor.extract_all_conditions(frames_list,
                                                 batch_size=batch_size)

    # Save
    np.savez_compressed(output_path, **features)
    print(f"Saved features to {output_path}")
    for name, feat in features.items():
        print(f"  {name}: {feat.shape}")

    extractor.cleanup()
    return features
