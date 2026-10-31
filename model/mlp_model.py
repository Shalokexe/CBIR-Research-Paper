"""
Metric-Learning MLP Architecture for Query-Adaptive Weight Prediction (M5).
Maps a 13-dimensional input feature representation [C, T] to 2 non-negative weights
[w_colour, w_texture] that sum to 1.0 via Softmax.
"""

from pathlib import Path
from typing import Tuple, Dict, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import config


class AdaptiveWeightMLP(nn.Module):
    """
    Lightweight Multi-Layer Perceptron:
    Input: 13 dimensions (9 HSV color moments + 4 GLCM texture properties)
    Hidden Layer 1: 16 neurons with LeakyReLU(negative_slope=0.1)
    Hidden Layer 2: 8 neurons with LeakyReLU(negative_slope=0.1)
    Output Layer: 2 neurons with Softmax(dim=-1)
    Output Interpretation: [w_colour, w_texture], w_colour >= 0, w_texture >= 0, w_colour + w_texture = 1.0
    """
    def __init__(self, input_dim: int = config.MLP_INPUT_DIM):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, config.MLP_HIDDEN_1)
        self.fc2 = nn.Linear(config.MLP_HIDDEN_1, config.MLP_HIDDEN_2)
        self.fc3 = nn.Linear(config.MLP_HIDDEN_2, config.MLP_OUTPUT_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Tensor of shape (batch_size, 13) or (13,).
        Returns:
            Tensor of shape (batch_size, 2) representing [w_colour, w_texture].
        """
        if x.ndim == 1:
            x = x.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        h1 = F.leaky_relu(self.fc1(x), negative_slope=config.MLP_LEAKY_RELU_SLOPE)
        h2 = F.leaky_relu(self.fc2(h1), negative_slope=config.MLP_LEAKY_RELU_SLOPE)
        logits = self.fc3(h2)

        weights = F.softmax(logits, dim=-1)
        return weights.squeeze(0) if squeeze_output else weights


def load_trained_mlp(checkpoint_path: Path | str | None = None) -> Tuple[AdaptiveWeightMLP, Dict[str, Any]]:
    """
    Loads trained AdaptiveWeightMLP model checkpoint.

    Args:
        checkpoint_path: Path to PyTorch checkpoint. Defaults to config.MODEL_CHECKPOINT_PATH.

    Returns:
        model: Evaluated PyTorch AdaptiveWeightMLP in eval mode.
        metadata: Checkpoint metadata dictionary.
    """
    if checkpoint_path is None:
        checkpoint_path = config.MODEL_CHECKPOINT_PATH
    path = Path(checkpoint_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at: {path}")

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = AdaptiveWeightMLP(input_dim=config.MLP_INPUT_DIM)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        metadata = {k: v for k, v in ckpt.items() if k != "model_state_dict"}
    else:
        model.load_state_dict(ckpt, strict=True)
        metadata = {}

    model.eval()
    return model, metadata
