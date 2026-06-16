from __future__ import annotations

import torch
import torch.nn as nn

from dpwm.envs.unlocked_door import ACTIONS
from dpwm.models.autoencoder import DEFAULT_CONTEXT_DIM, DEFAULT_IMAGE_SIZE, Encoder

NUM_CONTEXT_ACTIONS = len(ACTIONS)
LATENT_DIM = 2


def infer_context_dim(
    context_model_state: dict[str, torch.Tensor] | None = None,
    w_model_state: dict[str, torch.Tensor] | None = None,
    latent_dim: int = LATENT_DIM,
) -> int:
    """Read context width from a saved checkpoint (supports 1D or 3D c)."""
    if context_model_state:
        for key, weight in context_model_state.items():
            if key.endswith("head.2.weight"):
                return int(weight.shape[0])
    if w_model_state is not None:
        in_dim = int(w_model_state["net.0.weight"].shape[1])
        return in_dim - latent_dim
    return DEFAULT_CONTEXT_DIM


def infer_w_context_dim(w_model_state: dict[str, torch.Tensor]) -> int:
    return infer_context_dim(w_model_state=w_model_state)


class ContextModel(nn.Module):
    """
    Encode reachability regime for decoder, W, and WM.

    Input: anchor snapshot after the last transformative action + that action label.
    Output: context_dim vector (default 3: room-side / door / regime capacity for W).
    Trained jointly with the AE on trajectory observation/context pairs, then frozen
    or fine-tuned with W (ctx_w stage).
    """

    def __init__(
        self,
        context_dim: int = DEFAULT_CONTEXT_DIM,
        image_size: int = DEFAULT_IMAGE_SIZE,
        num_actions: int = NUM_CONTEXT_ACTIONS,
        image_dim: int = 32,
        action_dim: int = 16,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.context_dim = context_dim
        self.encoder = Encoder(image_dim, image_size)
        self.action_embed = nn.Embedding(num_actions, action_dim)
        self.head = nn.Sequential(
            nn.Linear(image_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, context_dim),
        )

    def forward(
        self, image: torch.Tensor, context_action: torch.Tensor
    ) -> torch.Tensor:
        image_feat = self.encoder(image)
        action_feat = self.action_embed(context_action.long())
        return self.head(torch.cat([image_feat, action_feat], dim=-1))
