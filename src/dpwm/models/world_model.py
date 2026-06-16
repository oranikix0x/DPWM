from __future__ import annotations

import torch
import torch.nn as nn

from dpwm.envs.unlocked_door import ACTIONS
from dpwm.models.context_model import DEFAULT_CONTEXT_DIM
from dpwm.models.ebm import PotentialV, PotentialW
from dpwm.models.context_model import ContextModel
from dpwm.models.autoencoder import Autoencoder

NUM_ACTIONS = len(ACTIONS)


class LatentDynamics(nn.Module):
    """Predict next AE latent: (z, context, action) -> z_next."""

    def __init__(
        self,
        latent_dim: int = 2,
        context_dim: int = DEFAULT_CONTEXT_DIM,
        num_actions: int = NUM_ACTIONS,
        action_dim: int = 32,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.action_embed = nn.Embedding(num_actions, action_dim)
        self.net = nn.Sequential(
            nn.Linear(latent_dim + context_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(
        self,
        z: torch.Tensor,
        context_latent: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        action_feat = self.action_embed(action.long())
        x = torch.cat([z, context_latent, action_feat], dim=-1)
        return self.net(x) + z #residual connection


def transition_energy(
    v_model: PotentialV,
    w_model: PotentialW,
    z: torch.Tensor,
    context_latent: torch.Tensor,
) -> torch.Tensor:
    return v_model(z) + w_model(z, context_latent)


def is_energy_allowed(
    v_model: PotentialV,
    w_model: PotentialW,
    z: torch.Tensor,
    context_latent: torch.Tensor,
    e_threshold: torch.Tensor | float,
) -> torch.Tensor:
    """True where V(z) + W(z | context) <= E."""
    energy = transition_energy(v_model, w_model, z, context_latent)
    return energy <= e_threshold


@torch.no_grad()
def nearest_state_indices(z: torch.Tensor, z_table: torch.Tensor) -> torch.Tensor:
    """Map latent vectors to nearest rows in a precomputed state embedding table."""
    return torch.cdist(z, z_table).argmin(dim=1)


@torch.no_grad()
def constrained_next_latent(
    dynamics: LatentDynamics,
    context_model: ContextModel,
    v_model: PotentialV,
    w_model: PotentialW,
    e_threshold: torch.Tensor | float,
    z_current: torch.Tensor,
    context_image: torch.Tensor,
    context_action: torch.Tensor,
    action: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Predict next latent; keep current when prediction violates V+W <= E.

    Returns (z_out, z_pred, allowed_mask).
    """
    dynamics.eval()
    context_model.eval()
    v_model.eval()
    w_model.eval()

    context_latent = context_model(context_image, context_action)
    z_pred = dynamics(z_current, context_latent, action)
    allowed = is_energy_allowed(
        v_model, w_model, z_pred, context_latent, e_threshold
    )
    z_out = torch.where(allowed.unsqueeze(-1), z_pred, z_current)
    return z_out, z_pred, allowed
