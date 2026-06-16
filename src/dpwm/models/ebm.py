from __future__ import annotations

import torch
import torch.nn as nn

from dpwm.models.context_model import DEFAULT_CONTEXT_DIM


class PotentialV(nn.Module):
    """Unconditional energy over 2D latent position."""

    def __init__(self, latent_dim: int = 2, hidden_dim: int = 512) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).squeeze(-1)


class PotentialW(nn.Module):
    """Reachability energy: W(z_candidate | context_latent)."""

    def __init__(
        self,
        latent_dim: int = 2,
        context_dim: int = DEFAULT_CONTEXT_DIM,
        hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        input_dim = latent_dim + context_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, z_candidate: torch.Tensor, context_latent: torch.Tensor
    ) -> torch.Tensor:
        if z_candidate.dim() == 3:
            context_latent = context_latent.unsqueeze(1).repeat(1, z_candidate.size(1), 1)
        #print(z_candidate.shape, context_latent.shape)
        return self.net(torch.cat([z_candidate, context_latent], dim=-1)).squeeze(-1)
