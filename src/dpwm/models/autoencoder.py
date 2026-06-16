from __future__ import annotations

import torch
import torch.nn as nn

from dpwm.envs.unlocked_door import UnlockedDoorEnv
from dpwm.render import TILE_SIZE

DEFAULT_IMAGE_SIZE = UnlockedDoorEnv.height * TILE_SIZE
DEFAULT_CONTEXT_DIM = 2


def conv_spatial_size(image_size: int, n_convs: int = 3) -> int:
    """Spatial H=W after n_convs with kernel 4, stride 2, padding 1."""
    size = image_size
    for _ in range(n_convs):
        size = (size + 2 - 4) // 2 + 1
    return size


class Encoder(nn.Module):
    def __init__(self, latent_dim: int = 2, image_size: int = DEFAULT_IMAGE_SIZE) -> None:
        super().__init__()
        spatial = conv_spatial_size(image_size)
        flat_dim = 64 * spatial * spatial
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 4, stride=2, padding=1),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.conv(x))


class Decoder(nn.Module):
    def __init__(
        self,
        latent_dim: int = 2,
        context_dim: int = DEFAULT_CONTEXT_DIM,
        image_size: int = DEFAULT_IMAGE_SIZE,
    ) -> None:
        super().__init__()
        self.spatial = conv_spatial_size(image_size)
        flat_dim = 64 * self.spatial * self.spatial
        self.fc = nn.Sequential(
            nn.Linear(latent_dim + context_dim, 128),
            nn.ReLU(),
            nn.Linear(128, flat_dim),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1)
        )

    def forward(self, z: torch.Tensor, context_latent: torch.Tensor, noise_std: float = 0.01) -> torch.Tensor:
        if self.training:
            z = z + torch.randn_like(z) * noise_std
        x = torch.cat([z, context_latent], dim=1)
        x = self.fc(x)
        x = x.view(-1, 64, self.spatial, self.spatial)
        x = self.deconv(x)
        if not self.training:
            x = torch.clamp(x, 0.0, 1.0)
        return x


class Autoencoder(nn.Module):
    def __init__(
        self,
        latent_dim: int = 2,
        context_dim: int = DEFAULT_CONTEXT_DIM,
        image_size: int = DEFAULT_IMAGE_SIZE,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.context_dim = context_dim
        self.encoder = Encoder(latent_dim, image_size)
        self.decoder = Decoder(latent_dim, context_dim, image_size)

    def decode(self, z: torch.Tensor, context_latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(z, context_latent)

    def forward(
        self, x: torch.Tensor, context_latent: torch.Tensor, noise_std: float = 0.01
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        recon = self.decoder(z, context_latent, noise_std)
        return z, recon

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)
