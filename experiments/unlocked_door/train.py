"""
Train the unlocked-door symbolic pipeline:

  1. ae     — Encoder + ContextModel + Decoder (door regime in c; anchor position not required)
  2. v      — unconditional EBM on AE latents
  3. w      — W(z | c) + energy budget E; then decoder refresh (encoder + c frozen)
  4. ctx_w  — fine-tune ContextModel + W + E + Decoder jointly
  5. decoder— optional Decoder-only refresh (Encoder + ContextModel frozen)
  6. e      — optional E-only refine (W frozen; same threshold loss)
  7. wm     — LatentDynamics (z, c, action) -> z_next, energy-gated
  8. all    — ae → v → w → ctx_w → e → wm

Example:
  python experiments/unlocked_door_symbolic/train.py --stage ae
  python experiments/unlocked_door_symbolic/train.py --stage ctx_w
  python experiments/unlocked_door_symbolic/train.py --stage all
"""

from __future__ import annotations

import argparse
from itertools import cycle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from dpwm.data.load import (
    ContextTransitionDataset,
    ImageActionTransitionDataset,
    ObservationContextDataset,
    SavedDataset,
    ae_training_indices,
    load_dataset,
    split_dataset,
    transition_sample_weights,
    unique_referenced_states,
)
from dpwm.envs.unlocked_door import ACTION_TO_IDX, UnlockedDoorEnv
from dpwm.models.autoencoder import Autoencoder
from dpwm.models.context_model import ContextModel, DEFAULT_CONTEXT_DIM, infer_w_context_dim
from dpwm.models.ebm import PotentialV, PotentialW
from dpwm.models.world_model import (
    LatentDynamics,
    constrained_next_latent,
    is_energy_allowed,
    nearest_state_indices,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
WALL_X_NORM = UnlockedDoorEnv.wall_x / (UnlockedDoorEnv.width - 1)


def save_checkpoint(path: Path, **objects) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(objects, path)
    print(f"saved {path}")


def clone_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def load_cloned_state_dict(
    module: torch.nn.Module, state: dict[str, torch.Tensor]
) -> None:
    module.load_state_dict(state)


def load_checkpoint(path: Path) -> dict:
    return torch.load(path, map_location=DEVICE, weights_only=False)


def load_ae_and_context(path: Path) -> tuple[Autoencoder, ContextModel]:
    ckpt = load_checkpoint(path)
    ae = Autoencoder().to(DEVICE)
    context_model = ContextModel(context_dim=DEFAULT_CONTEXT_DIM).to(DEVICE)
    if "context_model" in ckpt:
        ae.load_state_dict(ckpt["model"])
        context_model.load_state_dict(ckpt["context_model"])
    elif "model" in ckpt:
        raise RuntimeError(
            f"{path} is an old AE checkpoint (door decoder). Retrain with --stage ae."
        )
    else:
        raise KeyError(f"{path} missing model/context_model")
    ae.eval()
    context_model.eval()
    return ae, context_model


def load_w_model(path: Path, context_dim: int | None = None) -> PotentialW:
    ckpt = load_checkpoint(path)
    w_key = "w_model" if "w_model" in ckpt else "model"
    if context_dim is None:
        context_dim = infer_w_context_dim(ckpt[w_key])
    w_model = PotentialW(context_dim=context_dim).to(DEVICE)
    w_model.load_state_dict(ckpt[w_key])
    w_model.eval()
    return w_model


def load_energy_e(w_path: Path, e_path: Path) -> torch.Tensor:
    if e_path.exists():
        return torch.tensor(float(load_checkpoint(e_path)["E"]), device=DEVICE)
    if w_path.exists():
        w_ckpt = load_checkpoint(w_path)
        if "E" in w_ckpt:
            return torch.tensor(float(w_ckpt["E"]), device=DEVICE)
    raise FileNotFoundError(
        f"no E in {e_path} or {w_path}; run --stage w (or e) first"
    )


def resolve_init_e(
    w_checkpoint: Path,
    e_checkpoint: Path,
    loader: DataLoader,
    z_all: torch.Tensor,
    v_model: PotentialV,
    w_model: PotentialW,
    context_model: ContextModel,
    max_batches: int = 8,
) -> float:
    """Prefer saved E; otherwise estimate from a short positive-energy pass."""
    if e_checkpoint.exists():
        return float(load_checkpoint(e_checkpoint)["E"])
    if w_checkpoint.exists():
        w_ckpt = load_checkpoint(w_checkpoint)
        if "E" in w_ckpt:
            return float(w_ckpt["E"])
    return mean_positive_energy(
        loader,
        z_all,
        v_model,
        w_model,
        context_model,
        max_batches=max_batches,
    )


def print_split_summary(train_ds: SavedDataset, val_ds: SavedDataset) -> None:
    train_states = set(unique_referenced_states(train_ds).tolist())
    val_states = set(unique_referenced_states(val_ds).tolist())
    overlap = train_states & val_states
    print(
        f"Split: {len(train_ds.traj_lengths)} train trajectories "
        f"({len(train_ds.next_idx)} transitions), "
        f"{len(val_ds.traj_lengths)} val trajectories "
        f"({len(val_ds.next_idx)} transitions)"
    )
    print(
        f"  states: {len(train_states)} train, {len(val_states)} val, "
        f"{len(overlap)} shared in lookup table"
    )


@torch.no_grad()
def eval_ae_recon(
    ae: Autoencoder,
    context_model: ContextModel,
    dataset: SavedDataset,
    batch_size: int,
) -> float:
    if len(dataset.state_idx) == 0:
        return float("nan")
    ae.eval()
    context_model.eval()
    loader = DataLoader(ObservationContextDataset(dataset), batch_size=batch_size)
    total = 0.0
    n = 0
    for image, context_image, context_action, _xy in loader:
        image = image.to(DEVICE)
        context_image = context_image.to(DEVICE)
        context_action = context_action.to(DEVICE)
        context_latent = context_model(context_image, context_action)
        z = ae.encode(image)
        recon = ae.decode(z, context_latent)
        batch_mse = F.mse_loss(recon, image, reduction="mean").item()
        total += batch_mse * image.size(0)
        n += image.size(0)
    return total / n


@torch.no_grad()
def eval_v_loss(
    model: PotentialV,
    z_all: torch.Tensor,
    state_indices: np.ndarray,
    batch_size: int,
    margin: float,
    num_negatives: int,
    z_min: torch.Tensor,
    z_max: torch.Tensor,
) -> float:
    if len(state_indices) == 0:
        return float("nan")
    model.eval()
    idx = torch.from_numpy(state_indices).long().to(DEVICE)
    total = 0.0
    steps = 0
    for start in range(0, len(idx), batch_size):
        batch_idx = idx[start : start + batch_size]
        z_pos = z_all[batch_idx]
        z_neg = sample_box_negatives(len(z_pos), num_negatives, z_min, z_max)
        e_pos = model(z_pos)
        e_neg = model(z_neg)
        total += F.relu(e_pos - e_neg + margin).mean().item()
        steps += 1
    return total / steps


def threshold_energy_loss(
    e_pos: torch.Tensor,
    e_neg: torch.Tensor,
    e_threshold: torch.Tensor,
    margin: float,
    lambda_e: float,
) -> torch.Tensor:
    """Positives below E, negatives above E (same rule as rollout gating)."""
    above_pos = F.relu(e_pos - e_threshold + margin)
    below_neg = F.relu(e_threshold - e_neg + margin).mean(dim=1)
    return above_pos.mean() + below_neg.mean() + lambda_e * e_threshold


@torch.no_grad()
def mean_positive_energy(
    loader: DataLoader,
    z_all: torch.Tensor,
    v_model: PotentialV,
    w_model: PotentialW,
    context_model: ContextModel,
    max_batches: int | None = 8,
) -> float:
    total = 0.0
    count = 0
    for batch_idx, (next_idx, context_image, context_action) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        next_idx = next_idx.to(DEVICE)
        context_image = context_image.to(DEVICE)
        context_action = context_action.to(DEVICE)
        z_pos = z_all[next_idx]
        ctx_latent = context_model(context_image, context_action)
        e_pos = total_energy(v_model, w_model, z_pos, ctx_latent)
        total += e_pos.sum().item()
        count += e_pos.numel()
    if count == 0:
        raise RuntimeError("mean_positive_energy: empty loader")
    return total / count


@torch.no_grad()
def eval_w_e_loss(
    dataset: SavedDataset,
    z_all: torch.Tensor,
    v_model: PotentialV,
    w_model: PotentialW,
    context_model: ContextModel,
    e_threshold: torch.Tensor,
    batch_size: int,
    margin: float,
    num_negatives: int,
    lambda_e: float,
    z_min: torch.Tensor,
    z_max: torch.Tensor,
) -> float:
    if len(dataset.next_idx) == 0:
        return float("nan")
    v_model.eval()
    w_model.eval()
    context_model.eval()
    loader = DataLoader(ContextTransitionDataset(dataset), batch_size=batch_size)
    total = 0.0
    for next_idx, context_image, context_action in loader:
        next_idx = next_idx.to(DEVICE)
        context_image = context_image.to(DEVICE)
        context_action = context_action.to(DEVICE)
        z_pos = z_all[next_idx]
        context_latent = context_model(context_image, context_action)
        z_neg, ctx_latent = sample_w_negatives(context_latent, num_negatives, z_min, z_max)
        e_pos = total_energy(v_model, w_model, z_pos, ctx_latent)
        e_neg = total_energy(v_model, w_model, z_neg, ctx_latent)
        total += threshold_energy_loss(
            e_pos, e_neg, e_threshold, margin, lambda_e
        ).item()
    return total / len(loader)


@torch.no_grad()
def eval_e_loss(
    dataset: SavedDataset,
    z_all: torch.Tensor,
    v_model: PotentialV,
    w_model: PotentialW,
    context_model: ContextModel,
    e_threshold: torch.Tensor,
    batch_size: int,
    margin: float,
    num_negatives: int,
    lambda_e: float,
    z_min: torch.Tensor,
    z_max: torch.Tensor,
) -> float:
    return eval_w_e_loss(
        dataset,
        z_all,
        v_model,
        w_model,
        context_model,
        e_threshold,
        batch_size,
        margin,
        num_negatives,
        lambda_e,
        z_min,
        z_max,
    )


def train_autoencoder(
    train_ds: SavedDataset,
    val_ds: SavedDataset,
    epochs: int,
    batch_size: int,
    lr: float,
    lambda_xy: float,
    noise_std: float,
    checkpoint: Path,
) -> tuple[Autoencoder, ContextModel]:
    ae = Autoencoder(latent_dim=2, context_dim=DEFAULT_CONTEXT_DIM).to(DEVICE)
    context_model = ContextModel(context_dim=DEFAULT_CONTEXT_DIM).to(DEVICE)
    print(
        f"AE data: {len(train_ds.state_idx)} train transitions, "
        f"{len(val_ds.state_idx)} val transitions"
    )

    weights = transition_sample_weights(train_ds)
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(weights).double(),
        num_samples=len(train_ds.state_idx),
        replacement=True,
    )
    loader = DataLoader(
        ObservationContextDataset(train_ds),
        batch_size=min(batch_size, len(train_ds.state_idx)),
        sampler=sampler,
    )
    opt = torch.optim.Adam(
        list(ae.parameters()) + list(context_model.parameters()), lr=lr
    )
    eval_bs = min(batch_size, 4096)
    best_val = float("inf")
    best_ae_state: dict[str, torch.Tensor] | None = None
    best_ctx_state: dict[str, torch.Tensor] | None = None

    for epoch in range(1, epochs + 1):
        ae.train()
        context_model.train()
        total = 0.0
        for image, context_image, context_action, xy in loader:
            image = image.to(DEVICE)
            context_image = context_image.to(DEVICE)
            context_action = context_action.to(DEVICE)
            xy = xy.to(DEVICE)

            context_latent = context_model(context_image, context_action)
            z, recon = ae(image, context_latent, noise_std)
            recon_loss = F.mse_loss(recon, image)
            xy_loss = F.mse_loss(z, xy-0.5)
            loss = recon_loss + lambda_xy * xy_loss

            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()

        train_loss = total / len(loader)
        val_loss = eval_ae_recon(ae, context_model, val_ds, eval_bs)
        if val_loss < best_val:
            best_val = val_loss
            best_ae_state = clone_state_dict(ae)
            best_ctx_state = clone_state_dict(context_model)
            save_checkpoint(
                checkpoint,
                model=best_ae_state,
                context_model=best_ctx_state,
                best_val=best_val,
            )

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(
                f"[ae] epoch {epoch}/{epochs} "
                f"train={train_loss:.4f}  val={val_loss:.4f}  best_val={best_val:.4f}"
            )

    assert best_ae_state is not None and best_ctx_state is not None
    load_cloned_state_dict(ae, best_ae_state)
    load_cloned_state_dict(context_model, best_ctx_state)
    context_model.eval()
    for p in context_model.parameters():
        p.requires_grad = False

    from visualize import plot_ae_reconstructions

    z_all = encode_states(ae, train_ds)
    plot_ae_reconstructions(
        train_ds,
        ae,
        context_model,
        z_all,
        Path("figures/ae_recon.png"),
        show=False,
    )

    return ae, context_model


@torch.no_grad()
def encode_states(model: Autoencoder, dataset: SavedDataset) -> torch.Tensor:
    model.eval()
    images = (
        torch.from_numpy(dataset.states.astype("float32") / 255.0)
        .permute(0, 3, 1, 2)
        .to(DEVICE)
    )
    return model.encode(images)


def compute_latent_bounds(
    z_all: torch.Tensor, padding: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Axis-aligned box over encoded states, extended by padding on each side."""
    z_min = z_all.min(dim=0).values - padding
    z_max = z_all.max(dim=0).values + padding
    return z_min, z_max


def sample_box_negatives(
    batch_size: int,
    num_negatives: int,
    z_min: torch.Tensor,
    z_max: torch.Tensor,
) -> torch.Tensor:
    """Uniform random z inside the latent bounding box (including padding). num_negatives is the number of negatives to sample."""
    u = torch.rand(batch_size, num_negatives, z_min.size(0), device=z_min.device)
    return z_min + u * (z_max - z_min)


def sample_w_negatives(
    context_latent: torch.Tensor,
    num_negatives: int,
    z_min: torch.Tensor,
    z_max: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Random candidate z in the box; keep the batch context latent unchanged."""
    z_neg = sample_box_negatives(context_latent.size(0), num_negatives, z_min, z_max)
    return z_neg, context_latent


def train_v(
    z_all: torch.Tensor,
    train_state_idx: np.ndarray,
    val_state_idx: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    margin: float,
    neg_padding: float,
    num_negatives: int,
    noise_std: float,
    checkpoint: Path,
) -> PotentialV:
    model = PotentialV().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    train_idx = torch.from_numpy(train_state_idx).long().to(DEVICE)
    eval_bs = min(batch_size, 256)
    z_min, z_max = compute_latent_bounds(z_all, neg_padding)
    best_val = float("inf")
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(1, epochs + 1):
        perm = train_idx[torch.randperm(len(train_idx), device=DEVICE)]
        model.train()
        total = 0.0
        steps = 0
        for start in range(0, len(perm), batch_size):
            idx = perm[start : start + batch_size]
            z_pos = z_all[idx]
            z_pos = z_pos + torch.randn_like(z_pos) * noise_std
            z_neg = sample_box_negatives(len(z_pos), num_negatives, z_min, z_max)

            e_pos = model(z_pos)
            e_neg = model(z_neg)
            loss = F.relu(e_pos - e_neg + margin).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            steps += 1

        train_loss = total / steps
        val_loss = eval_v_loss(
            model, z_all, val_state_idx, eval_bs, margin, num_negatives, z_min, z_max
        )
        if val_loss < best_val:
            best_val = val_loss
            best_state = clone_state_dict(model)
            save_checkpoint(checkpoint, model=best_state, best_val=best_val)

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(
                f"[v] epoch {epoch}/{epochs} "
                f"train={train_loss:.4f}  val={val_loss:.4f}  best_val={best_val:.4f}"
            )

    assert best_state is not None
    model.load_state_dict(best_state)
    return model


def wrong_room_indices(dataset: SavedDataset) -> torch.Tensor:
    """States in the right room (for sanity checks only)."""
    x = dataset.state_meta[:, 0]
    return torch.from_numpy((x > WALL_X_NORM).nonzero()[0]).long()


def train_w(
    train_ds: SavedDataset,
    val_ds: SavedDataset,
    ae: Autoencoder,
    v_model: PotentialV,
    context_model: ContextModel,
    epochs: int,
    batch_size: int,
    lr: float,
    margin: float,
    lambda_e: float,
    neg_padding: float,
    num_negatives: int,
    checkpoint: Path,
    e_checkpoint: Path,
) -> tuple[PotentialW, torch.Tensor]:
    ae.eval()
    context_model.eval()
    for p in ae.parameters():
        p.requires_grad = False
    for p in context_model.parameters():
        p.requires_grad = False
    for p in v_model.parameters():
        p.requires_grad = False
    v_model.eval()

    z_all = encode_states(ae, train_ds)
    z_min, z_max = compute_latent_bounds(z_all, neg_padding)
    loader = DataLoader(
        ContextTransitionDataset(train_ds),
        batch_size=batch_size,
        shuffle=True,
    )
    w_model = PotentialW(context_dim=DEFAULT_CONTEXT_DIM).to(DEVICE)
    init_e = mean_positive_energy(loader, z_all, v_model, w_model, context_model)
    e_param = torch.nn.Parameter(torch.tensor(init_e, device=DEVICE))
    opt = torch.optim.Adam(list(w_model.parameters()) + [e_param], lr=lr)
    eval_bs = min(batch_size, 4096)
    best_val = float("inf")
    best_w: dict[str, torch.Tensor] | None = None
    best_e: torch.Tensor | None = None

    for epoch in range(1, epochs + 1):
        w_model.train()
        total = 0.0
        for next_idx, context_image, context_action in loader:
            next_idx = next_idx.to(DEVICE)
            context_image = context_image.to(DEVICE)
            context_action = context_action.to(DEVICE)

            z_pos = z_all[next_idx]
            with torch.no_grad():
                context_latent = context_model(context_image, context_action)
            z_neg, ctx_latent = sample_w_negatives(context_latent, num_negatives, z_min, z_max)

            with torch.no_grad():
                v_pos = v_model(z_pos)
                v_neg = v_model(z_neg)

            e_pos = v_pos + w_model(z_pos, ctx_latent)
            e_neg = v_neg + w_model(z_neg, ctx_latent)
            loss = threshold_energy_loss(e_pos, e_neg, e_param, margin, lambda_e)

            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()

        train_loss = total / len(loader)
        val_loss = eval_w_e_loss(
            val_ds,
            z_all,
            v_model,
            w_model,
            context_model,
            e_param,
            eval_bs,
            margin,
            num_negatives,
            lambda_e,
            z_min,
            z_max,
        )
        if val_loss < best_val:
            best_val = val_loss
            best_w = clone_state_dict(w_model)
            best_e = e_param.detach().clone()
            save_checkpoint(
                checkpoint,
                w_model=best_w,
                E=best_e.item(),
                best_val=best_val,
            )
            save_checkpoint(e_checkpoint, E=best_e.item(), best_val=best_val)

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(
                f"[w] epoch {epoch}/{epochs} "
                f"train={train_loss:.4f}  val={val_loss:.4f}  "
                f"E={e_param.item():.4f}  best_val={best_val:.4f}"
            )

    assert best_w is not None and best_e is not None
    w_model.load_state_dict(best_w)
    return w_model, best_e


def eval_context_w_loss(
    dataset: SavedDataset,
    z_all: torch.Tensor,
    v_model: PotentialV,
    w_model: PotentialW,
    context_model: ContextModel,
    e_threshold: torch.Tensor,
    batch_size: int,
    margin: float,
    num_negatives: int,
    lambda_e: float,
    z_min: torch.Tensor,
    z_max: torch.Tensor,
) -> float:
    return eval_w_e_loss(
        dataset,
        z_all,
        v_model,
        w_model,
        context_model,
        e_threshold,
        batch_size,
        margin,
        num_negatives,
        lambda_e,
        z_min,
        z_max,
    )


def train_context_with_w(
    train_ds: SavedDataset,
    val_ds: SavedDataset,
    ae: Autoencoder,
    v_model: PotentialV,
    w_model: PotentialW,
    context_model: ContextModel,
    epochs: int,
    batch_size: int,
    lr: float,
    margin: float,
    lambda_e: float,
    lambda_recon: float,
    neg_padding: float,
    num_negatives: int,
    noise_std: float,
    ae_checkpoint: Path,
    w_checkpoint: Path,
    e_checkpoint: Path,
) -> tuple[PotentialW, ContextModel, torch.Tensor]:
    """Jointly fine-tune ContextModel, W, E, and Decoder so c helps reachability and reconstruction."""
    v_model.eval()
    for p in v_model.parameters():
        p.requires_grad = False
    for p in ae.encoder.parameters():
        p.requires_grad = False
    for p in ae.decoder.parameters():
        p.requires_grad = True

    z_all = encode_states(ae, train_ds)
    z_min, z_max = compute_latent_bounds(z_all, neg_padding)
    w_loader = DataLoader(
        ContextTransitionDataset(train_ds),
        batch_size=batch_size,
        shuffle=True,
    )
    weights = transition_sample_weights(train_ds)
    recon_loader = DataLoader(
        ObservationContextDataset(train_ds),
        batch_size=min(batch_size, len(train_ds.state_idx)),
        sampler=WeightedRandomSampler(
            weights=torch.from_numpy(weights).double(),
            num_samples=len(train_ds.state_idx),
            replacement=True,
        ),
    )
    init_e = resolve_init_e(
        w_checkpoint, e_checkpoint, w_loader, z_all, v_model, w_model, context_model
    )
    e_param = torch.nn.Parameter(torch.tensor(init_e, device=DEVICE))
    opt = torch.optim.Adam(
        list(context_model.parameters())
        + list(w_model.parameters())
        + list(ae.decoder.parameters())
        + [e_param],
        lr=lr,
    )
    eval_bs = min(batch_size, 4096)
    best_val = float("inf")
    best_ctx: dict[str, torch.Tensor] | None = None
    best_w: dict[str, torch.Tensor] | None = None
    best_ae: dict[str, torch.Tensor] | None = None
    best_e: torch.Tensor | None = None

    for epoch in range(1, epochs + 1):
        context_model.train()
        w_model.train()
        ae.decoder.train()
        total_w = 0.0
        total_recon = 0.0
        n_steps = max(len(w_loader), len(recon_loader))
        w_iter = cycle(w_loader) if len(w_loader) < n_steps else iter(w_loader)
        recon_iter = (
            cycle(recon_loader) if len(recon_loader) < n_steps else iter(recon_loader)
        )
        for _ in range(n_steps):
            next_idx, context_image, context_action = next(w_iter)
            next_idx = next_idx.to(DEVICE)
            context_image = context_image.to(DEVICE)
            context_action = context_action.to(DEVICE)

            z_pos = z_all[next_idx]
            z_pos = z_pos + torch.randn_like(z_pos) * noise_std
            context_latent = context_model(context_image, context_action)
            z_neg, ctx_latent = sample_w_negatives(context_latent, num_negatives, z_min, z_max)

            with torch.no_grad():
                v_pos = v_model(z_pos)
                v_neg = v_model(z_neg)

            e_pos = v_pos + w_model(z_pos, ctx_latent)
            e_neg = v_neg + w_model(z_neg, ctx_latent)
            w_loss = threshold_energy_loss(e_pos, e_neg, e_param, margin, lambda_e)

            image, recon_context_image, recon_context_action, _xy = next(recon_iter)
            image = image.to(DEVICE)
            recon_context_image = recon_context_image.to(DEVICE)
            recon_context_action = recon_context_action.to(DEVICE)
            with torch.no_grad():
                z = ae.encode(image)
            recon_context_latent = context_model(recon_context_image, recon_context_action)
            recon = ae.decode(z, recon_context_latent)
            recon_loss = F.mse_loss(recon, image)

            loss = w_loss + lambda_recon * recon_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_w += w_loss.item()
            total_recon += recon_loss.item()

        train_w = total_w / n_steps
        train_recon = total_recon / n_steps
        val_w = eval_context_w_loss(
            val_ds,
            z_all,
            v_model,
            w_model,
            context_model,
            e_param,
            eval_bs,
            margin,
            num_negatives,
            lambda_e,
            z_min,
            z_max,
        )
        val_recon = eval_ae_recon(ae, context_model, val_ds, eval_bs)
        val_loss = val_w + lambda_recon * val_recon
        if val_loss < best_val:
            best_val = val_loss
            best_ctx = clone_state_dict(context_model)
            best_w = clone_state_dict(w_model)
            best_ae = clone_state_dict(ae)
            best_e = e_param.detach().clone()
            save_checkpoint(
                ae_checkpoint,
                model=best_ae,
                context_model=best_ctx,
                best_val=best_val,
            )
            save_checkpoint(
                w_checkpoint,
                w_model=best_w,
                E=best_e.item(),
                best_val=best_val,
            )
            save_checkpoint(e_checkpoint, E=best_e.item(), best_val=best_val)

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(
                f"[ctx_w] epoch {epoch}/{epochs} "
                f"train_w={train_w:.4f}  train_recon={train_recon:.4f}  "
                f"val_w={val_w:.4f}  val_recon={val_recon:.4f}  "
                f"E={e_param.item():.4f}  best_val={best_val:.4f}"
            )

    assert best_ctx is not None and best_w is not None and best_ae is not None and best_e is not None
    load_cloned_state_dict(context_model, best_ctx)
    load_cloned_state_dict(w_model, best_w)
    load_cloned_state_dict(ae, best_ae)
    context_model.eval()
    w_model.eval()
    ae.eval()
    print_context_diversity(val_ds, context_model)
    return w_model, context_model, best_e


def train_decoder_only(
    train_ds: SavedDataset,
    val_ds: SavedDataset,
    ae: Autoencoder,
    context_model: ContextModel,
    epochs: int,
    batch_size: int,
    lr: float,
    checkpoint: Path,
) -> Autoencoder:
    """Retrain decoder for the updated context vectors; encoder + context frozen."""
    ae.eval()
    context_model.eval()
    for p in ae.encoder.parameters():
        p.requires_grad = False
    for p in context_model.parameters():
        p.requires_grad = False
    for p in ae.decoder.parameters():
        p.requires_grad = True

    weights = transition_sample_weights(train_ds)
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(weights).double(),
        num_samples=len(train_ds.state_idx),
        replacement=True,
    )
    loader = DataLoader(
        ObservationContextDataset(train_ds),
        batch_size=min(batch_size, len(train_ds.state_idx)),
        sampler=sampler,
    )
    opt = torch.optim.Adam(ae.decoder.parameters(), lr=lr)
    eval_bs = min(batch_size, 4096)
    best_val = float("inf")
    best_decoder: dict[str, torch.Tensor] | None = None

    for epoch in range(1, epochs + 1):
        ae.decoder.train()
        total = 0.0
        for image, context_image, context_action, _xy in loader:
            image = image.to(DEVICE)
            context_image = context_image.to(DEVICE)
            context_action = context_action.to(DEVICE)

            with torch.no_grad():
                z = ae.encode(image)
                context_latent = context_model(context_image, context_action)
            recon = ae.decode(z, context_latent)
            loss = F.mse_loss(recon, image)

            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()

        train_loss = total / len(loader)
        val_loss = eval_ae_recon(ae, context_model, val_ds, eval_bs)
        if val_loss < best_val:
            best_val = val_loss
            best_decoder = clone_state_dict(ae.decoder)
            save_checkpoint(
                checkpoint,
                model=clone_state_dict(ae),
                context_model=clone_state_dict(context_model),
                best_val=best_val,
            )

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(
                f"[decoder] epoch {epoch}/{epochs} "
                f"train={train_loss:.4f}  val={val_loss:.4f}  best_val={best_val:.4f}"
            )

    assert best_decoder is not None
    load_cloned_state_dict(ae.decoder, best_decoder)
    ae.eval()
    return ae


@torch.no_grad()
def print_context_diversity(dataset: SavedDataset, context_model: ContextModel) -> None:
    """Report whether c varies with anchor side (not just action label)."""
    images = (
        torch.from_numpy(dataset.states.astype("float32") / 255.0)
        .permute(0, 3, 1, 2)
        .to(DEVICE)
    )
    noop = torch.tensor([ACTION_TO_IDX["noop"]], device=DEVICE)
    left_idx = right_idx = None
    for i in range(len(dataset.state_meta)):
        x, door = float(dataset.state_meta[i, 0]), float(dataset.state_meta[i, 2])
        if door > 0.5:
            continue
        if x < WALL_X_NORM and left_idx is None:
            left_idx = i
        if x > WALL_X_NORM and right_idx is None:
            right_idx = i
        if left_idx is not None and right_idx is not None:
            break
    if left_idx is None or right_idx is None:
        print("[ctx_w] could not find left/right closed anchors for diversity check")
        return
    c_left = context_model(images[left_idx : left_idx + 1], noop).squeeze().cpu().tolist()
    c_right = context_model(images[right_idx : right_idx + 1], noop).squeeze().cpu().tolist()
    delta = float(
        torch.dist(
            context_model(images[left_idx : left_idx + 1], noop).squeeze(),
            context_model(images[right_idx : right_idx + 1], noop).squeeze(),
        ).item()
    )
    print(
        f"[ctx_w] c(noop): left={c_left}  right={c_right}  L2 delta={delta:.4f}"
    )


@torch.no_grad()
def total_energy(
    v_model: PotentialV,
    w_model: PotentialW,
    z: torch.Tensor,
    context_latent: torch.Tensor,
) -> torch.Tensor:
    return v_model(z) + w_model(z, context_latent)


def train_e(
    train_ds: SavedDataset,
    val_ds: SavedDataset,
    ae: Autoencoder,
    v_model: PotentialV,
    w_model: PotentialW,
    context_model: ContextModel,
    epochs: int,
    batch_size: int,
    lr: float,
    margin: float,
    lambda_e: float,
    neg_padding: float,
    num_negatives: int,
    checkpoint: Path,
    w_checkpoint: Path,
    e_checkpoint: Path,
) -> torch.Tensor:
    """Optional E-only refine with W frozen (same threshold loss as w/ctx_w)."""
    ae.eval()
    v_model.eval()
    w_model.eval()
    context_model.eval()
    for module in (ae, v_model, w_model, context_model):
        for p in module.parameters():
            p.requires_grad = False

    z_all = encode_states(ae, train_ds)
    z_min, z_max = compute_latent_bounds(z_all, neg_padding)
    loader = DataLoader(
        ContextTransitionDataset(train_ds),
        batch_size=batch_size,
        shuffle=True,
    )
    eval_bs = min(batch_size, 4096)
    init_e = load_energy_e(w_checkpoint, e_checkpoint).item()
    e_param = torch.nn.Parameter(torch.tensor(init_e, device=DEVICE))
    opt = torch.optim.Adam([e_param], lr=lr)
    best_val = float("inf")
    best_e = e_param.detach().clone()

    for epoch in range(1, epochs + 1):
        total = 0.0
        for next_idx, context_image, context_action in loader:
            next_idx = next_idx.to(DEVICE)
            context_image = context_image.to(DEVICE)
            context_action = context_action.to(DEVICE)

            with torch.no_grad():
                z_pos = z_all[next_idx]
                ctx_latent = context_model(context_image, context_action)
                z_neg, ctx_batch = sample_w_negatives(ctx_latent, num_negatives, z_min, z_max)
                e_pos = total_energy(v_model, w_model, z_pos, ctx_batch)
                e_neg = total_energy(v_model, w_model, z_neg, ctx_batch)

            loss = threshold_energy_loss(e_pos, e_neg, e_param, margin, lambda_e)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()

        train_loss = total / len(loader)
        val_loss = eval_e_loss(
            val_ds,
            z_all,
            v_model,
            w_model,
            context_model,
            e_param,
            eval_bs,
            margin,
            num_negatives,
            lambda_e,
            z_min,
            z_max,
        )

        if val_loss < best_val:
            best_val = val_loss
            best_e = e_param.detach().clone()
            save_checkpoint(checkpoint, E=best_e.item(), best_val=best_val)

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(
                f"[e] epoch {epoch}/{epochs} "
                f"train={train_loss:.4f}  val={val_loss:.4f}  "
                f"E={e_param.item():.4f}  best_val={best_val:.4f}"
            )

    return best_e


@torch.no_grad()
def sanity_check(
    dataset: SavedDataset,
    ae: Autoencoder,
    v_model: PotentialV,
    w_model: PotentialW,
    context_model: ContextModel,
) -> None:
    z_all = encode_states(ae, dataset)
    images = (
        torch.from_numpy(dataset.states.astype("float32") / 255.0)
        .permute(0, 3, 1, 2)
        .to(DEVICE)
    )
    wrong = wrong_room_indices(dataset)
    left_closed = torch.from_numpy(
        (
            (dataset.state_meta[:, 0] < WALL_X_NORM)
            & (dataset.state_meta[:, 2] < 0.5)
        ).nonzero()[0]
    ).long()
    left_open = torch.from_numpy(
        (
            (dataset.state_meta[:, 0] < WALL_X_NORM)
            & (dataset.state_meta[:, 2] > 0.5)
        ).nonzero()[0]
    ).long()
    if len(wrong) == 0 or len(left_closed) == 0 or len(left_open) == 0:
        print("[check] need right-room, left-closed, and left-open states")
        return

    z_next = z_all[wrong[0]].unsqueeze(0)
    noop = torch.tensor([ACTION_TO_IDX["noop"]], device=DEVICE)
    open_door = torch.tensor([ACTION_TO_IDX["open_door"]], device=DEVICE)
    c_closed = context_model(images[left_closed[0]].unsqueeze(0), noop)
    c_open = context_model(images[left_open[0]].unsqueeze(0), open_door)

    e_closed = v_model(z_next) + w_model(z_next, c_closed)
    e_open = v_model(z_next) + w_model(z_next, c_open)
    print(
        "[check] right-room candidate "
        f"context_door_closed={e_closed.item():.3f}  context_door_open={e_open.item():.3f}  "
        f"c_closed={c_closed.squeeze().tolist()}  c_open={c_open.squeeze().tolist()}"
    )
    if e_closed.item() <= e_open.item():
        print("[check] warning: expected higher energy under closed-door context")


@torch.no_grad()
def sanity_check_e(
    dataset: SavedDataset,
    ae: Autoencoder,
    v_model: PotentialV,
    w_model: PotentialW,
    context_model: ContextModel,
    num_negatives: int,
    e_threshold: float,
) -> None:
    z_all = encode_states(ae, dataset)
    z_min, z_max = compute_latent_bounds(z_all, padding=0.1)
    loader = DataLoader(ContextTransitionDataset(dataset), batch_size=256)

    pos_vals: list[torch.Tensor] = []
    neg_vals: list[torch.Tensor] = []
    for next_idx, context_image, context_action in loader:
        next_idx = next_idx.to(DEVICE)
        context_image = context_image.to(DEVICE)
        context_action = context_action.to(DEVICE)
        z_pos = z_all[next_idx]
        ctx_latent = context_model(context_image, context_action)
        z_neg, ctx_batch = sample_w_negatives(ctx_latent, num_negatives, z_min, z_max)
        pos_vals.append(total_energy(v_model, w_model, z_pos, ctx_batch))
        neg_vals.append(total_energy(v_model, w_model, z_neg, ctx_batch))

    e_pos = torch.cat(pos_vals)
    e_neg = torch.cat(neg_vals)
    print(
        f"[check-e] E={e_threshold:.3f}  "
        f"pos mean={e_pos.mean().item():.3f} max={e_pos.max().item():.3f}  "
        f"neg mean={e_neg.mean().item():.3f} min={e_neg.min().item():.3f}"
    )
    if e_pos.max().item() >= e_threshold:
        print("[check-e] warning: some positive samples exceed E")
    if e_neg.min().item() <= e_threshold:
        print("[check-e] warning: some negative samples fall below E")


@torch.no_grad()
def eval_wm_mse(
    ae: Autoencoder,
    dynamics: LatentDynamics,
    context_model: ContextModel,
    dataset: SavedDataset,
    batch_size: int,
) -> float:
    if len(dataset.next_idx) == 0:
        return float("nan")
    ae.eval()
    dynamics.eval()
    context_model.eval()
    loader = DataLoader(ImageActionTransitionDataset(dataset), batch_size=batch_size)
    total = 0.0
    n = 0
    for image, action, next_image, context_image, context_action in loader:
        image = image.to(DEVICE)
        action = action.to(DEVICE)
        next_image = next_image.to(DEVICE)
        context_image = context_image.to(DEVICE)
        context_action = context_action.to(DEVICE)
        z_current = ae.encode(image)
        z_target = ae.encode(next_image)
        context_latent = context_model(context_image, context_action)
        z_pred = dynamics(z_current, context_latent, action)
        total += F.mse_loss(z_pred, z_target, reduction="sum").item()
        n += z_target.size(0)
    return total / n


@torch.no_grad()
def eval_wm_transition_errors(
    ae: Autoencoder,
    dynamics: LatentDynamics,
    context_model: ContextModel,
    v_model: PotentialV,
    w_model: PotentialW,
    e_threshold: torch.Tensor,
    dataset: SavedDataset,
    z_all: torch.Tensor,
    batch_size: int,
) -> dict[str, float]:
    if len(dataset.next_idx) == 0:
        return {}

    ae.eval()
    dynamics.eval()
    v_model.eval()
    w_model.eval()
    context_model.eval()
    loader = DataLoader(
        ImageActionTransitionDataset(dataset),
        batch_size=batch_size,
    )
    actual_next = torch.from_numpy(dataset.next_idx).long().to(DEVICE)

    n = 0
    raw_errors = 0
    constrained_errors = 0
    reject_pred = 0
    false_reject = 0
    raw_invalid = 0
    constrained_invalid = 0

    offset = 0
    for image, action, next_image, context_image, context_action in loader:
        batch_size_actual = image.size(0)
        image = image.to(DEVICE)
        action = action.to(DEVICE)
        next_image = next_image.to(DEVICE)
        context_image = context_image.to(DEVICE)
        context_action = context_action.to(DEVICE)
        target_idx = actual_next[offset : offset + batch_size_actual]
        offset += batch_size_actual

        z_current = ae.encode(image)
        z_out, z_pred, allowed_pred = constrained_next_latent(
            dynamics,
            context_model,
            v_model,
            w_model,
            e_threshold,
            z_current,
            context_image,
            context_action,
            action,
        )
        pred_raw_idx = nearest_state_indices(z_pred, z_all)
        pred_out_idx = nearest_state_indices(z_out, z_all)

        raw_errors += (pred_raw_idx != target_idx).sum().item()
        constrained_errors += (pred_out_idx != target_idx).sum().item()
        reject_pred += (~allowed_pred).sum().item()

        context_latent = context_model(context_image, context_action)
        z_next = ae.encode(next_image)
        allowed_true = is_energy_allowed(
            v_model, w_model, z_next, context_latent, e_threshold
        )
        false_reject += ((~allowed_pred) & allowed_true).sum().item()

        raw_invalid += (~allowed_pred).sum().item()
        pred_out_allowed = is_energy_allowed(
            v_model, w_model, z_out, context_latent, e_threshold
        )
        constrained_invalid += (~pred_out_allowed).sum().item()
        n += batch_size_actual

    return {
        "unconstrained_error_rate": raw_errors / n,
        "constrained_error_rate": constrained_errors / n,
        "reject_rate": reject_pred / n,
        "false_reject_rate": false_reject / n,
        "unconstrained_invalid_rate": raw_invalid / n,
        "constrained_invalid_rate": constrained_invalid / n,
    }


def train_wm(
    train_ds: SavedDataset,
    val_ds: SavedDataset,
    ae: Autoencoder,
    v_model: PotentialV,
    w_model: PotentialW,
    context_model: ContextModel,
    e_threshold: torch.Tensor,
    epochs: int,
    batch_size: int,
    lr: float,
    checkpoint: Path,
) -> LatentDynamics:
    ae.eval()
    v_model.eval()
    w_model.eval()
    context_model.eval()
    for module in (ae, v_model, w_model, context_model):
        for p in module.parameters():
            p.requires_grad = False

    z_all = encode_states(ae, train_ds)
    loader = DataLoader(
        ImageActionTransitionDataset(train_ds),
        batch_size=batch_size,
        shuffle=True,
    )
    dynamics = LatentDynamics().to(DEVICE)
    opt = torch.optim.Adam(dynamics.parameters(), lr=lr)
    eval_bs = min(batch_size, 4096)
    best_val = float("inf")
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(1, epochs + 1):
        dynamics.train()
        total = 0.0
        for image, action, next_image, context_image, context_action in loader:
            image = image.to(DEVICE)
            action = action.to(DEVICE)
            next_image = next_image.to(DEVICE)
            context_image = context_image.to(DEVICE)
            context_action = context_action.to(DEVICE)

            with torch.no_grad():
                z_current = ae.encode(image)
                z_target = ae.encode(next_image)
                context_latent = context_model(context_image, context_action)
            z_pred = dynamics(z_current, context_latent, action)
            loss = F.mse_loss(z_pred, z_target)

            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()

        train_loss = total / len(loader)
        val_loss = eval_wm_mse(
            ae, dynamics, context_model, val_ds, eval_bs
        )
        if val_loss < best_val:
            best_val = val_loss
            best_state = clone_state_dict(dynamics)
            save_checkpoint(checkpoint, model=best_state, best_val=best_val)

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(
                f"[wm] epoch {epoch}/{epochs} "
                f"train={train_loss:.6f}  val={val_loss:.6f}  best_val={best_val:.6f}"
            )

    assert best_state is not None
    dynamics.load_state_dict(best_state)
    stats = eval_wm_transition_errors(
        ae,
        dynamics,
        context_model,
        v_model,
        w_model,
        e_threshold,
        val_ds,
        z_all,
        eval_bs,
    )
    print(
        "[wm] val "
        f"unconstrained_err={stats['unconstrained_error_rate']:.3f}  "
        f"constrained_err={stats['constrained_error_rate']:.3f}  "
        f"reject={stats['reject_rate']:.3f}  "
        f"false_reject={stats['false_reject_rate']:.3f}  "
        f"unconstrained_invalid={stats['unconstrained_invalid_rate']:.3f}  "
        f"constrained_invalid={stats['constrained_invalid_rate']:.3f}"
    )

    return dynamics


@torch.no_grad()
def sanity_check_wm(
    dataset: SavedDataset,
    ae: Autoencoder,
    v_model: PotentialV,
    w_model: PotentialW,
    context_model: ContextModel,
    dynamics: LatentDynamics,
    e_threshold: float,
) -> None:
    z_all = encode_states(ae, dataset)
    stats = eval_wm_transition_errors(
        ae,
        dynamics,
        context_model,
        v_model,
        w_model,
        torch.tensor(e_threshold, device=DEVICE),
        dataset,
        z_all,
        batch_size=256,
    )
    print(
        f"[check-wm] unconstrained_err={stats['unconstrained_error_rate']:.3f}  "
        f"constrained_err={stats['constrained_error_rate']:.3f}  "
        f"reject={stats['reject_rate']:.3f}  "
        f"false_reject={stats['false_reject_rate']:.3f}  "
        f"unconstrained_invalid={stats['unconstrained_invalid_rate']:.3f}  "
        f"constrained_invalid={stats['constrained_invalid_rate']:.3f}"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--stage",
        choices=["ae", "v", "w", "ctx_w", "decoder", "e", "wm", "all"],
        default="all",
    )
    p.add_argument("--data", type=Path, default=Path("data/unlocked_door/train.npz"))
    p.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    p.add_argument("--ae-epochs", type=int, default=50)
    p.add_argument("--v-epochs", type=int, default=8000)
    p.add_argument("--w-epochs", type=int, default=40)
    p.add_argument("--ctx-w-epochs", type=int, default=100)
    p.add_argument("--decoder-epochs", type=int, default=50)
    p.add_argument("--e-epochs", type=int, default=100)
    p.add_argument("--wm-epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lambda-xy", type=float, default=10.0)
    p.add_argument("--margin", type=float, default=1.0)
    p.add_argument(
        "--neg-padding",
        type=float,
        default=0.1,
        help="extend latent bounding box for V/W negatives on each side",
    )
    p.add_argument("--num-negatives", type=int, default=16)
    p.add_argument("--lambda-e", type=float, default=0.0, help="aux loss weight to minimize E")
    p.add_argument(
        "--lambda-recon",
        type=float,
        default=1000.0,
        help="decoder reconstruction weight during ctx_w joint training",
    )
    p.add_argument("--noise-std", type=float, default=0.01)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    full_ds = load_dataset(str(args.data))
    train_ds, val_ds = split_dataset(full_ds, args.val_fraction, args.seed)
    print_split_summary(train_ds, val_ds)

    train_state_idx = unique_referenced_states(train_ds)
    val_state_idx = unique_referenced_states(val_ds)
    ckpt_dir = args.checkpoint_dir

    ae_path = ckpt_dir / "ae.pt"
    v_path = ckpt_dir / "v.pt"
    w_path = ckpt_dir / "w.pt"
    e_path = ckpt_dir / "e.pt"
    wm_path = ckpt_dir / "wm.pt"

    ae = None
    v_model = None
    w_model = None
    context_model = None
    e_value = None

    if args.stage in {"ae", "all"}:
        ae, context_model = train_autoencoder(
            train_ds,
            val_ds,
            epochs=args.ae_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            lambda_xy=args.lambda_xy,
            noise_std=args.noise_std,
            checkpoint=ae_path,
        )

    if args.stage in {"v", "w", "ctx_w", "decoder", "e", "wm", "all"}:
        if ae is None or context_model is None:
            ae, context_model = load_ae_and_context(ae_path)
        z_all = encode_states(ae, train_ds)
        if args.stage in {"v", "all"}:
            v_model = train_v(
                z_all,
                train_state_idx,
                val_state_idx,
                epochs=args.v_epochs,
                batch_size=min(args.batch_size, len(train_state_idx)),
                lr=args.lr,
                margin=args.margin,
                neg_padding=args.neg_padding,
                num_negatives=args.num_negatives,
                noise_std=args.noise_std,
                checkpoint=v_path,
            )

    if args.stage in {"w", "all"}:
        if ae is None or context_model is None:
            ae, context_model = load_ae_and_context(ae_path)
        if v_model is None:
            ckpt = load_checkpoint(v_path)
            v_model = PotentialV().to(DEVICE)
            v_model.load_state_dict(ckpt["model"])
        w_model, e_value = train_w(
            train_ds,
            val_ds,
            ae,
            v_model,
            context_model,
            epochs=args.w_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            margin=args.margin,
            lambda_e=args.lambda_e,
            neg_padding=args.neg_padding,
            num_negatives=args.num_negatives,
            checkpoint=w_path,
            e_checkpoint=e_path,
        )
        sanity_check(val_ds, ae, v_model, w_model, context_model)
        if args.stage == "w":
            ae = train_decoder_only(
                train_ds,
                val_ds,
                ae,
                context_model,
                epochs=args.decoder_epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                checkpoint=ae_path,
            )

    if args.stage in {"ctx_w", "all"}:
        if ae is None or context_model is None:
            ae, context_model = load_ae_and_context(ae_path)
        if v_model is None:
            ckpt = load_checkpoint(v_path)
            v_model = PotentialV().to(DEVICE)
            v_model.load_state_dict(ckpt["model"])
        if w_model is None:
            if w_path.exists():
                w_model = load_w_model(w_path)
            else:
                w_model = PotentialW(context_dim=DEFAULT_CONTEXT_DIM).to(DEVICE)
        w_model, context_model, e_value = train_context_with_w(
            train_ds,
            val_ds,
            ae,
            v_model,
            w_model,
            context_model,
            epochs=args.ctx_w_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            margin=args.margin,
            lambda_e=args.lambda_e,
            lambda_recon=args.lambda_recon,
            neg_padding=args.neg_padding,
            num_negatives=args.num_negatives,
            noise_std=args.noise_std,
            ae_checkpoint=ae_path,
            w_checkpoint=w_path,
            e_checkpoint=e_path,
        )
        sanity_check(val_ds, ae, v_model, w_model, context_model)

    if args.stage == "decoder":
        if ae is None or context_model is None:
            ae, context_model = load_ae_and_context(ae_path)
        ae = train_decoder_only(
            train_ds,
            val_ds,
            ae,
            context_model,
            epochs=args.decoder_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            checkpoint=ae_path,
        )

    if args.stage in {"e", "all"}:
        if ae is None or context_model is None:
            ae, context_model = load_ae_and_context(ae_path)
        if v_model is None:
            ckpt = load_checkpoint(v_path)
            v_model = PotentialV().to(DEVICE)
            v_model.load_state_dict(ckpt["model"])
        if w_model is None:
            w_model = load_w_model(w_path)
        e_value = train_e(
            train_ds,
            val_ds,
            ae,
            v_model,
            w_model,
            context_model,
            epochs=args.e_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            margin=args.margin,
            lambda_e=args.lambda_e,
            neg_padding=args.neg_padding,
            num_negatives=args.num_negatives,
            checkpoint=e_path,
            w_checkpoint=w_path,
            e_checkpoint=e_path,
        )
        sanity_check_e(val_ds, ae, v_model, w_model, context_model, args.num_negatives, e_value.item())

    if args.stage in {"wm", "all"}:
        if ae is None or context_model is None:
            ae, context_model = load_ae_and_context(ae_path)
        if v_model is None:
            ckpt = load_checkpoint(v_path)
            v_model = PotentialV().to(DEVICE)
            v_model.load_state_dict(ckpt["model"])
        if w_model is None:
            w_model = load_w_model(w_path)
        if e_value is None:
            e_value = load_energy_e(w_path, e_path)
        dynamics = train_wm(
            train_ds,
            val_ds,
            ae,
            v_model,
            w_model,
            context_model,
            e_value,
            epochs=args.wm_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            checkpoint=wm_path,
        )
        sanity_check_wm(
            val_ds,
            ae,
            v_model,
            w_model,
            context_model,
            dynamics,
            e_value.item(),
        )


if __name__ == "__main__":
    main()
