"""
Visualize learned energy landscapes over the 2D latent space.

For each context case (left/right room × door closed/open), plots 3D surfaces:
  V(z), W(z|context), V+W, V+W allowed (≤ E), and a 2D allowed-region map.
  Also writes probabilities_3d.png with exp(−energy) on the same layout.

Example:
  python experiments/unlocked_door_symbolic/visualize.py
  python experiments/unlocked_door_symbolic/visualize.py --show
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from dpwm.data.load import ContextTransitionDataset, SavedDataset, load_dataset
from dpwm.envs.unlocked_door import ACTION_TO_IDX, UnlockedDoorEnv
from dpwm.models.autoencoder import Autoencoder
from dpwm.models.context_model import ContextModel, infer_context_dim, infer_w_context_dim
from dpwm.models.ebm import PotentialV, PotentialW

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WALL_X_NORM = UnlockedDoorEnv.wall_x / (UnlockedDoorEnv.width - 1)


@dataclass(frozen=True)
class ContextCase:
    side: str
    door_open: bool
    label: str


CONTEXT_CASES = (
    ContextCase("left", False, "Left room · door closed"),
    ContextCase("left", True, "Left room · door open"),
    ContextCase("right", False, "Right room · door closed"),
    ContextCase("right", True, "Right room · door open"),
)


@torch.no_grad()
def encode_all_states(ae: Autoencoder, dataset: SavedDataset) -> torch.Tensor:
    images = (
        torch.from_numpy(dataset.states.astype(np.float32) / 255.0)
        .permute(0, 3, 1, 2)
        .to(DEVICE)
    )
    return ae.encode(images)


def load_models(
    checkpoint_dir: Path,
    *,
    require_energy_models: bool = True,
) -> tuple[Autoencoder, PotentialV | None, PotentialW | None, ContextModel]:
    ae_ckpt = torch.load(checkpoint_dir / "ae.pt", map_location=DEVICE, weights_only=False)
    context_dim = infer_context_dim(context_model_state=ae_ckpt["context_model"])
    print(f"ae context_dim={context_dim}")

    ae = Autoencoder(latent_dim=2, context_dim=context_dim).to(DEVICE)
    context_model = ContextModel(context_dim=context_dim).to(DEVICE)
    ae.load_state_dict(ae_ckpt["model"])
    context_model.load_state_dict(ae_ckpt["context_model"])
    ae.eval()
    context_model.eval()

    v_path = checkpoint_dir / "v.pt"
    w_path = checkpoint_dir / "w.pt"
    if not require_energy_models or not v_path.exists() or not w_path.exists():
        return ae, None, None, context_model

    w_ckpt = torch.load(w_path, map_location=DEVICE, weights_only=False)
    w_key = "w_model" if "w_model" in w_ckpt else "model"
    w_context_dim = infer_w_context_dim(w_ckpt[w_key])
    if w_context_dim != context_dim:
        raise RuntimeError(
            f"Checkpoint mismatch: ae.pt context_dim={context_dim} but w.pt expects "
            f"context_dim={w_context_dim}. Retrain W and downstream after AE:\n"
            f"  python experiments/unlocked_door_symbolic/train.py --stage w\n"
            f"  (or --stage ctx_w decoder e wm)\n"
            f"For AE recon only: python .../visualize.py --skip-potentials"
        )

    v_model = PotentialV().to(DEVICE)
    w_model = PotentialW(context_dim=context_dim).to(DEVICE)
    v_model.load_state_dict(torch.load(v_path, map_location=DEVICE, weights_only=False)["model"])
    w_model.load_state_dict(w_ckpt[w_key])
    v_model.eval()
    w_model.eval()
    return ae, v_model, w_model, context_model


def find_context_state_idx(dataset: SavedDataset, side: str, door_open: bool) -> int:
    meta = dataset.state_meta
    for i in range(len(meta)):
        x, door = float(meta[i, 0]), float(meta[i, 2])
        in_left = x < WALL_X_NORM
        if side == "left" and not in_left:
            continue
        if side == "right" and in_left:
            continue
        if (door > 0.5) != door_open:
            continue
        return i
    raise ValueError(f"No state found for context: {side=}, {door_open=}")


def latent_grid_bounds(
    z_all: torch.Tensor,
    neg_padding: float,
    outer_padding: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match train.py: neg box = states ± neg_padding; plot can extend further."""
    z_min = z_all.min(dim=0).values - neg_padding - outer_padding
    z_max = z_all.max(dim=0).values + neg_padding + outer_padding
    return z_min, z_max


def latent_grid(
    z_all: torch.Tensor,
    resolution: int,
    neg_padding: float,
    outer_padding: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, torch.Tensor, torch.Tensor, torch.Tensor]:
    z_min, z_max = latent_grid_bounds(z_all, neg_padding, outer_padding)
    z0 = np.linspace(z_min[0].item(), z_max[0].item(), resolution)
    z1 = np.linspace(z_min[1].item(), z_max[1].item(), resolution)
    g0, g1 = np.meshgrid(z0, z1)
    z_flat = torch.from_numpy(np.stack([g0.ravel(), g1.ravel()], axis=1)).float().to(DEVICE)
    neg_z_min = z_all.min(dim=0).values - neg_padding
    neg_z_max = z_all.max(dim=0).values + neg_padding
    return g0, g1, z_flat, neg_z_min, neg_z_max


@torch.no_grad()
def compute_energy_surfaces(
    z_flat: torch.Tensor,
    context_latent: torch.Tensor,
    v_model: PotentialV,
    w_model: PotentialW,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    v_list: list[torch.Tensor] = []
    w_list: list[torch.Tensor] = []
    ctx_batch = context_latent.expand(batch_size, -1)

    for start in range(0, len(z_flat), batch_size):
        z_batch = z_flat[start : start + batch_size]
        ctx = ctx_batch[: z_batch.size(0)]
        v_list.append(v_model(z_batch))
        w_list.append(w_model(z_batch, ctx))

    v = torch.cat(v_list).cpu().numpy()
    w = torch.cat(w_list).cpu().numpy()
    total = v + w
    return v, w, total


@torch.no_grad()
def estimate_threshold(
    dataset: SavedDataset,
    z_all: torch.Tensor,
    v_model: PotentialV,
    w_model: PotentialW,
    context_model: ContextModel,
) -> float:
    """Energy budget E: percentile of V+W on reachable transitions with context."""
    loader = torch.utils.data.DataLoader(
        ContextTransitionDataset(dataset),
        batch_size=4096,
    )
    totals: list[torch.Tensor] = []
    for next_idx, context_image, context_action in loader:
        next_idx = next_idx.to(DEVICE)
        context_image = context_image.to(DEVICE)
        context_action = context_action.to(DEVICE)
        z_next = z_all[next_idx]
        ctx_latent = context_model(context_image, context_action)
        totals.append(v_model(z_next) + w_model(z_next, ctx_latent))
    total = torch.cat(totals).cpu().numpy()
    return float(np.percentile(total, 90))


def load_energy_threshold(
    checkpoint_dir: Path,
    dataset: SavedDataset,
    z_all: torch.Tensor,
    v_model: PotentialV,
    w_model: PotentialW,
    context_model: ContextModel,
) -> float:
    e_path = checkpoint_dir / "e.pt"
    if e_path.exists():
        ckpt = torch.load(e_path, map_location="cpu", weights_only=False)
        e_value = float(ckpt["E"])
        print(f"loaded learned E={e_value:.3f} from {e_path}")
        return e_value
    e_value = estimate_threshold(dataset, z_all, v_model, w_model, context_model)
    print(f"no e.pt found; estimated E={e_value:.3f} from data (90th percentile)")
    return e_value


def plot_surface(
    ax,
    g0: np.ndarray,
    g1: np.ndarray,
    values: np.ndarray,
    title: str,
    *,
    floor: float | None = None,
    zlabel: str = "energy",
) -> None:
    v_grid = values.reshape(g0.shape)
    if floor is not None:
        ax.plot_surface(
            g0,
            g1,
            np.full_like(v_grid, floor),
            color="0.92",
            linewidth=0,
            antialiased=False,
            alpha=0.35,
            shade=False,
        )
    ax.plot_surface(
        g0,
        g1,
        v_grid,
        cmap="viridis",
        linewidth=0,
        antialiased=True,
        alpha=0.92,
    )
    ax.set_xlabel("z₀")
    ax.set_ylabel("z₁")
    ax.set_zlabel(zlabel)
    ax.set_title(title, fontsize=10)
    ax.view_init(elev=35, azim=-60)


def to_probability(values: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore"):
        return np.exp(-values)


def allowed_energy(total: np.ndarray, e_threshold: float) -> np.ndarray:
    """V+W on allowed cells only (total <= E); forbidden cells are NaN."""
    out = total.copy()
    out[total > e_threshold] = np.nan
    return out


def draw_neg_box_2d(
    ax,
    neg_z_min: torch.Tensor,
    neg_z_max: torch.Tensor,
) -> None:
    x0, y0 = neg_z_min[0].item(), neg_z_min[1].item()
    w = neg_z_max[0].item() - x0
    h = neg_z_max[1].item() - y0
    ax.add_patch(
        mpatches.Rectangle(
            (x0, y0),
            w,
            h,
            fill=False,
            edgecolor="#333333",
            linestyle="--",
            linewidth=1.2,
            zorder=4,
        )
    )


def plot_allowed_region_2d(
    ax,
    g0: np.ndarray,
    g1: np.ndarray,
    total: np.ndarray,
    e_threshold: float,
    z_all: torch.Tensor,
    neg_z_min: torch.Tensor,
    neg_z_max: torch.Tensor,
) -> None:
    allowed = (total <= e_threshold).reshape(g0.shape)
    rgba = np.zeros((*allowed.shape, 4))
    rgba[allowed] = (0.45, 0.82, 0.45, 0.45)
    rgba[~allowed] = (0.92, 0.92, 0.92, 0.85)
    ax.imshow(
        rgba,
        origin="lower",
        extent=(g0.min(), g0.max(), g1.min(), g1.max()),
        aspect="equal",
    )
    z_np = z_all.cpu().numpy()
    ax.scatter(
        z_np[:, 0],
        z_np[:, 1],
        s=8,
        c="#888888",
        alpha=0.5,
        linewidths=0,
        zorder=2,
    )
    ax.set_xlabel("z₀")
    ax.set_ylabel("z₁")
    ax.set_title(f"V + W ≤ E ({e_threshold:.2f})", fontsize=10)
    draw_neg_box_2d(ax, neg_z_min, neg_z_max)


def scatter_states(ax, z_all: torch.Tensor, g0: np.ndarray, g1: np.ndarray, floor: float) -> None:
    z = z_all.cpu().numpy()
    ax.scatter(z[:, 0], z[:, 1], floor, c="red", s=8, alpha=0.7, depthshade=False)


def visualize(
    dataset: SavedDataset,
    ae: Autoencoder,
    v_model: PotentialV,
    w_model: PotentialW,
    context_model: ContextModel,
    *,
    checkpoint_dir: Path,
    resolution: int,
    batch_size: int,
    neg_padding: float,
    outer_padding: float,
    output: Path,
    show: bool,
    as_probability: bool = False,
) -> None:
    z_all = encode_all_states(ae, dataset)
    images = (
        torch.from_numpy(dataset.states.astype(np.float32) / 255.0)
        .permute(0, 3, 1, 2)
        .to(DEVICE)
    )
    g0, g1, z_flat, neg_z_min, neg_z_max = latent_grid(
        z_all, resolution, neg_padding, outer_padding
    )
    e_threshold = load_energy_threshold(
        checkpoint_dir, dataset, z_all, v_model, w_model, context_model
    )

    if as_probability:
        suptitle = (
            f"Learned probabilities over latent space  "
            f"(p ∝ e^{{-E}}, gate E = {e_threshold:.3f})"
        )
        col_titles = (
            r"$e^{-V(z)}$",
            r"$e^{-W(z \mid c)}$",
            r"$e^{-(V+W)}$",
            r"$e^{-(V+W)}$ allowed ($\leq E$)",
            r"$V + W \leq E$ (2D allowed region)",
        )
        zlabel = r"$e^{-E}$"
    else:
        suptitle = (
            f"Learned potentials over latent space  (energy threshold E = {e_threshold:.3f})"
        )
        col_titles = (
            "V(z)",
            "W(z | context)",
            "V + W",
            "V+W allowed (≤ E)",
            "V+W ≤ E (2D allowed region)",
        )
        zlabel = "energy"

    n_cols = 5
    fig = plt.figure(figsize=(25, 16))
    fig.suptitle(suptitle, fontsize=14, y=0.98)

    for row, case in enumerate(CONTEXT_CASES):
        ctx_idx = find_context_state_idx(dataset, case.side, case.door_open)
        ctx_action_name = "open_door" if case.door_open else "noop"
        ctx_action = torch.tensor(
            [ACTION_TO_IDX[ctx_action_name]], device=DEVICE, dtype=torch.long
        )
        context_latent = context_model(images[ctx_idx : ctx_idx + 1], ctx_action)

        v, w, total = compute_energy_surfaces(
            z_flat, context_latent, v_model, w_model, batch_size
        )
        allowed = allowed_energy(total, e_threshold)
        surfaces = (v, w, total, allowed)
        if as_probability:
            surfaces = tuple(to_probability(s) for s in surfaces)

        finite_mins = [s.min() for s in surfaces if np.isfinite(s).any()]
        if as_probability:
            floor = max(0.0, float(np.min(finite_mins)) * 0.9 - 1e-6)
        else:
            floor = float(np.min(finite_mins) - 0.5)

        for col in range(n_cols):
            if col < 4:
                values = surfaces[col]
                ax = fig.add_subplot(len(CONTEXT_CASES), n_cols, row * n_cols + col + 1, projection="3d")
                title = col_titles[col] if col > 0 else f"{col_titles[col]}\n(same for all contexts)"
                plot_surface(
                    ax,
                    g0,
                    g1,
                    values,
                    title,
                    floor=floor if col == 3 else None,
                    zlabel=zlabel,
                )
                scatter_states(ax, z_all, g0, g1, floor)
            else:
                ax = fig.add_subplot(len(CONTEXT_CASES), n_cols, row * n_cols + col + 1)
                plot_allowed_region_2d(
                    ax, g0, g1, total, e_threshold, z_all, neg_z_min, neg_z_max
                )

            if col == 0:
                ax.text2D(
                    -0.05,
                    0.5,
                    case.label,
                    transform=ax.transAxes,
                    fontsize=11,
                    fontweight="bold",
                    rotation=90,
                    va="center",
                )

    fig.tight_layout(rect=[0.02, 0, 1, 0.96])
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"saved {output}")
    if show:
        plt.show()
    plt.close(fig)


def plot_latent_scatter(
    dataset: SavedDataset,
    z_all: torch.Tensor,
    output: Path,
    show: bool,
) -> None:
    """Diagnostic: latent vs ground-truth xy, colored by room and door."""
    z = z_all.cpu().numpy()
    xy = dataset.state_meta[:, :2]
    door = dataset.state_meta[:, 2]
    left = xy[:, 0] < WALL_X_NORM
    right = xy[:, 0] > WALL_X_NORM

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle("Latent alignment (red = target xy, dots = encoded z)", fontsize=12)

    for ax, title, mask in [
        (axes[0], "Left room", left),
        (axes[1], "Right room · door closed", right & (door < 0.5)),
        (axes[2], "Right room · door open", right & (door > 0.5)),
    ]:
        if not mask.any():
            ax.set_title(f"{title} (empty)")
            continue
        zm, xym = z[mask], (xy[mask]-0.5)
        ax.scatter(xym[:, 0], xym[:, 1], c="none", edgecolors="red", s=40, linewidths=1.2, label="target xy")
        ax.scatter(zm[:, 0], zm[:, 1], c="steelblue", s=18, alpha=0.8, label="latent z")
        for i in range(len(zm)):
            ax.plot([xym[i, 0], zm[i, 0]], [xym[i, 1], zm[i, 1]], "k-", alpha=0.15, linewidth=0.8)
        err = np.linalg.norm(zm - xym, axis=1).mean()
        ax.set_title(f"{title}\nmean |z − xy| = {err:.3f}")
        ax.set_xlabel("x / z₀")
        ax.set_ylabel("y / z₁")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"saved {output}")
    if show:
        plt.show()
    plt.close(fig)


def state_label(dataset: SavedDataset, idx: int) -> str:
    x, y, door = dataset.state_meta[idx]
    side = "left" if x < WALL_X_NORM else "right"
    door_s = "open" if door > 0.5 else "closed"
    return f"#{idx} {side} room, door {door_s}  (meta x={x:.2f}, y={y:.2f})"


def find_collapsed_latent_outlier(
    dataset: SavedDataset, z_all: torch.Tensor, tol: float = 0.015
) -> tuple[int, str]:
    """
    Find the isolated latent blob seen on energy plots.

    Usually many distinct door-closed states share one encoded z while
    door-open states spread near the origin.
    """
    z = z_all.cpu().numpy()
    rounded = np.round(z / tol).astype(int)
    groups: dict[tuple[int, ...], list[int]] = {}
    for i, key in enumerate(map(tuple, rounded)):
        groups.setdefault(key, []).append(i)

    median = np.median(z, axis=0)
    best_key = max(
        groups,
        key=lambda k: (
            len(groups[k]),
            float(np.linalg.norm(np.array(k) * tol - median)),
        ),
    )
    members = groups[best_key]
    idx = members[0]
    z_rep = z[idx]
    doors = dataset.state_meta[members, 2]
    door_closed = int(np.sum(doors < 0.5))
    note = (
        f"{len(members)} states share z~({z_rep[0]:.3f}, {z_rep[1]:.3f}) "
        f"({door_closed} door-closed); example {state_label(dataset, idx)}"
    )
    return idx, note


@torch.no_grad()
def plot_ae_reconstructions(
    dataset: SavedDataset,
    ae: Autoencoder,
    context_model: ContextModel,
    z_all: torch.Tensor,
    output: Path,
    *,
    show: bool,
) -> None:
    """Side-by-side original vs reconstruction for representative states."""
    outlier_idx, outlier_note = find_collapsed_latent_outlier(dataset, z_all)
    print(f"Latent outlier: {outlier_note}")

    meta = dataset.state_meta
    left = meta[:, 0] < WALL_X_NORM
    right = meta[:, 0] > WALL_X_NORM
    left_open = left & (meta[:, 2] > 0.5)
    left_closed = left & (meta[:, 2] < 0.5)
    right_open = right & (meta[:, 2] > 0.5)
    right_closed = right & (meta[:, 2] < 0.5)

    picks: list[tuple[int, str]] = [(outlier_idx, "Collapsed latent (outlier blob)")]
    if left_closed.any():
        picks.append((int(np.where(left_closed)[0][0]), "Left room, door closed"))
    if left_open.any():
        picks.append((int(np.where(left_open)[0][0]), "Left room, door open"))
    if right_closed.any():
        picks.append((int(np.where(right_closed)[0][0]), "Right room, door closed"))
    if right_open.any():
        picks.append((int(np.where(right_open)[0][0]), "Right room, door open"))

    images_t = (
        torch.from_numpy(dataset.states.astype(np.float32) / 255.0)
        .permute(0, 3, 1, 2)
        .to(DEVICE)
    )
    z = ae.encode(images_t)
    recon_list: list[torch.Tensor] = []
    mse = np.zeros(len(images_t), dtype=np.float64)
    for idx in range(len(images_t)):
        door_open = float(meta[idx, 2]) > 0.5
        ctx_action = torch.tensor(
            [ACTION_TO_IDX["open_door" if door_open else "noop"]],
            device=DEVICE,
            dtype=torch.long,
        )
        img = images_t[idx : idx + 1]
        c = context_model(img, ctx_action)
        rec = ae.decode(z[idx : idx + 1], c)
        recon_list.append(rec)
        mse[idx] = F.mse_loss(rec, img, reduction="mean").item()
    recon = torch.cat(recon_list, dim=0)

    worst = int(np.argmax(mse))
    if worst not in {i for i, _ in picks}:
        picks.append((worst, f"Highest recon MSE ({mse[worst]:.4f})"))

    n = len(picks)
    fig, axes = plt.subplots(n, 2, figsize=(6, 3 * n))
    if n == 1:
        axes = np.array([axes])
    fig.suptitle("Autoencoder: original vs reconstruction", fontsize=13, y=1.01)

    for row, (idx, row_title) in enumerate(picks):
        img = images_t[idx].cpu().permute(1, 2, 0).numpy()
        rec = recon[idx].cpu().permute(1, 2, 0).numpy().clip(0, 1)
        z_i = z_all[idx].cpu().numpy()
        subtitle = (
            f"{row_title} — {state_label(dataset, idx)}\n"
            f"z=({z_i[0]:.3f}, {z_i[1]:.3f})  mse={mse[idx]:.4f}"
        )

        axes[row, 0].imshow(img)
        axes[row, 0].set_title(f"Original\n{subtitle}", fontsize=9)
        axes[row, 0].axis("off")
        axes[row, 1].imshow(rec)
        axes[row, 1].set_title("Reconstruction", fontsize=9)
        axes[row, 1].axis("off")

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"saved {output}")
    if show:
        plt.show()
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize V, W, and total energy over latent space")
    p.add_argument("--data", type=Path, default=Path("data/unlocked_door/train.npz"))
    p.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    p.add_argument("--output", type=Path, default=Path("figures/potentials_3d.png"))
    p.add_argument(
        "--prob-output",
        type=Path,
        default=Path("figures/probabilities_3d.png"),
        help="Same layout as potentials but with exp(-energy) on 3D columns",
    )
    p.add_argument("--latent-output", type=Path, default=Path("figures/latent_scatter.png"))
    p.add_argument("--recon-output", type=Path, default=Path("figures/ae_recon.png"))
    p.add_argument("--skip-potentials", action="store_true")
    p.add_argument("--resolution", type=int, default=60, help="Grid points per latent axis")
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument(
        "--neg-padding",
        type=float,
        default=0.1,
        help="Same as train.py: latent box for V/W negatives (states ± this)",
    )
    p.add_argument(
        "--outer-padding",
        type=float,
        default=0.1,
        help="Extra plot margin beyond the neg box on each side",
    )
    p.add_argument("--show", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_dataset(str(args.data))
    ae, v_model, w_model, context_model = load_models(
        args.checkpoint_dir,
        require_energy_models=not args.skip_potentials,
    )
    z_all = encode_all_states(ae, dataset)
    plot_latent_scatter(dataset, z_all, args.latent_output, args.show)
    plot_ae_reconstructions(
        dataset, ae, context_model, z_all, args.recon_output, show=args.show
    )
    if not args.skip_potentials:
        visualize(
            dataset,
            ae,
            v_model,
            w_model,
            context_model,
            checkpoint_dir=args.checkpoint_dir,
            resolution=args.resolution,
            batch_size=args.batch_size,
            neg_padding=args.neg_padding,
            outer_padding=args.outer_padding,
            output=args.output,
            show=args.show,
            as_probability=False,
        )
        visualize(
            dataset,
            ae,
            v_model,
            w_model,
            context_model,
            checkpoint_dir=args.checkpoint_dir,
            resolution=args.resolution,
            batch_size=args.batch_size,
            neg_padding=args.neg_padding,
            outer_padding=args.outer_padding,
            output=args.prob_output,
            show=args.show,
            as_probability=True,
        )


if __name__ == "__main__":
    main()
