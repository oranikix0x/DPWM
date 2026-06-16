"""
Rollout ablation figures for the paper.

Compares ground-truth replay against world-model rollouts under:
  none (unconstrained), v_only, v+w (full dual-potential gate).

Layout per scenario figure:
  Row 0          : energy-landscape mini-plots, one per timestep column
                   shows V+W allowed region + all WM trajectories up to that step
  Rows 1..n_modes: rollout images  (row = mode, col = timestep)
                   images are flipped vertically to align with the latent z₁ axis

Example:
  python experiments/unlocked_door_symbolic/rollout.py
  python experiments/unlocked_door_symbolic/rollout.py --show
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch

from dpwm.context import NOOP, TRANSFORMATIVE, context_at_each_step
from dpwm.data.load import ContextTransitionDataset, SavedDataset, load_dataset
from dpwm.envs.unlocked_door import (
    ACTION_TO_IDX,
    Action,
    UnlockedDoorEnv,
    WorldState,
    is_door,
    is_internal_wall,
    is_valid_agent_pos,
    which_side,
)
from dpwm.models.autoencoder import Autoencoder
from dpwm.models.context_model import ContextModel, infer_context_dim, infer_w_context_dim
from dpwm.models.ebm import PotentialV, PotentialW
from dpwm.models.world_model import LatentDynamics, nearest_state_indices
from dpwm.render import render

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Fixed random walk used by the "random_walk" scenario
STEPS = 2048
_WALK_RNG = np.random.default_rng(seed=42)
_WALK_ACTIONS: tuple = tuple(
    str(a) for a in _WALK_RNG.choice(["up", "down", "left", "right", 'noop', 'open_door'], size=STEPS)
)


@torch.no_grad()
def encode_all_states(ae: Autoencoder, dataset: SavedDataset) -> torch.Tensor:
    images = (
        torch.from_numpy(dataset.states.astype(np.float32) / 255.0)
        .permute(0, 3, 1, 2)
        .to(DEVICE)
    )
    return ae.encode(images)


def load_models(checkpoint_dir: Path):
    ae_ckpt = torch.load(checkpoint_dir / "ae.pt", map_location=DEVICE, weights_only=False)
    w_ckpt = torch.load(checkpoint_dir / "w.pt", map_location=DEVICE, weights_only=False)
    w_key = "w_model" if "w_model" in w_ckpt else "model"
    context_dim = infer_context_dim(context_model_state=ae_ckpt["context_model"])
    w_context_dim = infer_w_context_dim(w_ckpt[w_key])
    if w_context_dim != context_dim:
        raise RuntimeError(
            f"Checkpoint mismatch: ae.pt context_dim={context_dim}, "
            f"w.pt context_dim={w_context_dim}. "
            "Retrain: python experiments/unlocked_door_symbolic/train.py --stage w"
        )
    print(f"checkpoint context_dim={context_dim}")

    ae = Autoencoder(latent_dim=2, context_dim=context_dim).to(DEVICE)
    v_model = PotentialV().to(DEVICE)
    w_model = PotentialW(context_dim=context_dim).to(DEVICE)
    context_model = ContextModel(context_dim=context_dim).to(DEVICE)

    ae.load_state_dict(ae_ckpt["model"])
    context_model.load_state_dict(ae_ckpt["context_model"])
    v_model.load_state_dict(
        torch.load(checkpoint_dir / "v.pt", map_location=DEVICE, weights_only=False)["model"]
    )
    w_model.load_state_dict(w_ckpt[w_key])

    ae.eval(); v_model.eval(); w_model.eval(); context_model.eval()
    return ae, v_model, w_model, context_model


@torch.no_grad()
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
    loader = torch.utils.data.DataLoader(ContextTransitionDataset(dataset), batch_size=4096)
    totals = []
    for next_idx, context_image, context_action in loader:
        next_idx = next_idx.to(DEVICE)
        context_image = context_image.to(DEVICE)
        context_action = context_action.to(DEVICE)
        z_next = z_all[next_idx]
        ctx_latent = context_model(context_image, context_action)
        totals.append(v_model(z_next) + w_model(z_next, ctx_latent))
    total = torch.cat(totals).cpu().numpy()
    e_value = float(np.percentile(total, 90))
    print(f"no e.pt found; estimated E={e_value:.3f} from data (90th percentile)")
    return e_value


AblationMode = Literal["none", "v_only", "v+w"]
WM_MODES = ("none", "v_only", "v+w")
MODE_LABELS = {
    "ground_truth": "Ground truth",
    "none":         "WM (none)",
    "v_only":       "WM + V",
    "v+w":          "WM + V+W",
}


def latent_grid_bounds(z_all, neg_padding, outer_padding=0.0):
    z_min = z_all.min(dim=0).values - neg_padding - outer_padding
    z_max = z_all.max(dim=0).values + neg_padding + outer_padding
    return z_min, z_max


def latent_grid(z_all, resolution, neg_padding, outer_padding=0.0):
    z_min, z_max = latent_grid_bounds(z_all, neg_padding, outer_padding)
    z0 = np.linspace(z_min[0].item(), z_max[0].item(), resolution)
    z1 = np.linspace(z_min[1].item(), z_max[1].item(), resolution)
    g0, g1 = np.meshgrid(z0, z1)
    z_flat = torch.from_numpy(
        np.stack([g0.ravel(), g1.ravel()], axis=1)
    ).float().to(DEVICE)
    neg_z_min = z_all.min(dim=0).values - neg_padding
    neg_z_max = z_all.max(dim=0).values + neg_padding
    return g0, g1, z_flat, neg_z_min, neg_z_max


def draw_neg_box_2d(ax, neg_z_min, neg_z_max):
    x0, y0 = neg_z_min[0].item(), neg_z_min[1].item()
    w = neg_z_max[0].item() - x0
    h = neg_z_max[1].item() - y0
    ax.add_patch(mpatches.Rectangle(
        (x0, y0), w, h,
        fill=False, edgecolor="#333333", linestyle="--", linewidth=1.0, zorder=5,
    ))


@torch.no_grad()
def allowed_region_mask(mode, z_flat, v_model, w_model, context_latent,
                         e_full, e_v, batch_size):
    if mode == "none":
        return np.ones(len(z_flat), dtype=bool)
    chunks = []
    ctx_batch = context_latent.expand(batch_size, -1)
    for start in range(0, len(z_flat), batch_size):
        z_batch = z_flat[start : start + batch_size]
        ctx = ctx_batch[: z_batch.size(0)]
        if mode == "v_only":
            chunks.append(v_model(z_batch) <= e_v)
        else:
            chunks.append(v_model(z_batch) + w_model(z_batch, ctx) <= e_full)
    return torch.cat(chunks).cpu().numpy()


@dataclass(frozen=True)
class Scenario:
    key: str
    title: str
    caption: str
    start: WorldState
    actions: tuple
    context_state: WorldState
    context_action: str = NOOP
    fixed_context: bool = False


SCENARIOS = (
    Scenario(
        "cross_door_closed",
        "Cross door · closed context",
        "Push right through a closed door (W should block; unconstrained WM may drift).",
        WorldState(x=1, y=5, door_open=False),
        ("right",) * STEPS,
        context_state=WorldState(x=1, y=5, door_open=False),
        context_action="noop",
    ),
    Scenario(
        "through_inner_wall",
        "Cross inner wall",
        "Push up into the dividing wall (V should block invalid geometry).",
        WorldState(x=5, y=5, door_open=True),
        ("up",) * STEPS,
        context_state=WorldState(x=4, y=5, door_open=True),
        context_action="open_door",
        fixed_context=True,
    ),
    Scenario(
        "open_then_cross",
        "Open door · then cross",
        "open_door then traverse; allowed region expands after context update.",
        WorldState(x=4, y=5, door_open=False),
        tuple(("open_door",) + ("right",) * (STEPS-1)),
        context_state=WorldState(x=4, y=5, door_open=False),
        context_action="noop",
    ),
    Scenario(
        "cross_with_open_context",
        "Cross door · open context",
        "Context already post-open_door (adjacent, door open); agent still in left room.",
        WorldState(x=1, y=5, door_open=True),
        ("right",) * STEPS,
        context_state=WorldState(x=4, y=5, door_open=True),
        context_action="open_door",
        fixed_context=True,
    ),
    Scenario(
        "noop_drift",
        "Long noop rollout",
        "Repeated noop: unconstrained latent drift and decoder artifacts.",
        WorldState(x=2, y=2, door_open=False),
        ("noop",) * STEPS,
        context_state=WorldState(x=2, y=2, door_open=False),
        context_action="noop",
    ),
    Scenario(
        "random_walk",
        "Random walk · door open",
        "Random movement with door open; unconstrained WM free to whiz off-manifold.",
        WorldState(x=3, y=3, door_open=True),
        _WALK_ACTIONS,
        context_state=WorldState(x=4, y=5, door_open=True),
        context_action="open_door",
        fixed_context=True,
    ),
)


@dataclass
class RolloutStep:
    image: np.ndarray
    z: object
    meta_xy_door: object
    allowed: object
    gt_state: WorldState
    context_latent_np: object = None   # np.ndarray stored by WM rollouts


@dataclass
class RolloutTrace:
    mode: str
    steps: list
    summary: str


def find_state_idx(dataset, state, env):
    x_norm = state.x / (env.width - 1)
    y_norm = state.y / (env.height - 1)
    for i in range(len(dataset.state_meta)):
        meta = dataset.state_meta[i]
        if (
            abs(float(meta[0]) - x_norm) < 1e-4
            and abs(float(meta[1]) - y_norm) < 1e-4
            and (float(meta[2]) > 0.5) == state.door_open
        ):
            return i
    raise ValueError(f"state not in dataset table: {state}")


def image_from_state(env, state):
    return render(state, env)


def state_index_to_tensor(dataset, idx):
    img = dataset.states[idx].astype(np.float32) / 255.0
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(DEVICE)


@torch.no_grad()
def estimate_v_threshold(z_all, v_model, indices, percentile=90.0):
    v = v_model(z_all[torch.from_numpy(indices).long().to(DEVICE)]).cpu().numpy()
    return float(np.percentile(v, percentile))


@torch.no_grad()
def estimate_w_threshold(dataset, z_all, w_model, context_model, percentile=90.0):
    loader = torch.utils.data.DataLoader(
        ContextTransitionDataset(dataset), batch_size=4096,
    )
    values = []
    for next_idx, context_image, context_action in loader:
        next_idx = next_idx.to(DEVICE)
        context_image = context_image.to(DEVICE)
        context_action = context_action.to(DEVICE)
        z_next = z_all[next_idx]
        ctx = context_model(context_image, context_action)
        values.append(w_model(z_next, ctx))
    return float(np.percentile(torch.cat(values).cpu().numpy(), percentile))


@torch.no_grad()
def is_allowed_ablation(mode, v_model, w_model, z, context_latent, e_full, e_v, e_w):
    if mode == "none":
        return torch.ones(z.size(0), dtype=torch.bool, device=z.device)
    v = v_model(z)
    if mode == "v_only":
        return v <= e_v
    return (v + w_model(z, context_latent)) <= e_full


def project_along_step(mode, v_model, w_model, z_current, z_pred, context_latent,
                        e_full, e_v, e_w, n_iters=12):
    batch_size = z_current.shape[0]
    lo = torch.zeros(batch_size, device=z_current.device)
    hi = torch.ones(batch_size, device=z_current.device)
    for _ in range(n_iters):
        mid = 0.5 * (lo + hi)
        z_mid = z_current + mid.unsqueeze(-1) * (z_pred - z_current)
        ok = is_allowed_ablation(mode, v_model, w_model, z_mid, context_latent, e_full, e_v, e_w)
        lo = torch.where(ok, mid, lo)
        hi = torch.where(ok, hi, mid)
    return z_current + lo.unsqueeze(-1) * (z_pred - z_current)


@torch.no_grad()
def wm_step(mode, dynamics, ae, context_model, v_model, w_model, z_current,
            context_image, context_action, action, e_full, e_v, e_w):
    context_latent = context_model(context_image, context_action)
    z_pred = dynamics(z_current, context_latent, action)
    allowed = is_allowed_ablation(mode, v_model, w_model, z_pred, context_latent, e_full, e_v, e_w)
    z_proj = project_along_step(mode, v_model, w_model, z_current, z_pred,
                                 context_latent, e_full, e_v, e_w)
    z_out = torch.where(allowed.unsqueeze(-1), z_pred, z_proj)
    return z_out, z_pred, allowed


def meta_from_nearest(z, z_all, dataset):
    idx = int(nearest_state_indices(z, z_all).item())
    meta = dataset.state_meta[idx]
    return float(meta[0]), float(meta[1]), float(meta[2])


def decode_to_numpy(ae, context_model, z, context_image, context_action):
    ctx = context_model(context_image, context_action)
    recon = ae.decode(z, ctx)
    return recon.squeeze(0).permute(1, 2, 0).cpu().numpy().clip(0, 1)


def update_context(gt_state, prev_state, action, context_state, context_action):
    """Mirror context_at_each_step: only update on a *successful* transformative action."""
    if action in TRANSFORMATIVE and state_key(gt_state) != state_key(prev_state):
        return gt_state, action
    return context_state, context_action


def rollout_ground_truth(env, scenario, context_state, context_action):
    state = scenario.start
    ctx_state, ctx_action = context_state, context_action
    steps = [RolloutStep(image=image_from_state(env, state), z=None,
                         meta_xy_door=None, allowed=None, gt_state=state)]
    for action in scenario.actions:
        prev_state = state
        state = env.step(state, action)
        ctx_state, ctx_action = update_context(state, prev_state, action, ctx_state, ctx_action)
        steps.append(RolloutStep(image=image_from_state(env, state), z=None,
                                 meta_xy_door=None, allowed=None, gt_state=state))
    return RolloutTrace(mode="ground_truth", steps=steps, summary="")


@torch.no_grad()
def enrich_gt_trace(gt_trace, scenario, dataset, z_all, context_model, env):
    """
    Post-hoc enrichment of the ground-truth trace with latent coordinates and
    context latents, so it can be plotted on the energy landscape just like WM traces.

    z is looked up from z_all by exact state match in the dataset.
    context_latent_np follows the same update_context logic used during rollout.
    """
    ctx_state, ctx_action = scenario.context_state, scenario.context_action
    actions = list(scenario.actions)

    for i, step in enumerate(gt_trace.steps):
        # ── z from dataset lookup ──────────────────────────────────────────
        try:
            idx = find_state_idx(dataset, step.gt_state, env)
            step.z = z_all[idx].cpu().numpy()
        except ValueError:
            step.z = None

        # ── context latent at this step (before this step's action) ───────
        ctx_img, ctx_act_t = context_tensors(dataset, env, ctx_state, ctx_action)
        step.context_latent_np = (
            context_model(ctx_img, ctx_act_t).squeeze(0).cpu().numpy()
        )

        # ── advance context using the action that led to step i+1 ─────────
        if i < len(actions) and i + 1 < len(gt_trace.steps):
            next_step = gt_trace.steps[i + 1]
            ctx_state, ctx_action = update_context(
                next_step.gt_state, step.gt_state, actions[i], ctx_state, ctx_action
            )



@torch.no_grad()
def rollout_wm(mode, env, dataset, scenario, ae, dynamics, context_model,
               v_model, w_model, z_all, e_full, e_v, e_w, context_state, context_action):
    gt_state = scenario.start
    ctx_state, ctx_action = context_state, context_action

    start_idx = find_state_idx(dataset, gt_state, env)
    z = ae.encode(state_index_to_tensor(dataset, start_idx))
    ctx_image, ctx_action_t = context_tensors(dataset, env, ctx_state, ctx_action)

    steps = []
    rejects = 0
    invalid_nn = 0

    def append_step(z_t, allowed):
        nonlocal invalid_nn
        meta = meta_from_nearest(z_t, z_all, dataset)
        x, y, door = meta
        gx = int(round(x * (env.width - 1)))
        gy = int(round(y * (env.height - 1)))
        if not is_valid_agent_pos(gx, gy, door > 0.5):
            invalid_nn += 1
        img = decode_to_numpy(ae, context_model, z_t, ctx_image, ctx_action_t)
        ctx_lat_np = context_model(ctx_image, ctx_action_t).squeeze(0).cpu().numpy()
        steps.append(RolloutStep(image=img, z=z_t.squeeze(0).cpu().numpy(),
                                 meta_xy_door=meta, allowed=allowed, gt_state=gt_state,
                                 context_latent_np=ctx_lat_np))

    append_step(z, None)
    for action_name in scenario.actions:
        prev_gt = gt_state
        action_t = torch.tensor([ACTION_TO_IDX[action_name]], device=DEVICE)
        z, _, allowed = wm_step(mode, dynamics, ae, context_model, v_model, w_model,
                                 z, ctx_image, ctx_action_t, action_t, e_full, e_v, e_w)
        gt_state = env.step(prev_gt, action_name)
        ctx_state, ctx_action = update_context(gt_state, prev_gt, action_name, ctx_state, ctx_action)
        ctx_image, ctx_action_t = context_tensors(dataset, env, ctx_state, ctx_action)
        if not bool(allowed.item()):
            rejects += 1
        append_step(z, bool(allowed.item()))

    n_actions = len(scenario.actions)
    return RolloutTrace(
        mode=mode, steps=steps,
        summary=f"rejects={rejects}/{n_actions}  invalid-nn={invalid_nn}/{len(steps)}",
    )


def context_tensors(dataset, env, ctx_state, ctx_action):
    ctx_image = state_index_to_tensor(dataset, find_state_idx(dataset, ctx_state, env))
    ctx_action_t = torch.tensor([ACTION_TO_IDX[ctx_action]], device=DEVICE)
    return ctx_image, ctx_action_t


def verify_scenario_contexts(scenario, env):
    if scenario.fixed_context:
        return
    state = scenario.start
    trajectory = []
    for action in scenario.actions:
        nxt = env.step(state, action)
        trajectory.append((state, action, nxt))
        state = nxt
    expected = context_at_each_step(trajectory)
    if not expected:
        return
    first_ctx_state, first_ctx_action = expected[0]
    if (
        state_key(first_ctx_state) != state_key(scenario.context_state)
        or first_ctx_action != scenario.context_action
    ):
        raise ValueError(
            f"{scenario.key}: initial context {scenario.context_state}/{scenario.context_action} "
            f"!= replay {first_ctx_state}/{first_ctx_action}"
        )


def state_key(state):
    return (state.x, state.y, state.door_open)


def diagnose_trace(trace, env):
    flags = []
    if trace.mode == "ground_truth":
        return flags
    for step in trace.steps:
        if step.meta_xy_door is None:
            continue
        x, y, door = step.meta_xy_door
        gx = int(round(x * (env.width - 1)))
        gy = int(round(y * (env.height - 1)))
        if gx < 0 or gy < 0 or gx >= env.width or gy >= env.height:
            flags.append("out-of-bounds"); break
        if is_internal_wall(gx, gy):
            flags.append("inner wall"); break
        if is_door(gx, gy) and not (door > 0.5):
            flags.append("door (closed)"); break
    return flags


def pick_frame_indices(n_steps, max_frames=8):
    if n_steps <= max_frames:
        return list(range(n_steps))
    idx = np.round(np.linspace(0, n_steps - 1, max_frames)).astype(int)
    out = []
    for i in idx:
        if int(i) not in out:
            out.append(int(i))
    return out


# ---------------------------------------------------------------------------
# Per-scenario figure
# ---------------------------------------------------------------------------

_TRACE_COLORS = {
    "ground_truth": "#555555",
    "none":         "#d62728",
    "v_only":       "#ff7f0e",
    "v+w":          "#2ca02c",
}


def _make_rgba_bg(mask2d):
    rgba = np.zeros((*mask2d.shape, 4))
    rgba[mask2d]  = (0.42, 0.78, 0.42, 0.42)
    rgba[~mask2d] = (0.91, 0.91, 0.91, 0.82)
    return rgba


def _ctx_lat_for_step(trace, step_idx, fallback):
    """Return a (1, ctx_dim) tensor for the context at step_idx."""
    if (
        step_idx < len(trace.steps)
        and trace.steps[step_idx].context_latent_np is not None
    ):
        return (
            torch.from_numpy(trace.steps[step_idx].context_latent_np)
            .unsqueeze(0).to(DEVICE)
        )
    return fallback


def plot_scenario_figure(
    scenario,
    traces,
    z_all,
    dataset,
    env,
    v_model,
    w_model,
    context_model,
    e_full,
    e_v,
    output,
    show,
    *,
    grid_resolution=80,
    neg_padding=0.1,
    outer_padding=0.1,
    batch_size=4096,
    max_frames=6,
):
    """
    Layout — one energy-landscape row + one image row per trace:

      [label] | energy landscape (t=0) | … | energy landscape (t=N)   ← trace 0
              | image (t=0)           | … | image (t=N)               ← trace 0
      [label] | energy landscape (t=0) | … | energy landscape (t=N)   ← trace 1
              | image (t=0)           | … | image (t=N)               ← trace 1
      …

    Each energy landscape shows only *that trace's* trajectory up to the displayed
    step, so the reader can compare how each mode explores the latent space.

    Backgrounds:
      ground_truth → V+W region with initial context (reference)
      none         → plain (no constraint)
      v_only       → V-only region (static, context-independent)
      v+w          → V+W region, updated when context changes

    Images are flipped vertically to align with the latent z₁ axis.
    """
    n_steps = len(traces[0].steps)
    frame_indices = pick_frame_indices(n_steps, max_frames=max_frames)
    n_cols = len(frame_indices)
    n_traces = len(traces)

    # --- energy-landscape grid setup --------------------------------------
    ctx_image, ctx_action_t = context_tensors(
        dataset, env, scenario.context_state, scenario.context_action
    )
    context_latent = context_model(ctx_image, ctx_action_t)  # fallback
    g0, g1, z_flat, neg_z_min, neg_z_max = latent_grid(
        z_all, grid_resolution, neg_padding, outer_padding
    )
    z_np = z_all.cpu().numpy()

    # Pre-compute static V-only mask (context-independent)
    allowed_v = allowed_region_mask(
        "v_only", z_flat, v_model, w_model, context_latent, e_full, e_v, batch_size
    )
    rgba_v = _make_rgba_bg(allowed_v.reshape(g0.shape))

    # Context-aware V+W cache (recomputed on context change)
    _vw_cache: dict = {}

    def _get_rgba_vw(ctx_lat_t: torch.Tensor) -> np.ndarray:
        key = ctx_lat_t.cpu().numpy().tobytes()
        if key not in _vw_cache:
            mask = allowed_region_mask(
                "v+w", z_flat, v_model, w_model, ctx_lat_t,
                e_full, e_v, batch_size,
            ).reshape(g0.shape)
            _vw_cache[key] = _make_rgba_bg(mask)
        return _vw_cache[key]

    # --- figure layout ----------------------------------------------------
    label_w  = 0.75   # inches
    col_w    = 1.90   # inches
    energy_h = 1.85   # inches
    img_h    = 1.30   # inches

    fig_w = label_w + col_w * n_cols
    fig_h = (energy_h + img_h) * n_traces + 0.35

    # Alternating energy / image rows
    height_ratios = []
    for _ in range(n_traces):
        height_ratios.extend([energy_h / img_h, 1.0])

    width_ratios = [label_w / col_w] + [1.0] * n_cols

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs  = fig.add_gridspec(
        2 * n_traces, 1 + n_cols,
        height_ratios=height_ratios,
        width_ratios=width_ratios,
        hspace=0.08,
        wspace=0.18,
    )

    # ── per-trace rows ────────────────────────────────────────────────────
    for t_i, trace in enumerate(traces):
        e_row = 2 * t_i      # energy landscape row index
        i_row = 2 * t_i + 1  # image row index
        color = _TRACE_COLORS[trace.mode]

        # Label spanning both energy + image rows
        ax_lbl = fig.add_subplot(gs[e_row : i_row + 1, 0])
        ax_lbl.axis("off")
        ax_lbl.text(
            0.5, 0.5, MODE_LABELS[trace.mode],
            transform=ax_lbl.transAxes,
            va="center", ha="center",
            fontsize=7, rotation=90,
        )

        for ci, step_idx in enumerate(frame_indices):

            # ── energy landscape ──────────────────────────────────────────
            ax_e = fig.add_subplot(gs[e_row, ci + 1])

            if trace.mode == "none":
                ax_e.set_facecolor("#f2f2f2")
                ax_e.set_xlim(g0.min(), g0.max())
                ax_e.set_ylim(g1.min(), g1.max())
            elif trace.mode == "v_only":
                ax_e.imshow(
                    rgba_v, origin="lower",
                    extent=(g0.min(), g0.max(), g1.min(), g1.max()),
                    aspect="auto", zorder=0,
                )
                ax_e.set_xlim(g0.min(), g0.max())
                ax_e.set_ylim(g1.min(), g1.max())
            else:  # "v+w" or "ground_truth"
                ctx_t = _ctx_lat_for_step(trace, step_idx, context_latent)
                ax_e.imshow(
                    _get_rgba_vw(ctx_t), origin="lower",
                    extent=(g0.min(), g0.max(), g1.min(), g1.max()),
                    aspect="auto", zorder=0,
                )
                ax_e.set_xlim(g0.min(), g0.max())
                ax_e.set_ylim(g1.min(), g1.max())

            # Faint all-state scatter
            ax_e.scatter(
                z_np[:, 0], z_np[:, 1],
                s=2, c="#888888", alpha=0.18, linewidths=0, zorder=1,
            )

            # This trace's trajectory up to step_idx
            zs = [s.z for s in trace.steps[: step_idx + 1] if s.z is not None]
            if zs:
                arr = np.array(zs)
                ax_e.plot(arr[:, 0], arr[:, 1], "-",
                          lw=0.9, color=color, alpha=0.65, zorder=2)

            # Current-position dot
            if step_idx < len(trace.steps) and trace.steps[step_idx].z is not None:
                z_now = trace.steps[step_idx].z
                ax_e.scatter(
                    z_now[0], z_now[1],
                    s=30, color=color,
                    edgecolors="white", linewidths=0.6, zorder=4,
                )

            draw_neg_box_2d(ax_e, neg_z_min, neg_z_max)

            # Column header only on first trace row
            if t_i == 0:
                ax_e.set_title(f"t={step_idx}", fontsize=7, pad=2)

            ax_e.tick_params(labelbottom=False, labelleft=False, length=2)

            # ── image ─────────────────────────────────────────────────────
            ax_i = fig.add_subplot(gs[i_row, ci + 1])
            step = trace.steps[step_idx]

            ax_i.imshow(np.flipud(step.image))
            ax_i.set_xticks([])
            ax_i.set_yticks([])
            for spine in ax_i.spines.values():
                spine.set_visible(False)

            # Red border when nearest-neighbour is geometrically invalid
            if trace.mode != "ground_truth" and step.meta_xy_door is not None:
                xm, ym, dm = step.meta_xy_door
                gx = int(round(xm * (env.width - 1)))
                gy = int(round(ym * (env.height - 1)))
                if not is_valid_agent_pos(gx, gy, dm > 0.5):
                    h, w = step.image.shape[:2]
                    ax_i.add_patch(mpatches.Rectangle(
                        (-0.5, -0.5), w, h,
                        linewidth=1.8, edgecolor="red",
                        facecolor="none", zorder=5,
                    ))

    fig.suptitle(
        f"{scenario.title}  ·  {scenario.caption}",
        fontsize=8, y=0.999, va="bottom",
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"saved {output}")
    if show:
        plt.show()
    plt.close(fig)


def plot_summary_grid(scenarios, all_traces, output, show):
    """Compact overview: one row per scenario, columns = methods, last frame only."""
    methods = ("ground_truth", "none", "v_only", "v+w")
    n_rows, n_cols = len(scenarios), len(methods)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.2 * n_cols, 2.4 * n_rows))
    if n_rows == 1:
        axes = np.expand_dims(axes, 0)

    fig.suptitle("Rollout ablations (final timestep)", fontsize=13, y=0.995)

    for row, scenario in enumerate(scenarios):
        traces = {t.mode: t for t in all_traces[scenario.key]}
        for col, mode in enumerate(methods):
            ax = axes[row, col]
            trace = traces[mode]
            ax.imshow(np.flipud(trace.steps[-1].image))
            if row == 0:
                ax.set_title(MODE_LABELS[mode], fontsize=9)
            if col == 0:
                ax.set_ylabel(scenario.title, fontsize=8)
            ax.axis("off")
            diag = diagnose_trace(trace, UnlockedDoorEnv())
            if diag:
                ax.text(0.02, 0.98, ", ".join(diag), transform=ax.transAxes,
                        va="top", ha="left", fontsize=7, color="red",
                        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1))

    fig.tight_layout(rect=[0, 0, 1, 0.98])
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"saved {output}")
    if show:
        plt.show()
    plt.close(fig)


def load_dynamics(checkpoint_dir):
    ckpt = torch.load(checkpoint_dir / "wm.pt", map_location=DEVICE, weights_only=False)
    in_dim = int(ckpt["model"]["net.0.weight"].shape[1])
    latent_dim, action_dim = 2, 32
    context_dim = in_dim - latent_dim - action_dim
    if context_dim < 1:
        raise ValueError(f"unexpected WM input dim {in_dim}; retrain with current ContextModel")
    dynamics = LatentDynamics(context_dim=context_dim).to(DEVICE)
    dynamics.load_state_dict(ckpt["model"])
    dynamics.eval()
    return dynamics


def parse_args():
    p = argparse.ArgumentParser(description="Paper rollout ablation figures")
    p.add_argument("--data",           type=Path, default=Path("data/unlocked_door/train.npz"))
    p.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    p.add_argument("--output-dir",     type=Path, default=Path("figures/rollouts"))
    p.add_argument("--show",           action="store_true")
    p.add_argument("--scenarios",      nargs="*", default=None,
                   help="Scenario keys to run (default: all)")
    p.add_argument("--neg-padding",    type=float, default=0.1)
    p.add_argument("--outer-padding",  type=float, default=0.1)
    p.add_argument("--max-frames",     type=int,   default=6,
                   help="Timestep columns per figure (default 6)")
    return p.parse_args()


def main():
    args = parse_args()
    env = UnlockedDoorEnv()
    dataset = load_dataset(str(args.data))

    ae, v_model, w_model, context_model = load_models(args.checkpoint_dir)
    dynamics = load_dynamics(args.checkpoint_dir)
    z_all = encode_all_states(ae, dataset)

    referenced = np.unique(np.concatenate(
        [dataset.state_idx, dataset.next_idx, dataset.context_state_idx]
    ))
    e_full = load_energy_threshold(
        args.checkpoint_dir, dataset, z_all, v_model, w_model, context_model
    )
    e_v = estimate_v_threshold(z_all, v_model, referenced)
    e_w = estimate_w_threshold(dataset, z_all, w_model, context_model)
    print(f"thresholds: E={e_full:.3f}  E_v={e_v:.3f}  E_w={e_w:.3f}")

    keys = set(args.scenarios) if args.scenarios else None
    selected = [s for s in SCENARIOS if keys is None or s.key in keys]
    if not selected:
        raise SystemExit("no matching scenarios")

    all_traces = {}
    for scenario in selected:
        verify_scenario_contexts(scenario, env)
        gt_trace = rollout_ground_truth(env, scenario, scenario.context_state, scenario.context_action)
        enrich_gt_trace(gt_trace, scenario, dataset, z_all, context_model, env)
        traces = [gt_trace]
        for mode in WM_MODES:
            traces.append(rollout_wm(
                mode, env, dataset, scenario,
                ae, dynamics, context_model, v_model, w_model,
                z_all, e_full, e_v, e_w,
                scenario.context_state, scenario.context_action,
            ))
        all_traces[scenario.key] = traces
        plot_scenario_figure(
            scenario, traces, z_all, dataset, env,
            v_model, w_model, context_model, e_full, e_v,
            args.output_dir / f"{scenario.key}.png",
            args.show,
            neg_padding=args.neg_padding,
            outer_padding=args.outer_padding,
            max_frames=args.max_frames,
        )

    plot_summary_grid(
        tuple(selected), all_traces,
        args.output_dir / "ablation_summary.png", args.show,
    )


if __name__ == "__main__":
    main()