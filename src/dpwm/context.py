"""Context tracking for trajectory export."""

from __future__ import annotations

from dpwm.envs.unlocked_door import ACTION_TO_IDX, TRANSFORMATIVE_ACTIONS, Action, WorldState

TRANSFORMATIVE: set[Action] = set(TRANSFORMATIVE_ACTIONS)
NOOP: Action = "noop"


def context_at_each_step(
    trajectory: list[tuple[WorldState, Action, WorldState]],
) -> list[tuple[WorldState, Action]]:
    """
    Per-transition context: snapshot after the last successful transformative action.

    Before any transformative action: first trajectory state with ``noop`` action.
    """
    if not trajectory:
        return []

    ctx_state = trajectory[0][0]
    ctx_action: Action = NOOP
    contexts: list[tuple[WorldState, Action]] = []

    for s, action, s_next in trajectory:
        contexts.append((ctx_state, ctx_action))
        if action in TRANSFORMATIVE and s_next != s:
            ctx_state = s_next
            ctx_action = action

    return contexts
