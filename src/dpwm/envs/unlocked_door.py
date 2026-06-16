from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import random
import matplotlib.pyplot as plt
import numpy as np


Action = str


ACTIONS: Tuple[Action, ...] = (
    "up",
    "down",
    "left",
    "right",
    "open_door",
    "noop",
)

ACTION_TO_IDX = {name: i for i, name in enumerate(ACTIONS)}

EXPLORATION_ACTIONS: Tuple[Action, ...] = (
    "up",
    "down",
    "left",
    "right",
    "noop",
)

TRANSFORMATIVE_ACTIONS: Tuple[Action, ...] = (
    "open_door",
)

def is_border_wall(x: int, y: int) -> bool:
    if x < 0 or y < 0 or x >= UnlockedDoorEnv.width or y >= UnlockedDoorEnv.height:
        return True
    return x == 0 or x == UnlockedDoorEnv.width - 1 or y == 0 or y == UnlockedDoorEnv.height - 1


def is_valid_agent_pos(x: int, y: int, door_open: bool) -> bool:
    """Agent must sit on an in-bounds, non-wall cell (door tile only if open)."""
    if is_border_wall(x, y) or is_internal_wall(x, y):
        return False
    if is_door(x, y) and not door_open:
        return False
    return True


def iter_agent_positions(door_open: bool = False) -> tuple[tuple[int, int], ...]:
    positions: list[tuple[int, int]] = []
    for x in range(UnlockedDoorEnv.width):
        for y in range(UnlockedDoorEnv.height):
            if is_valid_agent_pos(x, y, door_open):
                positions.append((x, y))
    return tuple(positions)

def is_internal_wall(x: int, y: int) -> bool:
    return x == UnlockedDoorEnv.wall_x and y != UnlockedDoorEnv.door_y

def is_door(x: int, y: int) -> bool:
    return x == UnlockedDoorEnv.wall_x and y == UnlockedDoorEnv.door_y

def is_walkable(x: int, y: int) -> bool:
    return not is_border_wall(x, y) and not is_internal_wall(x, y)

def which_side(x: int, y: int) -> str:
    if x < UnlockedDoorEnv.wall_x:
        return "left"
    elif x > UnlockedDoorEnv.wall_x:
        return "right"
    elif y == UnlockedDoorEnv.door_y:
        return "door"
    else:
        return "wall"

def door_adjacent_left(x: int, y: int) -> bool:
    return x == UnlockedDoorEnv.wall_x - 1 and y == UnlockedDoorEnv.door_y

def door_adjacent_right(x: int, y: int) -> bool:
    return x == UnlockedDoorEnv.wall_x + 1 and y == UnlockedDoorEnv.door_y

def can_move_into(x: int, y: int, door_open: bool) -> bool:
    if door_open:
        return is_walkable(x, y) and not is_internal_wall(x, y) and not is_border_wall(x, y)
    else:
        return is_walkable(x, y) and not is_internal_wall(x, y) and not is_border_wall(x, y) and not is_door(x, y)


@dataclass(frozen=True)
class WorldState:
    """
    Symbolic state for the unlocked-door environment.

    x, y:
        Agent position on the grid.

    door_open:
        Whether the door between the two rooms is open.

    Important:
        The right room is globally valid even when the door is closed.
        It is just not reachable from the left room until the door is opened.
    """

    x: int
    y: int
    door_open: bool

class UnlockedDoorEnv:
    """
    Environment for the unlocked-door task.

    The room is a 10x9 grid looking like this, with the door at in the middle.

    #---------#---------#
    |         |         |
    |         D         |
    |         |         |
    #---------#---------#

    """

    width: int = 11
    height: int = 11

    wall_x: int = 5

    door_y: int = 5

    def __init__(
        self,
        start_state: WorldState | None = None,
        max_steps: int = 100,
        seed: int | None = None,
    ) -> None:
        self.rng = random.Random(seed)
        self.max_steps = max_steps
        self.start_state = start_state or WorldState(x=1, y=5, door_open=False)
        self.state = self.start_state
        self.t = 0

    def reset(self) -> None:
        ...

    def step(self, current_state: WorldState, action: Action) -> None:
        if action == "open_door":
            if door_adjacent_left(current_state.x, current_state.y) or door_adjacent_right(current_state.x, current_state.y):
                new_state = WorldState(x=current_state.x, y=current_state.y, door_open=True)
            else:
                new_state = current_state
        elif action == "up":
            if can_move_into(current_state.x, current_state.y + 1, current_state.door_open):
                new_state = WorldState(x=current_state.x, y=current_state.y + 1, door_open=current_state.door_open)
            else:
                new_state = current_state
        elif action == "down":
            if can_move_into(current_state.x, current_state.y - 1, current_state.door_open):
                new_state = WorldState(x=current_state.x, y=current_state.y - 1, door_open=current_state.door_open)
            else:
                new_state = current_state
        elif action == "left":
            if can_move_into(current_state.x - 1, current_state.y, current_state.door_open):
                new_state = WorldState(x=current_state.x - 1, y=current_state.y, door_open=current_state.door_open)
            else:
                new_state = current_state
        elif action == "right":
            if can_move_into(current_state.x + 1, current_state.y, current_state.door_open):
                new_state = WorldState(x=current_state.x + 1, y=current_state.y, door_open=current_state.door_open)
            else:
                new_state = current_state
        elif action == "noop":
            new_state = current_state
        else:
            new_state = current_state
        return new_state

class DataVisualizer:
    def __init__(
        self,
        data: List[Tuple[WorldState, Action]],
        env: UnlockedDoorEnv,
    ) -> None:
        self.data = data
        self.env = env

    def visualize_data(self) -> None:
        steps = []
        positions = []
        trajectory = self.data[0]
        for i, (state, action, next_state) in enumerate(trajectory):
            steps.append(i)
            positions.append((state.x, state.y))
        # Draw path line
        plt.plot(np.array(positions)[:, 0], np.array(positions)[:, 1], "-")

        # Draw colored points
        sc = plt.scatter(
            np.array(positions)[:, 0],
            np.array(positions)[:, 1],
            c=np.array(steps),
            cmap="viridis",
        )

        # draw wall
        plt.plot([self.env.wall_x, self.env.wall_x], [0, self.env.height], "k-")

        plt.colorbar(sc)
        plt.xlim(0, self.env.width)
        plt.ylim(0, self.env.height)
        plt.title("Trajectory of the agent in the environment")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()
        plt.clf()