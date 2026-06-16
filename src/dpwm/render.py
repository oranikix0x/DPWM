import numpy as np

from dpwm.envs.unlocked_door import UnlockedDoorEnv, WorldState

TILE_SIZE = 8


def render(state: WorldState, env: UnlockedDoorEnv, tile_size: int = TILE_SIZE) -> np.ndarray:
    """Render the world state as an RGB image."""

    floor = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    wall = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    closed_door = np.array([0.45, 0.22, 0.05], dtype=np.float32)
    open_door = floor
    agent = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    img = np.ones((env.height * tile_size, env.width * tile_size, 3), dtype=np.float32)
    img[:, :, :] = floor

    def fill_cell(x: int, y: int, color: np.ndarray) -> None:
        y0 = y * tile_size
        y1 = (y + 1) * tile_size
        x0 = x * tile_size
        x1 = (x + 1) * tile_size
        img[y0:y1, x0:x1, :] = color

    for x in range(env.width):
        fill_cell(x, 0, wall)
        fill_cell(x, env.height - 1, wall)
    for y in range(env.height):
        fill_cell(0, y, wall)
        fill_cell(env.width - 1, y, wall)

    for y in range(1, env.height - 1):
        if y == env.door_y:
            fill_cell(env.wall_x, y, open_door if state.door_open else closed_door)
        else:
            fill_cell(env.wall_x, y, wall)

    fill_cell(state.x, state.y, agent)

    return img
