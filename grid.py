from __future__ import annotations

import math

from config import Config


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def in_bounds_cell(cfg: Config, cx: int, cy: int) -> bool:
    return 0 <= cx < cfg.tiles_x and 0 <= cy < cfg.tiles_y


def world_to_cell(cfg: Config, px: float, py: float) -> tuple[int, int]:
    return int(px // cfg.tile_size), int(py // cfg.tile_size)


def cell_center(cfg: Config, cx: int, cy: int) -> tuple[float, float]:
    return (cx + 0.5) * cfg.tile_size, (cy + 0.5) * cfg.tile_size


def rotate_toward(current: float, target: float, max_delta: float) -> float:
    diff = (target - current + math.pi) % (2 * math.pi) - math.pi
    if abs(diff) <= max_delta:
        return target
    return current + max_delta * (1 if diff > 0 else -1)


def is_wall_cell(cfg: Config, state, cx: int, cy: int) -> bool:
    if not in_bounds_cell(cfg, cx, cy):
        return True
    return state.objective[cx][cy] == 1


def is_confirmed_wall(cfg: Config, state, cx: int, cy: int) -> bool:
    if cx == 0 or cy == 0 or cx == cfg.tiles_x - 1 or cy == cfg.tiles_y - 1:
        return True
    return state.ever_seen[cx][cy] and state.subjective[cx][cy] >= cfg.wall_conf_thresh


def is_known_open(cfg: Config, state, cx: int, cy: int) -> bool:
    return state.ever_seen[cx][cy] and state.subjective[cx][cy] < cfg.wall_conf_thresh
