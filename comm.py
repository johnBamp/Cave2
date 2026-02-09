from __future__ import annotations

import random

from config import Config
from grid import world_to_cell


def region_from_cell(cfg: Config, cell: tuple[int, int]) -> int:
    rx = min(cfg.comm_grid_x - 1, max(0, cell[0] * cfg.comm_grid_x // cfg.tiles_x))
    ry = min(cfg.comm_grid_y - 1, max(0, cell[1] * cfg.comm_grid_y // cfg.tiles_y))
    return ry * cfg.comm_grid_x + rx


def choose_intent_region(cfg: Config, agent) -> int:
    if agent.target_cell is not None:
        return region_from_cell(cfg, agent.target_cell)
    current_cell = world_to_cell(cfg, agent.pos[0], agent.pos[1])
    return region_from_cell(cfg, current_cell)


def choose_call(cfg: Config, agent, region_id: int, rng: random.Random) -> int:
    if cfg.comm_calls <= 0:
        return 0
    if rng.random() < cfg.comm_eps:
        return rng.randrange(cfg.comm_calls)
    row_best = 0
    best_val = -1.0
    for call_id in range(cfg.comm_calls):
        val = agent.comm_call_probs[call_id][region_id]
        if val > best_val:
            best_val = val
            row_best = call_id
    return row_best


def update_call_mapping(cfg: Config, agent, call_id: int, region_id: int) -> None:
    if call_id < 0 or call_id >= len(agent.comm_call_probs):
        return
    row = agent.comm_call_probs[call_id]
    if region_id < 0 or region_id >= len(row):
        return
    for i in range(len(row)):
        row[i] *= (1.0 - cfg.comm_lr)
    row[region_id] += cfg.comm_lr
    total = sum(row)
    if total > 1e-8:
        for i in range(len(row)):
            row[i] /= total


def apply_region_merge(cfg: Config, agent, region_id: int) -> None:
    if region_id < 0 or region_id >= len(agent.comm_region_known):
        return
    agent.comm_region_known[region_id] = min(1.0, agent.comm_region_known[region_id] + cfg.comm_merge_alpha)
