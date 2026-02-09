from __future__ import annotations

import random

from config import Config
from grid import clamp, in_bounds_cell


def init_bushes(cfg: Config, objective, main_region, rng: random.Random):
    bushes = [[False for _ in range(cfg.tiles_y)] for _ in range(cfg.tiles_x)]
    fruit = [[False for _ in range(cfg.tiles_y)] for _ in range(cfg.tiles_x)]
    last_harvest_time = [[-1e9 for _ in range(cfg.tiles_y)] for _ in range(cfg.tiles_x)]
    bush_positions: list[tuple[int, int]] = []

    if cfg.bush_cluster_count <= 0:
        return bushes, fruit, last_harvest_time, bush_positions

    candidates = [(gx, gy) for (gx, gy) in main_region if objective[gx][gy] == 0]
    if not candidates:
        return bushes, fruit, last_harvest_time, bush_positions

    cluster_count = min(cfg.bush_cluster_count, len(candidates))
    centers = rng.sample(candidates, cluster_count)
    r2 = cfg.bush_cluster_radius * cfg.bush_cluster_radius

    allowed = set()
    for (gx, gy) in candidates:
        for (cx, cy) in centers:
            dx = gx - cx
            dy = gy - cy
            if dx * dx + dy * dy <= r2:
                allowed.add((gx, gy))
                break

    if not allowed:
        allowed = set(candidates)

    for (gx, gy) in allowed:
        if rng.random() < cfg.bush_density:
            bushes[gx][gy] = True
            fruit[gx][gy] = cfg.fruit_enabled
            last_harvest_time[gx][gy] = -1e9
            bush_positions.append((gx, gy))

    return bushes, fruit, last_harvest_time, bush_positions


def reset_bush_memory(cfg: Config):
    bush_known = [[False for _ in range(cfg.tiles_y)] for _ in range(cfg.tiles_x)]
    bush_last_seen_time = [[-1e9 for _ in range(cfg.tiles_y)] for _ in range(cfg.tiles_x)]
    bush_last_seen_fruit = [[False for _ in range(cfg.tiles_y)] for _ in range(cfg.tiles_x)]
    bush_last_harvest_time = [[-1e9 for _ in range(cfg.tiles_y)] for _ in range(cfg.tiles_x)]
    bush_last_empty_time = [[-1e9 for _ in range(cfg.tiles_y)] for _ in range(cfg.tiles_x)]
    bush_last_checked_empty_time = [[-1e9 for _ in range(cfg.tiles_y)] for _ in range(cfg.tiles_x)]
    return (
        bush_known,
        bush_last_seen_time,
        bush_last_seen_fruit,
        bush_last_harvest_time,
        bush_last_empty_time,
        bush_last_checked_empty_time,
    )


def observe_bush_at(cfg: Config, state, cx: int, cy: int, game_time: float) -> None:
    if not state.bushes[cx][cy]:
        return

    state.bush_known[cx][cy] = True
    state.bush_last_seen_time[cx][cy] = game_time

    if state.fruit[cx][cy]:
        if not state.bush_last_seen_fruit[cx][cy] and state.bush_last_empty_time[cx][cy] >= 0.0:
            sample = game_time - state.bush_last_empty_time[cx][cy]
            sample = clamp(sample, 2.0, 300.0)
            if state.respawn_estimate_sec is None:
                state.respawn_estimate_sec = sample
            else:
                state.respawn_estimate_sec = (
                    (1.0 - state.respawn_ema_alpha) * state.respawn_estimate_sec
                    + state.respawn_ema_alpha * sample
                )
            if state.respawn_estimate_sec is not None:
                state.respawn_estimate_sec = max(state.respawn_estimate_sec, state.respawn_lower_bound)
        state.bush_last_seen_fruit[cx][cy] = True
    else:
        state.bush_last_checked_empty_time[cx][cy] = game_time
        if state.bush_last_empty_time[cx][cy] >= 0.0:
            elapsed = game_time - state.bush_last_empty_time[cx][cy]
            if elapsed > state.respawn_lower_bound:
                state.respawn_lower_bound = elapsed
        if state.bush_last_seen_fruit[cx][cy] or state.bush_last_empty_time[cx][cy] < 0.0:
            state.bush_last_empty_time[cx][cy] = game_time
        state.bush_last_seen_fruit[cx][cy] = False


def observe_bushes_from_cells(cfg: Config, state, cells, game_time: float) -> None:
    for (cx, cy) in cells:
        if in_bounds_cell(cfg, cx, cy):
            observe_bush_at(cfg, state, cx, cy, game_time)


def update_fruit_respawn(cfg: Config, state, game_time: float) -> None:
    if not cfg.fruit_enabled:
        return
    for (cx, cy) in state.bush_positions:
        if not state.fruit[cx][cy]:
            if game_time - state.last_harvest_time[cx][cy] >= cfg.fruit_respawn_sec:
                state.fruit[cx][cy] = True


def pick_best_forage(cfg: Config, state, dist_known, game_time: float, hunger: float, current_cell):
    best_cell = None
    best_score = 0.0
    if not cfg.fruit_enabled:
        return None, 0.0
    if hunger <= 0.0:
        return None, 0.0

    for (cx, cy) in state.bush_positions:
        if not state.bush_known[cx][cy]:
            continue
        if dist_known[cx][cy] == -1:
            continue
        if (cx, cy) == current_cell:
            continue

        expected_value = 0.0
        if state.bush_last_seen_fruit[cx][cy]:
            expected_value = 1.0
        else:
            if state.respawn_estimate_sec is not None:
                estimate = max(state.respawn_estimate_sec, state.respawn_lower_bound)
                if state.bush_last_checked_empty_time[cx][cy] >= 0.0:
                    elapsed = game_time - state.bush_last_checked_empty_time[cx][cy]
                elif state.bush_last_empty_time[cx][cy] >= 0.0:
                    elapsed = game_time - state.bush_last_empty_time[cx][cy]
                else:
                    elapsed = max(0.0, game_time - state.bush_last_seen_time[cx][cy])
                expected_value = clamp(elapsed / max(1e-6, estimate), 0.0, 1.0)
            else:
                expected_value = 0.0

        if expected_value <= 0.0:
            continue

        score = cfg.forage_weight * hunger * expected_value / (1.0 + dist_known[cx][cy])
        if score > best_score:
            best_score = score
            best_cell = (cx, cy)

    return best_cell, best_score
