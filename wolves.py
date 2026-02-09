from __future__ import annotations

from dataclasses import dataclass
import math
import random

from config import Config
from grid import clamp, in_bounds_cell, world_to_cell, cell_center
from animals import has_line_of_sight, bfs_objective, pick_random_open_near, find_nearest_open


@dataclass
class Wolf:
    id: int
    pos: tuple[float, float]
    vel: tuple[float, float]
    speed: float
    alive: bool
    target_cell: tuple[int, int] | None
    path: list[tuple[int, int]]
    move_accum: float
    repath_timer: float
    hp: float
    attack_cooldown: float


def spawn_wolves(cfg: Config, objective, main_region, rng: random.Random) -> list[Wolf]:
    wolves: list[Wolf] = []
    if cfg.wolf_count <= 0:
        return wolves

    candidates = [(gx, gy) for (gx, gy) in main_region if objective[gx][gy] == 0]
    if not candidates:
        return wolves

    rng.shuffle(candidates)
    count = min(cfg.wolf_count, len(candidates))
    for idx in range(count):
        cx, cy = candidates[idx]
        pos = cell_center(cfg, cx, cy)
        wolves.append(
            Wolf(
                id=idx,
                pos=pos,
                vel=(0.0, 0.0),
                speed=cfg.wolf_speed,
                alive=True,
                target_cell=None,
                path=[],
                move_accum=0.0,
                repath_timer=0.0,
                hp=cfg.wolf_max_hp,
                attack_cooldown=0.0,
            )
        )
    return wolves


def wolf_can_see_player(cfg: Config, objective, wolf: Wolf, player_pos) -> bool:
    dist = math.hypot(wolf.pos[0] - player_pos[0], wolf.pos[1] - player_pos[1])
    if dist > cfg.wolf_sight_range_tiles * cfg.tile_size:
        return False
    return has_line_of_sight(cfg, objective, wolf.pos, player_pos)


def update_wolves(cfg: Config, state, dt: float, rng: random.Random, player_pos) -> None:
    for wolf in state.wolves:
        if not wolf.alive:
            continue

        wolf.repath_timer += dt
        wolf.attack_cooldown = max(0.0, wolf.attack_cooldown - dt)
        current_cell = world_to_cell(cfg, wolf.pos[0], wolf.pos[1])

        dist_to_player = math.hypot(wolf.pos[0] - player_pos[0], wolf.pos[1] - player_pos[1])
        if dist_to_player <= cfg.wolf_attack_range_tiles * cfg.tile_size:
            wolf.path = []
            wolf.target_cell = None
            continue
        can_hear = dist_to_player <= cfg.wolf_hearing_range_tiles * cfg.tile_size
        can_see = dist_to_player <= cfg.wolf_sight_range_tiles * cfg.tile_size and has_line_of_sight(
            cfg, state.objective, wolf.pos, player_pos
        )

        if can_see or can_hear:
            if wolf.repath_timer >= cfg.wolf_repath_sec or not wolf.path:
                target_cell = world_to_cell(cfg, player_pos[0], player_pos[1])
                if not in_bounds_cell(cfg, target_cell[0], target_cell[1]) or state.objective[
                    target_cell[0]
                ][target_cell[1]] == 1:
                    target_cell = find_nearest_open(cfg, state.objective, target_cell, 3)
                if target_cell is not None:
                    path = bfs_objective(cfg, state.objective, current_cell, target_cell)
                    if path:
                        wolf.target_cell = target_cell
                        wolf.path = path
                        wolf.repath_timer = 0.0
        else:
            if wolf.target_cell is None or current_cell == wolf.target_cell or not wolf.path:
                target_cell = pick_random_open_near(
                    cfg, state.objective, current_cell, cfg.wolf_wander_radius, rng
                )
                if target_cell is not None:
                    path = bfs_objective(cfg, state.objective, current_cell, target_cell)
                    if path:
                        wolf.target_cell = target_cell
                        wolf.path = path

        step_time = cfg.tile_size / max(1e-6, wolf.speed)
        wolf.move_accum += dt
        while wolf.move_accum >= step_time:
            wolf.move_accum -= step_time
            if wolf.path and len(wolf.path) > 1:
                next_cell = wolf.path[1]
                if state.objective[next_cell[0]][next_cell[1]] == 0:
                    wolf.pos = cell_center(cfg, next_cell[0], next_cell[1])
                    wolf.path = wolf.path[1:]
                else:
                    wolf.path = []
                    wolf.target_cell = None
                    break
            else:
                wolf.path = []
                wolf.target_cell = None
                break


def wolf_attack(cfg: Config, state, wolf: Wolf, dt: float) -> None:
    if not wolf.alive:
        return
    dist = math.hypot(wolf.pos[0] - state.player_pos[0], wolf.pos[1] - state.player_pos[1])
    if dist > cfg.wolf_attack_range_tiles * cfg.tile_size:
        return
    if wolf.attack_cooldown > 0.0:
        return
    damage = cfg.wolf_dps * cfg.wolf_attack_cooldown
    state.player_hp = max(0.0, state.player_hp - damage)
    wolf.attack_cooldown = cfg.wolf_attack_cooldown


def respawn_wolf(cfg: Config, state, wolf: Wolf, rng: random.Random, avoid_cell=None, avoid_radius=None) -> None:
    candidates = []
    for gx in range(cfg.tiles_x):
        for gy in range(cfg.tiles_y):
            if state.objective[gx][gy] != 0:
                continue
            if avoid_cell is not None and avoid_radius is not None:
                if abs(gx - avoid_cell[0]) + abs(gy - avoid_cell[1]) <= avoid_radius:
                    continue
            candidates.append((gx, gy))

    if not candidates:
        return

    cx, cy = rng.choice(candidates)
    wolf.pos = cell_center(cfg, cx, cy)
    wolf.target_cell = None
    wolf.path = []
    wolf.move_accum = 0.0
    wolf.repath_timer = 0.0
    wolf.hp = cfg.wolf_max_hp
    wolf.attack_cooldown = 0.0
    wolf.alive = True
