from __future__ import annotations

from dataclasses import dataclass
import math
import random

from config import Config
from grid import clamp, in_bounds_cell, world_to_cell, cell_center


@dataclass
class Animal:
    id: int
    pos: tuple[float, float]
    vel: tuple[float, float]
    speed: float
    alive: bool
    respawn_timer: float
    target_cell: tuple[int, int] | None
    path: list[tuple[int, int]]
    move_accum: float
    repath_timer: float


def _random_dir(rng: random.Random) -> tuple[float, float]:
    ang = rng.random() * 2.0 * math.pi
    return math.cos(ang), math.sin(ang)


def has_line_of_sight(cfg: Config, objective, start_pos, end_pos) -> bool:
    dx = end_pos[0] - start_pos[0]
    dy = end_pos[1] - start_pos[1]
    dist = math.hypot(dx, dy)
    if dist <= 1e-6:
        return True
    step = max(1.0, min(cfg.ray_step, cfg.tile_size / 2.0))
    steps = int(dist // step)
    vx = dx / dist
    vy = dy / dist
    x, y = start_pos
    for _ in range(steps):
        x += vx * step
        y += vy * step
        cx, cy = world_to_cell(cfg, x, y)
        if not in_bounds_cell(cfg, cx, cy):
            return False
        if objective[cx][cy] == 1:
            return False
    return True


def bfs_objective(cfg: Config, objective, start, goal):
    if start == goal:
        return [start]
    dist = [[-1 for _ in range(cfg.tiles_y)] for _ in range(cfg.tiles_x)]
    parent = [[None for _ in range(cfg.tiles_y)] for _ in range(cfg.tiles_x)]
    q = [start]
    dist[start[0]][start[1]] = 0
    while q:
        x, y = q.pop(0)
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if not in_bounds_cell(cfg, nx, ny):
                continue
            if objective[nx][ny] == 1:
                continue
            if dist[nx][ny] != -1:
                continue
            dist[nx][ny] = dist[x][y] + 1
            parent[nx][ny] = (x, y)
            if (nx, ny) == goal:
                q.clear()
                break
            q.append((nx, ny))

    if parent[goal[0]][goal[1]] is None:
        return []
    path = [goal]
    cur = goal
    while cur != start:
        cur = parent[cur[0]][cur[1]]
        if cur is None:
            return []
        path.append(cur)
    path.reverse()
    return path


def pick_random_open_near(cfg: Config, objective, center, radius: int, rng: random.Random):
    cx0, cy0 = center
    candidates = []
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            cx = cx0 + dx
            cy = cy0 + dy
            if not in_bounds_cell(cfg, cx, cy):
                continue
            if objective[cx][cy] == 1:
                continue
            candidates.append((cx, cy))
    if not candidates:
        return None
    return rng.choice(candidates)


def find_nearest_open(cfg: Config, objective, target_cell, radius: int):
    cx0, cy0 = target_cell
    for r in range(radius + 1):
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                cx = cx0 + dx
                cy = cy0 + dy
                if not in_bounds_cell(cfg, cx, cy):
                    continue
                if objective[cx][cy] == 0:
                    return (cx, cy)
    return None


def spawn_animals(cfg: Config, objective, main_region, rng: random.Random) -> list[Animal]:
    animals: list[Animal] = []
    candidates = [(gx, gy) for (gx, gy) in main_region if objective[gx][gy] == 0]
    if not candidates or cfg.animal_count <= 0:
        return animals

    rng.shuffle(candidates)
    count = min(cfg.animal_count, len(candidates))
    for idx in range(count):
        cx, cy = candidates[idx]
        pos = cell_center(cfg, cx, cy)
        dx, dy = _random_dir(rng)
        vel = (dx * cfg.animal_speed, dy * cfg.animal_speed)
        animals.append(
            Animal(
                id=idx,
                pos=pos,
                vel=vel,
                speed=cfg.animal_speed,
                alive=True,
                respawn_timer=0.0,
                target_cell=None,
                path=[],
                move_accum=0.0,
                repath_timer=0.0,
            )
        )
    return animals


def _attempt_move(cfg: Config, objective, animal: Animal, dt: float) -> bool:
    new_x = animal.pos[0] + animal.vel[0] * dt
    new_y = animal.pos[1] + animal.vel[1] * dt
    cx, cy = world_to_cell(cfg, new_x, new_y)
    if not in_bounds_cell(cfg, cx, cy):
        return False
    if objective[cx][cy] == 1:
        return False
    animal.pos = (
        clamp(new_x, cfg.tile_size, cfg.world_w - cfg.tile_size),
        clamp(new_y, cfg.tile_size, cfg.world_h - cfg.tile_size),
    )
    return True


def update_animals(cfg: Config, state, dt: float, rng: random.Random, player_pos=None) -> None:
    turn_prob = cfg.animal_turn_chance * dt
    for animal in state.animals:
        if not animal.alive:
            continue

        current_cell = world_to_cell(cfg, animal.pos[0], animal.pos[1])
        fleeing = False
        if player_pos is not None:
            dist = math.hypot(animal.pos[0] - player_pos[0], animal.pos[1] - player_pos[1])
            if dist <= cfg.animal_flee_radius * cfg.tile_size:
                if has_line_of_sight(cfg, state.objective, animal.pos, player_pos):
                    fleeing = True

        animal.repath_timer += dt

        if fleeing:
            if animal.repath_timer >= cfg.animal_repath_sec or animal.target_cell is None or not animal.path:
                dxp = animal.pos[0] - player_pos[0]
                dyp = animal.pos[1] - player_pos[1]
                dist = math.hypot(dxp, dyp)
                if dist > 1e-6:
                    nx = dxp / dist
                    ny = dyp / dist
                else:
                    nx, ny = _random_dir(rng)
                flee_dist = cfg.animal_flee_radius * cfg.tile_size
                target_pos = (animal.pos[0] + nx * flee_dist, animal.pos[1] + ny * flee_dist)
                target_cell = world_to_cell(cfg, target_pos[0], target_pos[1])
                target_cell = (
                    clamp(target_cell[0], 1, cfg.tiles_x - 2),
                    clamp(target_cell[1], 1, cfg.tiles_y - 2),
                )
                if state.objective[target_cell[0]][target_cell[1]] == 1:
                    target_cell = find_nearest_open(cfg, state.objective, target_cell, 4)
                if target_cell is not None:
                    path = bfs_objective(cfg, state.objective, current_cell, target_cell)
                    if path:
                        animal.target_cell = target_cell
                        animal.path = path
                        animal.repath_timer = 0.0
        else:
            if (
                animal.target_cell is None
                or current_cell == animal.target_cell
                or not animal.path
                or rng.random() < turn_prob
            ):
                target_cell = pick_random_open_near(
                    cfg, state.objective, current_cell, cfg.animal_wander_radius, rng
                )
                if target_cell is not None:
                    path = bfs_objective(cfg, state.objective, current_cell, target_cell)
                    if path:
                        animal.target_cell = target_cell
                        animal.path = path

        speed = animal.speed * (cfg.animal_flee_speed_mult if fleeing else 1.0)
        step_time = cfg.tile_size / max(1e-6, speed)
        animal.move_accum += dt
        while animal.move_accum >= step_time:
            animal.move_accum -= step_time
            if animal.path and len(animal.path) > 1:
                next_cell = animal.path[1]
                if state.objective[next_cell[0]][next_cell[1]] == 0:
                    animal.pos = cell_center(cfg, next_cell[0], next_cell[1])
                    animal.path = animal.path[1:]
                else:
                    animal.path = []
                    animal.target_cell = None
                    break
            else:
                animal.path = []
                animal.target_cell = None
                break


def respawn_animal(cfg: Config, state, animal: Animal, rng: random.Random, avoid_cell=None, avoid_radius=None) -> None:
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
    animal.pos = cell_center(cfg, cx, cy)
    dx, dy = _random_dir(rng)
    animal.vel = (dx * animal.speed, dy * animal.speed)
    animal.alive = True
    animal.respawn_timer = 0.0
