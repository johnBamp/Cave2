from __future__ import annotations

import math
from collections import deque

from config import Config
from grid import in_bounds_cell, world_to_cell, is_confirmed_wall, is_known_open
from foraging import observe_bush_at


def cast_and_update(cfg: Config, state, agent):
    seen_empty = set()
    seen_wall = set()
    newly_seen = 0

    half_fov = math.radians(cfg.fov_deg) / 2.0
    start_angle = agent.angle - half_fov
    end_angle = agent.angle + half_fov

    max_dist = min(cfg.vision_range, int(math.hypot(cfg.world_w, cfg.world_h)) + 10)
    steps = int(max_dist // cfg.ray_step)

    for i in range(cfg.rays):
        t = i / (cfg.rays - 1) if cfg.rays > 1 else 0.5
        ang = start_angle + (end_angle - start_angle) * t
        dx = math.cos(ang)
        dy = math.sin(ang)

        x, y = agent.pos
        for _ in range(steps):
            x += dx * cfg.ray_step
            y += dy * cfg.ray_step
            cx, cy = world_to_cell(cfg, x, y)
            if not in_bounds_cell(cfg, cx, cy):
                break
            if state.objective[cx][cy] == 1:
                seen_wall.add((cx, cy))
                break
            seen_empty.add((cx, cy))

    for (cx, cy) in seen_empty:
        if not agent.ever_seen[cx][cy]:
            newly_seen += 1
        agent.subjective[cx][cy] += (0.0 - agent.subjective[cx][cy]) * cfg.empty_alpha
        agent.ever_seen[cx][cy] = True

    for (cx, cy) in seen_wall:
        if not agent.ever_seen[cx][cy]:
            newly_seen += 1
        agent.subjective[cx][cy] += (1.0 - agent.subjective[cx][cy]) * cfg.wall_alpha
        agent.ever_seen[cx][cy] = True

    return seen_empty | seen_wall, newly_seen


def infer_enclosed_voids(cfg: Config, agent) -> None:
    visited = [[False for _ in range(cfg.tiles_y)] for _ in range(cfg.tiles_x)]
    q = deque()

    for gx in range(cfg.tiles_x):
        for gy in range(cfg.tiles_y):
            if is_known_open(cfg, agent, gx, gy) and not visited[gx][gy]:
                visited[gx][gy] = True
                q.append((gx, gy))

    while q:
        x, y = q.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if in_bounds_cell(cfg, nx, ny) and not visited[nx][ny] and not is_confirmed_wall(cfg, agent, nx, ny):
                visited[nx][ny] = True
                q.append((nx, ny))

    for gx in range(1, cfg.tiles_x - 1):
        for gy in range(1, cfg.tiles_y - 1):
            if is_confirmed_wall(cfg, agent, gx, gy) or visited[gx][gy]:
                continue
            region = []
            dq = deque()
            dq.append((gx, gy))
            visited[gx][gy] = True
            while dq:
                x, y = dq.popleft()
                region.append((x, y))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if in_bounds_cell(cfg, nx, ny) and not visited[nx][ny] and not is_confirmed_wall(cfg, agent, nx, ny):
                        visited[nx][ny] = True
                        dq.append((nx, ny))

            if len(region) >= cfg.min_enclosed_size:
                for (x, y) in region:
                    agent.subjective[x][y] = 1.0
                    agent.ever_seen[x][y] = True


def observe_local_3x3(cfg: Config, state, agent, current_cell, game_time: float) -> int:
    newly_seen = 0
    cx0, cy0 = current_cell
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            cx, cy = cx0 + dx, cy0 + dy
            if not in_bounds_cell(cfg, cx, cy):
                continue
            if not agent.ever_seen[cx][cy]:
                newly_seen += 1
            if state.objective[cx][cy] == 1:
                agent.subjective[cx][cy] = 1.0
            else:
                agent.subjective[cx][cy] = 0.0
            agent.ever_seen[cx][cy] = True
            if state.bushes[cx][cy]:
                observe_bush_at(cfg, state, cx, cy, game_time)
    return newly_seen
