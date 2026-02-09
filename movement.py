from __future__ import annotations

from collections import deque

from config import Config
from grid import in_bounds_cell, is_known_open, is_wall_cell


def bfs_known_open(cfg: Config, agent, start):
    dist = [[-1 for _ in range(cfg.tiles_y)] for _ in range(cfg.tiles_x)]
    parent = [[None for _ in range(cfg.tiles_y)] for _ in range(cfg.tiles_x)]
    q = deque()

    sx, sy = start
    dist[sx][sy] = 0
    q.append((sx, sy))

    while q:
        x, y = q.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if not in_bounds_cell(cfg, nx, ny):
                continue
            if dist[nx][ny] != -1:
                continue
            if not is_known_open(cfg, agent, nx, ny):
                continue
            dist[nx][ny] = dist[x][y] + 1
            parent[nx][ny] = (x, y)
            q.append((nx, ny))

    return dist, parent


def reconstruct_path(parent, start, goal):
    if start == goal:
        return [start]
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


def can_move_to(cfg: Config, state, px: float, py: float, player_radius: float) -> bool:
    checks = [
        (player_radius, 0),
        (-player_radius, 0),
        (0, player_radius),
        (0, -player_radius),
        (player_radius, player_radius),
        (-player_radius, player_radius),
        (player_radius, -player_radius),
        (-player_radius, -player_radius),
    ]
    for ox, oy in checks:
        cx = int((px + ox) // cfg.tile_size)
        cy = int((py + oy) // cfg.tile_size)
        if is_wall_cell(cfg, state, cx, cy):
            return False
    return True
