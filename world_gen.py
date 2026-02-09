from __future__ import annotations

import random
from collections import deque

from config import Config
from grid import in_bounds_cell


EMPTY = 0
WALL = 1


def count_wall_neighbors(cfg: Config, grid: list[list[int]], cx: int, cy: int) -> int:
    count = 0
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            nx, ny = cx + dx, cy + dy
            if not in_bounds_cell(cfg, nx, ny):
                count += 1
            elif grid[nx][ny] == WALL:
                count += 1
    return count


def smooth_grid(cfg: Config, grid: list[list[int]], wall_threshold: int, empty_threshold: int) -> list[list[int]]:
    new_grid = [[WALL for _ in range(cfg.tiles_y)] for _ in range(cfg.tiles_x)]
    for gx in range(1, cfg.tiles_x - 1):
        for gy in range(1, cfg.tiles_y - 1):
            walls = count_wall_neighbors(cfg, grid, gx, gy)
            if walls >= wall_threshold:
                new_grid[gx][gy] = WALL
            elif walls <= empty_threshold:
                new_grid[gx][gy] = EMPTY
            else:
                new_grid[gx][gy] = grid[gx][gy]

    for gx in range(cfg.tiles_x):
        new_grid[gx][0] = WALL
        new_grid[gx][cfg.tiles_y - 1] = WALL
    for gy in range(cfg.tiles_y):
        new_grid[0][gy] = WALL
        new_grid[cfg.tiles_x - 1][gy] = WALL
    return new_grid


def sprinkle_walls(cfg: Config, grid: list[list[int]], rng: random.Random, prob: float) -> None:
    for gx in range(1, cfg.tiles_x - 1):
        for gy in range(1, cfg.tiles_y - 1):
            if grid[gx][gy] == EMPTY and rng.random() < prob:
                grid[gx][gy] = WALL


def get_empty_regions(cfg: Config, grid: list[list[int]]) -> list[list[tuple[int, int]]]:
    visited = [[False for _ in range(cfg.tiles_y)] for _ in range(cfg.tiles_x)]
    regions = []
    for gx in range(1, cfg.tiles_x - 1):
        for gy in range(1, cfg.tiles_y - 1):
            if visited[gx][gy] or grid[gx][gy] == WALL:
                continue
            region = []
            q = deque()
            q.append((gx, gy))
            visited[gx][gy] = True
            while q:
                x, y = q.popleft()
                region.append((x, y))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if in_bounds_cell(cfg, nx, ny) and not visited[nx][ny] and grid[nx][ny] == EMPTY:
                        visited[nx][ny] = True
                        q.append((nx, ny))
            regions.append(region)
    return regions


def carve_tunnel(cfg: Config, grid: list[list[int]], start, end, radius: int) -> None:
    x, y = start
    ex, ey = end

    def carve_at(cx, cy):
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if abs(dx) + abs(dy) > radius:
                    continue
                nx, ny = cx + dx, cy + dy
                if 0 < nx < cfg.tiles_x - 1 and 0 < ny < cfg.tiles_y - 1:
                    grid[nx][ny] = EMPTY

    while x != ex:
        carve_at(x, y)
        x += 1 if ex > x else -1
    while y != ey:
        carve_at(x, y)
        y += 1 if ey > y else -1
    carve_at(ex, ey)


def connect_regions(cfg: Config, grid: list[list[int]], regions, rng: random.Random):
    if not regions:
        return []
    largest = max(regions, key=len)
    others = [r for r in regions if r is not largest]
    for region in others:
        start = rng.choice(region)
        end = min(largest, key=lambda p: abs(p[0] - start[0]) + abs(p[1] - start[1]))
        carve_tunnel(cfg, grid, start, end, cfg.corridor_radius)
    return largest


def generate_cave(cfg: Config, seed: int):
    rng = random.Random(seed)
    grid = [[WALL for _ in range(cfg.tiles_y)] for _ in range(cfg.tiles_x)]

    for gx in range(1, cfg.tiles_x - 1):
        for gy in range(1, cfg.tiles_y - 1):
            grid[gx][gy] = WALL if rng.random() < cfg.fill_prob else EMPTY

    for i in range(cfg.smooth_passes):
        if i < cfg.smooth_passes // 2:
            grid = smooth_grid(cfg, grid, cfg.smooth_wall_threshold, cfg.smooth_empty_threshold)
        else:
            grid = smooth_grid(cfg, grid, cfg.smooth_wall_threshold, cfg.smooth_empty_threshold + 1)

    regions = get_empty_regions(cfg, grid)
    for region in regions:
        if len(region) < cfg.min_region_size:
            for (x, y) in region:
                grid[x][y] = WALL

    regions = get_empty_regions(cfg, grid)
    if not regions:
        return generate_cave(cfg, seed + 1)

    connect_regions(cfg, grid, regions, rng)

    for _ in range(cfg.noise_passes):
        sprinkle_walls(cfg, grid, rng, cfg.noise_wall_prob)
        grid = smooth_grid(cfg, grid, cfg.smooth_wall_threshold, cfg.smooth_empty_threshold + 1)

    regions = get_empty_regions(cfg, grid)
    if regions:
        connect_regions(cfg, grid, regions, rng)

    for _ in range(cfg.final_smooth_passes):
        grid = smooth_grid(cfg, grid, cfg.smooth_wall_threshold, cfg.smooth_empty_threshold + 1)

    for gx in range(cfg.tiles_x):
        grid[gx][0] = WALL
        grid[gx][cfg.tiles_y - 1] = WALL
    for gy in range(cfg.tiles_y):
        grid[0][gy] = WALL
        grid[cfg.tiles_x - 1][gy] = WALL

    regions = get_empty_regions(cfg, grid)
    if not regions:
        return generate_cave(cfg, seed + 2)
    main_region = max(regions, key=len)

    # Fill any fully enclosed empty regions (no path to main cave)
    for region in regions:
        if region is main_region:
            continue
        for (x, y) in region:
            grid[x][y] = WALL

    return grid, main_region


def find_spawn_cell(cfg: Config, objective, region, rng: random.Random, clearance: int):
    candidates = region[:]
    rng.shuffle(candidates)
    for (cx, cy) in candidates:
        ok = True
        for dx in range(-clearance, clearance + 1):
            for dy in range(-clearance, clearance + 1):
                nx, ny = cx + dx, cy + dy
                if not in_bounds_cell(cfg, nx, ny) or objective[nx][ny] == WALL:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            return cx, cy
    return candidates[0]
