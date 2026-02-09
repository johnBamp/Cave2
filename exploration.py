from __future__ import annotations

from config import Config
from grid import in_bounds_cell, is_known_open


def is_frontier_cell(cfg: Config, agent, cx: int, cy: int) -> bool:
    if not in_bounds_cell(cfg, cx, cy):
        return False
    if not is_known_open(cfg, agent, cx, cy):
        return False
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nx, ny = cx + dx, cy + dy
        if in_bounds_cell(cfg, nx, ny) and not agent.ever_seen[nx][ny]:
            return True
    return False


def find_frontier_clusters(cfg: Config, agent, dist):
    visited = [[False for _ in range(cfg.tiles_y)] for _ in range(cfg.tiles_x)]
    clusters = []

    for gx in range(cfg.tiles_x):
        for gy in range(cfg.tiles_y):
            if visited[gx][gy]:
                continue
            if dist[gx][gy] == -1:
                continue
            if not is_frontier_cell(cfg, agent, gx, gy):
                continue

            q = [(gx, gy)]
            visited[gx][gy] = True
            cluster = []

            while q:
                x, y = q.pop(0)
                cluster.append((x, y))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if not in_bounds_cell(cfg, nx, ny):
                        continue
                    if visited[nx][ny]:
                        continue
                    if dist[nx][ny] == -1:
                        continue
                    if is_frontier_cell(cfg, agent, nx, ny):
                        visited[nx][ny] = True
                        q.append((nx, ny))

            clusters.append(cluster)

    return clusters


def pick_cluster_representative(cluster, dist, avoid=None):
    if avoid is not None:
        candidates = [c for c in cluster if c != avoid]
        if candidates:
            return min(candidates, key=lambda c: dist[c[0]][c[1]])
    return min(cluster, key=lambda c: dist[c[0]][c[1]])


def pick_exploration_target_from_dist(cfg: Config, agent, current_cell, dist):
    clusters = find_frontier_clusters(cfg, agent, dist)
    if not clusters:
        return None

    candidates = []
    for cluster in clusters:
        rep = pick_cluster_representative(cluster, dist, avoid=current_cell)
        if rep == current_cell:
            continue
        d = dist[rep[0]][rep[1]]
        size_bonus = -0.25 * len(cluster)
        candidates.append((d + size_bonus, rep))

    if not candidates:
        return None

    return min(candidates, key=lambda item: item[0])[1]


def pick_frontier_near_home(cfg: Config, agent, dist, home_cell, radius: int):
    clusters = find_frontier_clusters(cfg, agent, dist)
    best = None
    best_dist = None
    for cluster in clusters:
        rep = pick_cluster_representative(cluster, dist, avoid=None)
        if rep is None:
            continue
        if abs(rep[0] - home_cell[0]) + abs(rep[1] - home_cell[1]) > radius:
            continue
        d = dist[rep[0]][rep[1]]
        if d == -1:
            continue
        if best_dist is None or d < best_dist:
            best_dist = d
            best = rep
    return best


def pick_loiter_target(cfg: Config, agent, dist, home_cell, radius: int, current_cell):
    best = None
    best_dist = None
    hx, hy = home_cell
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            cx, cy = hx + dx, hy + dy
            if not in_bounds_cell(cfg, cx, cy):
                continue
            if not is_known_open(cfg, agent, cx, cy):
                continue
            if (cx, cy) == current_cell:
                continue
            d = dist[cx][cy]
            if d == -1:
                continue
            if best_dist is None or d < best_dist:
                best_dist = d
                best = (cx, cy)
    return best


def pick_nearest_known_bush(cfg: Config, state, agent, dist, current_cell, home_cell=None, radius=None):
    best = None
    best_dist = None
    for (cx, cy) in state.bush_positions:
        if not state.bush_known[cx][cy]:
            continue
        if (cx, cy) == current_cell:
            continue
        if dist[cx][cy] == -1:
            continue
        if home_cell is not None and radius is not None:
            if abs(cx - home_cell[0]) + abs(cy - home_cell[1]) > radius:
                continue
        d = dist[cx][cy]
        if best_dist is None or d < best_dist:
            best_dist = d
            best = (cx, cy)
    return best


def pick_nearest_known_open(cfg: Config, state, agent, dist, current_cell, home_cell=None, radius=None, avoid_bushes=False):
    best = None
    best_dist = None
    for gx in range(cfg.tiles_x):
        for gy in range(cfg.tiles_y):
            if not is_known_open(cfg, agent, gx, gy):
                continue
            if (gx, gy) == current_cell:
                continue
            if avoid_bushes and state.bushes[gx][gy]:
                continue
            if dist[gx][gy] == -1:
                continue
            if home_cell is not None and radius is not None:
                if abs(gx - home_cell[0]) + abs(gy - home_cell[1]) > radius:
                    continue
            d = dist[gx][gy]
            if best_dist is None or d < best_dist:
                best_dist = d
                best = (gx, gy)
    return best
