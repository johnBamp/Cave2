import pygame
import sys
import math
import random
from collections import deque

import settings as cfg

# Window
WIDTH, HEIGHT = cfg.WIDTH, cfg.HEIGHT
screen = None

# World
tilesX = cfg.TILES_X
tilesY = cfg.TILES_Y
TILE_SIZE = cfg.TILE_SIZE
WORLD_W = tilesX * TILE_SIZE
WORLD_H = tilesY * TILE_SIZE

# Cell types
EMPTY = 0
WALL = 1

# Colors
GRID = cfg.GRID_COLOR
BLACK = cfg.BLACK
PLAYER_COLOR = cfg.PLAYER_COLOR
UNKNOWN_FILL = cfg.UNKNOWN_FILL
TARGET_COLOR = cfg.TARGET_COLOR

# Raycast
FOV_DEG = cfg.FOV_DEG
RAYS = cfg.RAYS
STEP = cfg.RAY_STEP
VISION_RANGE = cfg.VISION_RANGE

# Movement
ROT_SPEED = math.radians(cfg.ROT_SPEED_DEG)
MOVE_SPEED = cfg.MOVE_SPEED
PLAYER_RADIUS = TILE_SIZE * cfg.PLAYER_RADIUS_RATIO
WAYPOINT_EPS = TILE_SIZE * cfg.WAYPOINT_EPS_RATIO
STUCK_FRAMES = cfg.STUCK_FRAMES

# Cave generation
FILL_PROB = cfg.FILL_PROB
SMOOTH_PASSES = cfg.SMOOTH_PASSES
SMOOTH_WALL_THRESHOLD = cfg.SMOOTH_WALL_THRESHOLD
SMOOTH_EMPTY_THRESHOLD = cfg.SMOOTH_EMPTY_THRESHOLD
MIN_REGION_SIZE = cfg.MIN_REGION_SIZE
FINAL_SMOOTH_PASSES = cfg.FINAL_SMOOTH_PASSES
NOISE_WALL_PROB = cfg.NOISE_WALL_PROB
NOISE_PASSES = cfg.NOISE_PASSES
CORRIDOR_RADIUS = cfg.CORRIDOR_RADIUS
SPAWN_CLEARANCE = cfg.SPAWN_CLEARANCE

# Belief update
EMPTY_ALPHA = cfg.EMPTY_ALPHA
WALL_ALPHA = cfg.WALL_ALPHA
WALL_CONF_THRESH = cfg.WALL_CONF_THRESH
MIN_ENCLOSED_SIZE = cfg.MIN_ENCLOSED_SIZE

# World state
objective = []  # objective[gx][gy]
subjective = []  # belief in [0,1]
ever_seen = []  # boolean


# -------------------- Utilities --------------------

def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def in_bounds_cell(cx, cy):
    return 0 <= cx < tilesX and 0 <= cy < tilesY


def is_wall_cell(cx, cy):
    if not in_bounds_cell(cx, cy):
        return True
    return objective[cx][cy] == WALL


def world_to_cell(px, py):
    cx = int(px // TILE_SIZE)
    cy = int(py // TILE_SIZE)
    return cx, cy


def cell_center(cx, cy):
    return (cx + 0.5) * TILE_SIZE, (cy + 0.5) * TILE_SIZE


def rotate_toward(current, target, max_delta):
    diff = (target - current + math.pi) % (2 * math.pi) - math.pi
    if abs(diff) <= max_delta:
        return target
    return current + max_delta * (1 if diff > 0 else -1)


# -------------------- Cave Generation --------------------

def count_wall_neighbors(grid, cx, cy):
    count = 0
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < tilesX and 0 <= ny < tilesY):
                count += 1
            elif grid[nx][ny] == WALL:
                count += 1
    return count


def smooth_grid(grid, wall_threshold, empty_threshold):
    new_grid = [[WALL for _ in range(tilesY)] for _ in range(tilesX)]
    for gx in range(1, tilesX - 1):
        for gy in range(1, tilesY - 1):
            walls = count_wall_neighbors(grid, gx, gy)
            if walls >= wall_threshold:
                new_grid[gx][gy] = WALL
            elif walls <= empty_threshold:
                new_grid[gx][gy] = EMPTY
            else:
                new_grid[gx][gy] = grid[gx][gy]

    for gx in range(tilesX):
        new_grid[gx][0] = WALL
        new_grid[gx][tilesY - 1] = WALL
    for gy in range(tilesY):
        new_grid[0][gy] = WALL
        new_grid[tilesX - 1][gy] = WALL
    return new_grid


def sprinkle_walls(grid, rng, prob):
    for gx in range(1, tilesX - 1):
        for gy in range(1, tilesY - 1):
            if grid[gx][gy] == EMPTY and rng.random() < prob:
                grid[gx][gy] = WALL


def get_empty_regions(grid):
    visited = [[False for _ in range(tilesY)] for _ in range(tilesX)]
    regions = []
    for gx in range(1, tilesX - 1):
        for gy in range(1, tilesY - 1):
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
                    if 0 <= nx < tilesX and 0 <= ny < tilesY:
                        if not visited[nx][ny] and grid[nx][ny] == EMPTY:
                            visited[nx][ny] = True
                            q.append((nx, ny))
            regions.append(region)
    return regions


def carve_tunnel(grid, start, end, radius):
    x, y = start
    ex, ey = end

    def carve_at(cx, cy):
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if abs(dx) + abs(dy) > radius:
                    continue
                nx, ny = cx + dx, cy + dy
                if 0 < nx < tilesX - 1 and 0 < ny < tilesY - 1:
                    grid[nx][ny] = EMPTY

    while x != ex:
        carve_at(x, y)
        x += 1 if ex > x else -1
    while y != ey:
        carve_at(x, y)
        y += 1 if ey > y else -1
    carve_at(ex, ey)


def connect_regions(grid, regions, rng):
    if not regions:
        return []
    largest = max(regions, key=len)
    others = [r for r in regions if r is not largest]
    for region in others:
        start = rng.choice(region)
        end = min(largest, key=lambda p: abs(p[0] - start[0]) + abs(p[1] - start[1]))
        carve_tunnel(grid, start, end, CORRIDOR_RADIUS)
    return largest


def generate_cave(seed):
    rng = random.Random(seed)
    grid = [[WALL for _ in range(tilesY)] for _ in range(tilesX)]

    for gx in range(1, tilesX - 1):
        for gy in range(1, tilesY - 1):
            grid[gx][gy] = WALL if rng.random() < FILL_PROB else EMPTY

    for i in range(SMOOTH_PASSES):
        if i < SMOOTH_PASSES // 2:
            grid = smooth_grid(grid, SMOOTH_WALL_THRESHOLD, SMOOTH_EMPTY_THRESHOLD)
        else:
            grid = smooth_grid(grid, SMOOTH_WALL_THRESHOLD, SMOOTH_EMPTY_THRESHOLD + 1)

    regions = get_empty_regions(grid)
    for region in regions:
        if len(region) < MIN_REGION_SIZE:
            for (x, y) in region:
                grid[x][y] = WALL

    regions = get_empty_regions(grid)
    if not regions:
        return generate_cave(seed + 1)

    main_region = connect_regions(grid, regions, rng)

    for _ in range(NOISE_PASSES):
        sprinkle_walls(grid, rng, NOISE_WALL_PROB)
        grid = smooth_grid(grid, SMOOTH_WALL_THRESHOLD, SMOOTH_EMPTY_THRESHOLD + 1)

    regions = get_empty_regions(grid)
    if regions:
        main_region = connect_regions(grid, regions, rng)

    for _ in range(FINAL_SMOOTH_PASSES):
        grid = smooth_grid(grid, SMOOTH_WALL_THRESHOLD, SMOOTH_EMPTY_THRESHOLD + 1)

    for gx in range(tilesX):
        grid[gx][0] = WALL
        grid[gx][tilesY - 1] = WALL
    for gy in range(tilesY):
        grid[0][gy] = WALL
        grid[tilesX - 1][gy] = WALL

    regions = get_empty_regions(grid)
    if not regions:
        return generate_cave(seed + 2)
    main_region = max(regions, key=len)

    # Fill any fully enclosed empty regions (no path to main cave)
    for region in regions:
        if region is main_region:
            continue
        for (x, y) in region:
            grid[x][y] = WALL

    return grid, main_region


# -------------------- Perception --------------------

def cast_and_update(player_pos, player_angle_rad):
    seen_empty = set()
    seen_wall = set()

    half_fov = math.radians(FOV_DEG) / 2
    start_angle = player_angle_rad - half_fov
    end_angle = player_angle_rad + half_fov

    max_dist = min(VISION_RANGE, int(math.hypot(WORLD_W, WORLD_H)) + 10)
    steps = int(max_dist // STEP)

    for i in range(RAYS):
        t = i / (RAYS - 1) if RAYS > 1 else 0.5
        ang = start_angle + (end_angle - start_angle) * t
        dx = math.cos(ang)
        dy = math.sin(ang)

        x, y = player_pos
        for _ in range(steps):
            x += dx * STEP
            y += dy * STEP
            cx, cy = world_to_cell(x, y)
            if not in_bounds_cell(cx, cy):
                break
            if objective[cx][cy] == WALL:
                seen_wall.add((cx, cy))
                break
            seen_empty.add((cx, cy))

    for (cx, cy) in seen_empty:
        subjective[cx][cy] += (0.0 - subjective[cx][cy]) * EMPTY_ALPHA
        ever_seen[cx][cy] = True

    for (cx, cy) in seen_wall:
        subjective[cx][cy] += (1.0 - subjective[cx][cy]) * WALL_ALPHA
        ever_seen[cx][cy] = True

    return seen_empty | seen_wall


def is_confirmed_wall(cx, cy):
    if cx == 0 or cy == 0 or cx == tilesX - 1 or cy == tilesY - 1:
        return True
    return ever_seen[cx][cy] and subjective[cx][cy] >= WALL_CONF_THRESH


def is_known_open(cx, cy):
    return ever_seen[cx][cy] and subjective[cx][cy] < WALL_CONF_THRESH


def infer_enclosed_voids():
    visited = [[False for _ in range(tilesY)] for _ in range(tilesX)]
    q = deque()

    for gx in range(tilesX):
        for gy in range(tilesY):
            if is_known_open(gx, gy) and not visited[gx][gy]:
                visited[gx][gy] = True
                q.append((gx, gy))

    while q:
        x, y = q.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if in_bounds_cell(nx, ny) and not visited[nx][ny] and not is_confirmed_wall(nx, ny):
                visited[nx][ny] = True
                q.append((nx, ny))

    for gx in range(1, tilesX - 1):
        for gy in range(1, tilesY - 1):
            if is_confirmed_wall(gx, gy) or visited[gx][gy]:
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
                    if in_bounds_cell(nx, ny) and not visited[nx][ny] and not is_confirmed_wall(nx, ny):
                        visited[nx][ny] = True
                        dq.append((nx, ny))

            if len(region) >= MIN_ENCLOSED_SIZE:
                for (x, y) in region:
                    subjective[x][y] = 1.0
                    ever_seen[x][y] = True


# -------------------- Spawn --------------------

def find_spawn_cell(region, rng, clearance=2):
    candidates = region[:]
    rng.shuffle(candidates)
    for (cx, cy) in candidates:
        ok = True
        for dx in range(-clearance, clearance + 1):
            for dy in range(-clearance, clearance + 1):
                nx, ny = cx + dx, cy + dy
                if not in_bounds_cell(nx, ny) or objective[nx][ny] == WALL:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            return cx, cy
    return candidates[0]


# -------------------- Movement --------------------

def can_move_to(px, py):
    checks = [
        (PLAYER_RADIUS, 0),
        (-PLAYER_RADIUS, 0),
        (0, PLAYER_RADIUS),
        (0, -PLAYER_RADIUS),
        (PLAYER_RADIUS, PLAYER_RADIUS),
        (-PLAYER_RADIUS, PLAYER_RADIUS),
        (PLAYER_RADIUS, -PLAYER_RADIUS),
        (-PLAYER_RADIUS, -PLAYER_RADIUS),
    ]
    for ox, oy in checks:
        cx, cy = world_to_cell(px + ox, py + oy)
        if is_wall_cell(cx, cy):
            return False
    return True


def move_toward(pos, target_pos, dt):
    dx = target_pos[0] - pos[0]
    dy = target_pos[1] - pos[1]
    dist = math.hypot(dx, dy)
    if dist < 1e-6:
        return pos

    vx = dx / dist
    vy = dy / dist
    step_x = vx * MOVE_SPEED * dt
    step_y = vy * MOVE_SPEED * dt

    new_x = pos[0] + step_x
    new_y = pos[1]
    if can_move_to(new_x, new_y):
        pos = (new_x, new_y)

    new_x = pos[0]
    new_y = pos[1] + step_y
    if can_move_to(new_x, new_y):
        pos = (new_x, new_y)

    pos = (
        clamp(pos[0], TILE_SIZE, WORLD_W - TILE_SIZE),
        clamp(pos[1], TILE_SIZE, WORLD_H - TILE_SIZE),
    )
    return pos


def bfs_reachable(start):
    dist = [[-1 for _ in range(tilesY)] for _ in range(tilesX)]
    parent = [[None for _ in range(tilesY)] for _ in range(tilesX)]
    q = deque()
    dist[start[0]][start[1]] = 0
    q.append(start)

    while q:
        x, y = q.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if not in_bounds_cell(nx, ny):
                continue
            if is_confirmed_wall(nx, ny):
                continue
            if dist[nx][ny] == -1:
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


def pick_frontier(dist, target_cell):
    best = None
    best_score = None
    for gx in range(tilesX):
        for gy in range(tilesY):
            if dist[gx][gy] == -1:
                continue
            if ever_seen[gx][gy]:
                continue
            if target_cell is None:
                score = dist[gx][gy]
            else:
                score = dist[gx][gy] + abs(gx - target_cell[0]) + abs(gy - target_cell[1])
            if best_score is None or score < best_score:
                best_score = score
                best = (gx, gy)
    return best


# -------------------- Rendering --------------------

def draw_player(screen, pos, angle, cam_x, cam_y):
    px = int(pos[0] - cam_x)
    py = int(pos[1] - cam_y)
    pygame.draw.circle(screen, PLAYER_COLOR, (px, py), int(PLAYER_RADIUS))
    length = 40
    fx = px + int(math.cos(angle) * length)
    fy = py + int(math.sin(angle) * length)
    pygame.draw.line(screen, PLAYER_COLOR, (px, py), (fx, fy), 3)


def draw_world(screen, cam_x, cam_y, seen_cells):
    start_x = max(0, int(cam_x // TILE_SIZE))
    end_x = min(tilesX - 1, int((cam_x + WIDTH) // TILE_SIZE) + 1)
    start_y = max(0, int(cam_y // TILE_SIZE))
    end_y = min(tilesY - 1, int((cam_y + HEIGHT) // TILE_SIZE) + 1)

    for gx in range(start_x, end_x + 1):
        for gy in range(start_y, end_y + 1):
            sx = int(gx * TILE_SIZE - cam_x)
            sy = int(gy * TILE_SIZE - cam_y)
            rect = pygame.Rect(sx, sy, TILE_SIZE, TILE_SIZE)

            if ever_seen[gx][gy]:
                v = int(clamp(subjective[gx][gy], 0.0, 1.0) * 255)
                fill = (v, v, v)
            else:
                fill = UNKNOWN_FILL

            pygame.draw.rect(screen, fill, rect)
            pygame.draw.rect(screen, GRID, rect, 1)

            if (gx, gy) in seen_cells:
                pygame.draw.rect(screen, PLAYER_COLOR, rect, 2)


def draw_target(screen, target_cell, cam_x, cam_y):
    if target_cell is None:
        return
    gx, gy = target_cell
    sx = int(gx * TILE_SIZE - cam_x)
    sy = int(gy * TILE_SIZE - cam_y)
    rect = pygame.Rect(sx, sy, TILE_SIZE, TILE_SIZE)
    pygame.draw.rect(screen, TARGET_COLOR, rect, 2)


# -------------------- Main --------------------

def reset_belief():
    global subjective, ever_seen
    subjective = [[0.5 for _ in range(tilesY)] for _ in range(tilesX)]
    ever_seen = [[False for _ in range(tilesY)] for _ in range(tilesX)]


def build_world(seed):
    global objective
    objective, main_region = generate_cave(seed)
    reset_belief()
    return main_region


def parse_seed():
    if len(sys.argv) > 1:
        try:
            return int(sys.argv[1])
        except ValueError:
            return random.randrange(1 << 30)
    return random.randrange(1 << 30)


def main():
    pygame.init()
    global screen
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("AI Test")

    current_seed = parse_seed()
    rng = random.Random(current_seed)
    main_region = build_world(current_seed)
    spawn_cell = find_spawn_cell(main_region, rng, clearance=SPAWN_CLEARANCE)
    player_pos = cell_center(*spawn_cell)
    player_angle = 0.0
    target_cell = None
    search_state = "idle"
    stuck_frames = 0

    clock = pygame.time.Clock()
    running = True

    while running:
        dt = clock.tick(60) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                if len(sys.argv) > 1:
                    current_seed += 1
                else:
                    current_seed = random.randrange(1 << 30)
                rng = random.Random(current_seed)
                main_region = build_world(current_seed)
                spawn_cell = find_spawn_cell(main_region, rng, clearance=SPAWN_CLEARANCE)
                player_pos = cell_center(*spawn_cell)
                player_angle = 0.0
                target_cell = None
                search_state = "idle"
                stuck_frames = 0
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                cam_x = clamp(player_pos[0] - WIDTH / 2, 0, WORLD_W - WIDTH)
                cam_y = clamp(player_pos[1] - HEIGHT / 2, 0, WORLD_H - HEIGHT)
                world_x = cam_x + event.pos[0]
                world_y = cam_y + event.pos[1]
                cx, cy = world_to_cell(world_x, world_y)
                if in_bounds_cell(cx, cy):
                    target_cell = (cx, cy)
                    search_state = "searching"
                    stuck_frames = 0

        current_cell = world_to_cell(player_pos[0], player_pos[1])
        path = []
        if target_cell is not None and search_state not in ("reached", "unreachable"):
            if current_cell == target_cell and not is_wall_cell(*target_cell):
                search_state = "reached"
            elif is_confirmed_wall(target_cell[0], target_cell[1]) and ever_seen[target_cell[0]][target_cell[1]]:
                search_state = "unreachable"
            else:
                dist, parent = bfs_reachable(current_cell)
                if dist[target_cell[0]][target_cell[1]] != -1:
                    path = reconstruct_path(parent, current_cell, target_cell)
                else:
                    frontier = pick_frontier(dist, target_cell)
                    if frontier is not None:
                        path = reconstruct_path(parent, current_cell, frontier)
                    else:
                        search_state = "unreachable"

        if path and len(path) > 1:
            next_cell = path[1]
            target_pos = cell_center(next_cell[0], next_cell[1])
            desired_angle = math.atan2(target_pos[1] - player_pos[1], target_pos[0] - player_pos[0])
            player_angle = rotate_toward(player_angle, desired_angle, ROT_SPEED * dt)

            prev_pos = player_pos
            if math.hypot(target_pos[0] - player_pos[0], target_pos[1] - player_pos[1]) > WAYPOINT_EPS:
                player_pos = move_toward(player_pos, target_pos, dt)
            moved = math.hypot(player_pos[0] - prev_pos[0], player_pos[1] - prev_pos[1]) > 1e-3
            if moved:
                stuck_frames = 0
            else:
                stuck_frames += 1
                if stuck_frames >= STUCK_FRAMES:
                    subjective[next_cell[0]][next_cell[1]] = 1.0
                    ever_seen[next_cell[0]][next_cell[1]] = True
                    stuck_frames = 0

        seen_cells = cast_and_update(player_pos, player_angle)
        infer_enclosed_voids()

        cam_x = clamp(player_pos[0] - WIDTH / 2, 0, WORLD_W - WIDTH)
        cam_y = clamp(player_pos[1] - HEIGHT / 2, 0, WORLD_H - HEIGHT)

        screen.fill(BLACK)
        draw_world(screen, cam_x, cam_y, seen_cells)
        draw_target(screen, target_cell, cam_x, cam_y)
        draw_player(screen, player_pos, player_angle, cam_x, cam_y)
        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
