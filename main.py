from __future__ import annotations

import math
import random
import sys
from collections import deque

import pygame

from config import load_config
from state import init_state
from grid import (
    clamp,
    world_to_cell,
    cell_center,
    rotate_toward,
    is_confirmed_wall,
    is_wall_cell,
    is_known_open,
    in_bounds_cell,
)
from movement import bfs_known_open, reconstruct_path
from perception import cast_and_update, infer_enclosed_voids, observe_local_3x3
from foraging import observe_bushes_from_cells, update_fruit_respawn, pick_best_forage
from exploration import (
    find_frontier_clusters,
    pick_cluster_representative,
    pick_exploration_target_from_dist,
    pick_frontier_near_home,
    pick_loiter_target,
    pick_nearest_known_bush,
    pick_nearest_known_open,
)
from animals import update_animals, respawn_animal
from rendering import draw_world, draw_player, draw_target, draw_frontiers, draw_animals
from sound import emit_sound, process_hearing


def parse_seed() -> int:
    if len(sys.argv) > 1:
        try:
            return int(sys.argv[1])
        except ValueError:
            return random.randrange(1 << 30)
    return random.randrange(1 << 30)


def score_unknown_cone(cfg, state, angle: float, probe_rays: int = 9) -> float:
    half_fov = math.radians(cfg.fov_deg) / 2.0
    max_dist = cfg.vision_range
    steps = int(max_dist // cfg.ray_step)
    score = 0.0

    for i in range(probe_rays):
        t = i / (probe_rays - 1) if probe_rays > 1 else 0.5
        ang = angle - half_fov + (2 * half_fov) * t
        dx = math.cos(ang)
        dy = math.sin(ang)

        x, y = state.player_pos
        for s in range(steps):
            x += dx * cfg.ray_step
            y += dy * cfg.ray_step
            cx, cy = world_to_cell(cfg, x, y)
            if not (0 <= cx < cfg.tiles_x and 0 <= cy < cfg.tiles_y):
                break
            if state.objective[cx][cy] == 1:
                break
            if not state.ever_seen[cx][cy]:
                score += 1.0 / (1.0 + s)

    return score


def best_unknown_angle(cfg, state, samples: int = 24):
    best_angle = None
    best_score = 0.0
    for i in range(samples):
        ang = (2 * math.pi) * (i / samples)
        score = score_unknown_cone(cfg, state, ang)
        if score > best_score:
            best_score = score
            best_angle = ang
    if best_score <= 0.0:
        return None
    return best_angle


def pick_random_known_open_in_radius(cfg, state, center, radius, rng, avoid_cell=None):
    cx0, cy0 = center
    candidates = []
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            cx = cx0 + dx
            cy = cy0 + dy
            if not (0 <= cx < cfg.tiles_x and 0 <= cy < cfg.tiles_y):
                continue
            if avoid_cell is not None and (cx, cy) == avoid_cell:
                continue
            if not is_known_open(cfg, state, cx, cy):
                continue
            candidates.append((cx, cy))
    if not candidates:
        return None
    return rng.choice(candidates)


def project_to_known_open(cfg, state, start_cell, max_radius):
    sx, sy = start_cell
    if not (0 <= sx < cfg.tiles_x and 0 <= sy < cfg.tiles_y):
        return None
    if is_known_open(cfg, state, sx, sy):
        return (sx, sy)
    dist = [[-1 for _ in range(cfg.tiles_y)] for _ in range(cfg.tiles_x)]
    q = deque()
    dist[sx][sy] = 0
    q.append((sx, sy))
    while q:
        x, y = q.popleft()
        if dist[x][y] > max_radius:
            continue
        if is_known_open(cfg, state, x, y):
            return (x, y)
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if not (0 <= nx < cfg.tiles_x and 0 <= ny < cfg.tiles_y):
                continue
            if dist[nx][ny] != -1:
                continue
            dist[nx][ny] = dist[x][y] + 1
            q.append((nx, ny))
    return None


def pick_ring_search_cell(cfg, state, center, start_radius, bias_heading, bias_deg):
    if center is None:
        return None
    cx0, cy0 = center
    max_radius = start_radius + max(2, cfg.hunt_search_radius * 2)
    bias_rad = math.radians(bias_deg)

    def within_bias(angle):
        diff = (angle - bias_heading + math.pi) % (2 * math.pi) - math.pi
        return abs(diff) <= bias_rad

    for r in range(max(1, start_radius), max_radius + 1, cfg.animal_search_ring_step):
        in_bias = []
        others = []
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                if abs(dx) + abs(dy) != r:
                    continue
                cx = cx0 + dx
                cy = cy0 + dy
                if not (0 <= cx < cfg.tiles_x and 0 <= cy < cfg.tiles_y):
                    continue
                if not is_known_open(cfg, state, cx, cy):
                    continue
                angle = math.atan2(dy, dx)
                if within_bias(angle):
                    in_bias.append((cx, cy))
                else:
                    others.append((cx, cy))
        if in_bias:
            return random.choice(in_bias)
        if others:
            return random.choice(others)
    return None


def main() -> None:
    pygame.init()
    cfg = load_config()
    screen = pygame.display.set_mode((cfg.width, cfg.height))
    pygame.display.set_caption("AI Test")

    current_seed = parse_seed()
    state = init_state(cfg, current_seed)
    animals_rng = random.Random(current_seed + 9999)

    def shortest_angle(from_angle: float, to_angle: float) -> float:
        return (to_angle - from_angle + math.pi) % (2 * math.pi) - math.pi

    def start_scan(reason: str, target_angle: float | None = None) -> None:
        state.scan_reason = reason
        state.scan_goal = "unknown"

        if target_angle is None:
            target_angle = best_unknown_angle(cfg, state)
            if target_angle is None:
                if state.last_blocked_cell is not None:
                    bx, by = cell_center(cfg, state.last_blocked_cell[0], state.last_blocked_cell[1])
                    target_angle = math.atan2(by - state.player_pos[1], bx - state.player_pos[0])
                    state.scan_goal = "blocked"
                elif state.last_move_angle is not None:
                    target_angle = state.last_move_angle + math.pi
                    state.scan_goal = "behind"
                else:
                    target_angle = state.player_angle + math.pi
                    state.scan_goal = "behind"

        state.scan_target_angle = target_angle
        delta = shortest_angle(state.player_angle, target_angle)
        half_fov = math.radians(cfg.fov_deg) / 2.0
        needed = max(0.0, abs(delta) - half_fov)
        state.scan_remaining = min(math.pi, needed)
        state.scan_dir = 1.0 if delta >= 0 else -1.0

    seen_cells, _ = cast_and_update(cfg, state)
    infer_enclosed_voids(cfg, state)
    observe_bushes_from_cells(cfg, state, seen_cells, state.game_time)
    observe_local_3x3(cfg, state, world_to_cell(cfg, state.player_pos[0], state.player_pos[1]), state.game_time)

    clock = pygame.time.Clock()
    running = True

    while running:
        dt = clock.tick(60) / 1000.0
        state.game_time += dt
        state.hunger = min(cfg.hunger_max, state.hunger + cfg.hunger_decay_per_sec * dt)
        update_animals(cfg, state, dt, animals_rng, player_pos=state.player_pos)
        state.sound_events.clear()
        if cfg.sound_enabled:
            for animal in state.animals:
                if not animal.alive:
                    continue
                emit_sound(state, animal.pos, cfg.animal_sound_level, animal.id, "animal")
        update_fruit_respawn(cfg, state, state.game_time)

        if state.respawn_estimate_sec is None:
            est_text = "unknown"
        else:
            est_text = f"{state.respawn_estimate_sec:.1f}s"
        pygame.display.set_caption(f"AI Test | respawn_est={est_text}")

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

                state = init_state(cfg, current_seed)
                animals_rng = random.Random(current_seed + 9999)

                seen_cells, _ = cast_and_update(cfg, state)
                infer_enclosed_voids(cfg, state)
                observe_bushes_from_cells(cfg, state, seen_cells, state.game_time)
                observe_local_3x3(
                    cfg, state, world_to_cell(cfg, state.player_pos[0], state.player_pos[1]), state.game_time
                )

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                cam_x = clamp(state.player_pos[0] - cfg.width / 2, 0, cfg.world_w - cfg.width)
                cam_y = clamp(state.player_pos[1] - cfg.height / 2, 0, cfg.world_h - cfg.height)
                world_x = cam_x + event.pos[0]
                world_y = cam_y + event.pos[1]
                cx, cy = world_to_cell(cfg, world_x, world_y)
                if 0 <= cx < cfg.tiles_x and 0 <= cy < cfg.tiles_y:
                    state.target_cell = (cx, cy)
                    state.search_state = "searching"
                    state.target_kind = None
                    state.hunt_target_id = None
                    state.hunt_retarget_timer = 0.0
                    state.hunt_search_timer = 0.0
                    state.hunt_last_search_center = None
                    state.stuck_frames = 0
                    state.scan_remaining = 0.0
                    state.scan_dir = 1.0
                    state.scan_reason = None
                    state.scan_goal = None
                    state.scan_target_angle = None
                    state.last_blocked_cell = None
                    state.last_target_cell = None
                    state.last_target_dist = None
                    state.frames_no_new = 0
                    state.frames_no_progress = 0
                    state.move_accum = 0.0
                    state.last_harvest_cell = None

        is_scanning = state.scan_remaining > 0.0
        heard, sound_angle, sound_vol, sound_event = process_hearing(cfg, state, state.player_pos)
        if heard and not is_scanning:
            state.sound_target_angle = sound_angle
            state.sound_turn_remaining = cfg.sound_turn_sec

        seen_cells, newly_seen = cast_and_update(cfg, state)
        infer_enclosed_voids(cfg, state)
        observe_bushes_from_cells(cfg, state, seen_cells, state.game_time)
        newly_seen += observe_local_3x3(
            cfg, state, world_to_cell(cfg, state.player_pos[0], state.player_pos[1]), state.game_time
        )
        if newly_seen > 0:
            state.frames_no_new = 0
        else:
            state.frames_no_new += 1

        visible_animals = set()
        for animal in state.animals:
            if not animal.alive:
                continue
            cx, cy = world_to_cell(cfg, animal.pos[0], animal.pos[1])
            mem = state.animal_memory[animal.id]
            if (cx, cy) in seen_cells:
                visible_animals.add(animal.id)
                prev_pos = mem.last_seen_pos
                prev_time = mem.last_seen_time
                if prev_pos is not None and state.game_time > prev_time:
                    dt_seen = state.game_time - prev_time
                    if dt_seen > 1e-6:
                        mem.vel_est = (
                            (animal.pos[0] - prev_pos[0]) / dt_seen,
                            (animal.pos[1] - prev_pos[1]) / dt_seen,
                        )
                mem.last_seen_pos = animal.pos
                mem.last_seen_cell = (cx, cy)
                mem.last_seen_time = state.game_time
                mem.confidence = 1.0
                mem.belief_pos = animal.pos
                mem.belief_cell = (cx, cy)
                mem.belief_radius = 0.0
                mem.belief_last_update = state.game_time
                if abs(mem.vel_est[0]) + abs(mem.vel_est[1]) > 1e-6:
                    mem.belief_heading = math.atan2(mem.vel_est[1], mem.vel_est[0])
            else:
                if mem.last_seen_time > -1e8:
                    age = state.game_time - mem.last_seen_time
                    mem.confidence = max(0.0, 1.0 - age / max(1e-6, cfg.hunt_lost_sec))
                else:
                    mem.confidence = 0.0
                if mem.belief_pos is None and mem.last_seen_pos is not None:
                    mem.belief_pos = mem.last_seen_pos
                    mem.belief_cell = mem.last_seen_cell
                    mem.belief_radius = 0.0
                    mem.belief_last_update = mem.last_seen_time
                if mem.belief_pos is not None and mem.belief_last_update > -1e8:
                    dt_pred = min(
                        max(0.0, state.game_time - mem.belief_last_update),
                        cfg.animal_predict_max_sec,
                    )
                    mem.belief_pos = (
                        mem.belief_pos[0] + mem.vel_est[0] * dt_pred,
                        mem.belief_pos[1] + mem.vel_est[1] * dt_pred,
                    )
                    mem.belief_radius += cfg.animal_belief_growth * dt_pred
                    mem.belief_last_update = state.game_time
                    pred_cell = world_to_cell(cfg, mem.belief_pos[0], mem.belief_pos[1])
                    max_proj = int(math.ceil(mem.belief_radius)) + 2
                    if not in_bounds_cell(cfg, pred_cell[0], pred_cell[1]):
                        if mem.belief_cell is None:
                            mem.belief_cell = mem.last_seen_cell
                    elif not is_known_open(cfg, state, pred_cell[0], pred_cell[1]):
                        proj = project_to_known_open(cfg, state, pred_cell, max_proj)
                        if proj is not None:
                            mem.belief_cell = proj
                        elif mem.belief_cell is None:
                            mem.belief_cell = mem.last_seen_cell
                    else:
                        mem.belief_cell = pred_cell

        if (
            sound_event is not None
            and sound_event.kind == "animal"
            and sound_event.source_id is not None
            and sound_event.source_id in state.animal_memory
            and sound_event.source_id not in visible_animals
        ):
            mem = state.animal_memory[sound_event.source_id]
            mem.belief_pos = sound_event.pos
            mem.belief_cell = world_to_cell(cfg, sound_event.pos[0], sound_event.pos[1])
            mem.belief_radius = max(mem.belief_radius, 1.0)
            mem.belief_heading = sound_angle if sound_angle is not None else mem.belief_heading
            mem.belief_last_update = state.game_time
            mem.confidence = max(mem.confidence, cfg.sound_confidence)

        current_cell = world_to_cell(cfg, state.player_pos[0], state.player_pos[1])
        state.ever_seen[current_cell[0]][current_cell[1]] = True
        state.subjective[current_cell[0]][current_cell[1]] = min(
            state.subjective[current_cell[0]][current_cell[1]], 0.0
        )

        dist_known, parent_known = bfs_known_open(cfg, state, current_cell)
        is_scanning = state.scan_remaining > 0.0
        half_fov = math.radians(cfg.fov_deg) / 2.0

        if is_scanning:
            if find_frontier_clusters(cfg, state, dist_known):
                state.scan_remaining = 0.0
                is_scanning = False
            elif state.scan_goal == "blocked" and state.last_blocked_cell is not None:
                bx, by = cell_center(cfg, state.last_blocked_cell[0], state.last_blocked_cell[1])
                ang_to_blocked = math.atan2(by - state.player_pos[1], bx - state.player_pos[0])
                if (
                    abs(shortest_angle(state.player_angle, ang_to_blocked)) <= half_fov
                    and is_confirmed_wall(cfg, state, state.last_blocked_cell[0], state.last_blocked_cell[1])
                ):
                    state.scan_remaining = 0.0
                    is_scanning = False

        if (not is_scanning) and state.search_state in ("reached", "unreachable") and state.scan_remaining <= 0.0:
            start_scan("reached_or_unreachable")
            state.search_state = "scanning"
            state.target_cell = None
            state.target_kind = None
            is_scanning = state.scan_remaining > 0.0

        if state.sound_turn_remaining > 0.0 and state.sound_target_angle is not None and not is_scanning:
            state.player_angle = rotate_toward(
                state.player_angle,
                state.sound_target_angle,
                cfg.rot_speed * cfg.sound_turn_speed_mult * dt,
            )
            state.sound_turn_remaining = max(0.0, state.sound_turn_remaining - dt)

        if is_scanning:
            step = cfg.rot_speed * dt
            state.player_angle = (state.player_angle + state.scan_dir * step) % (2 * math.pi)
            state.scan_remaining = max(0.0, state.scan_remaining - step)
            if state.scan_remaining <= 0.0:
                is_scanning = False

        if not is_scanning and state.search_state == "scanning":
            state.search_state = "idle"

        if state.target_kind == "hunt":
            state.hunt_retarget_timer += dt
        else:
            state.hunt_retarget_timer = 0.0

        if state.target_kind == "search":
            state.hunt_search_timer += dt
        else:
            state.hunt_search_timer = 0.0
            if state.target_kind != "search" and state.hunt_search_timer == 0.0:
                state.hunt_last_search_center = None

        if state.target_kind == "hunt" and state.hunt_target_id is not None:
            if state.hunt_target_id in visible_animals:
                if state.hunt_retarget_timer >= cfg.hunt_retarget_sec or state.target_cell is None:
                    mem = state.animal_memory[state.hunt_target_id]
                    animal = next(
                        (a for a in state.animals if a.id == state.hunt_target_id and a.alive),
                        None,
                    )
                    if animal is not None:
                        dist = math.hypot(
                            animal.pos[0] - state.player_pos[0],
                            animal.pos[1] - state.player_pos[1],
                        )
                        tau = clamp(dist / max(cfg.move_speed, 1.0), 0.2, 1.0)
                        pred_x = animal.pos[0] + mem.vel_est[0] * tau
                        pred_y = animal.pos[1] + mem.vel_est[1] * tau
                        pred_cell = world_to_cell(cfg, pred_x, pred_y)
                        cx, cy = world_to_cell(cfg, animal.pos[0], animal.pos[1])
                        target = None
                        if 0 <= pred_cell[0] < cfg.tiles_x and 0 <= pred_cell[1] < cfg.tiles_y:
                            if dist_known[pred_cell[0]][pred_cell[1]] != -1:
                                target = pred_cell
                        if target is None:
                            if dist_known[cx][cy] != -1:
                                target = (cx, cy)
                        if target is None and mem.last_seen_cell is not None:
                            if dist_known[mem.last_seen_cell[0]][mem.last_seen_cell[1]] != -1:
                                target = mem.last_seen_cell
                        if target is not None:
                            state.target_cell = target
                            state.search_state = "searching"
                            state.hunt_retarget_timer = 0.0
            else:
                mem = state.animal_memory[state.hunt_target_id]
                if mem.confidence > 0.0 and mem.belief_cell is not None:
                    state.target_kind = "track"
                    state.target_cell = mem.belief_cell
                    state.search_state = "searching"
                else:
                    state.target_cell = None
                    state.target_kind = None
                    state.hunt_target_id = None
                    state.search_state = "idle"

        if (
            state.target_kind == "track"
            and state.hunt_target_id is not None
            and state.target_cell is not None
            and current_cell == state.target_cell
            and state.hunt_target_id not in visible_animals
        ):
            mem = state.animal_memory[state.hunt_target_id]
            if mem.belief_cell is not None:
                state.target_kind = "search"
                state.hunt_search_timer = 0.0
                state.hunt_last_search_center = mem.belief_cell
                state.target_cell = None
                state.search_state = "idle"

        if state.target_kind == "search":
            if state.hunt_search_timer >= cfg.hunt_search_time:
                state.target_cell = None
                state.target_kind = None
                state.hunt_target_id = None
                state.hunt_last_search_center = None
                state.search_state = "idle"
            elif state.target_cell is None and state.hunt_last_search_center is not None:
                mem = state.animal_memory[state.hunt_target_id] if state.hunt_target_id is not None else None
                bias_heading = mem.belief_heading if mem is not None else 0.0
                start_radius = 1
                if mem is not None:
                    start_radius = max(1, int(math.ceil(mem.belief_radius)))
                search_target = pick_ring_search_cell(
                    cfg,
                    state,
                    state.hunt_last_search_center,
                    start_radius,
                    bias_heading,
                    cfg.animal_search_bias_deg,
                )
                if search_target is not None and dist_known[search_target[0]][search_target[1]] != -1:
                    state.target_cell = search_target
                    state.search_state = "searching"

        if state.target_cell != state.last_target_cell:
            state.last_target_cell = state.target_cell
            state.last_target_dist = None
            state.frames_no_progress = 0

        if state.target_cell is None:
            state.last_target_dist = None
            state.frames_no_progress = 0
        else:
            dist_to_target = dist_known[state.target_cell[0]][state.target_cell[1]]
            if dist_to_target == -1:
                state.frames_no_progress += 1
            else:
                if state.last_target_dist is None or dist_to_target < state.last_target_dist:
                    state.frames_no_progress = 0
                else:
                    state.frames_no_progress += 1
                state.last_target_dist = dist_to_target

        if (not is_scanning) and (
            state.frames_no_new >= cfg.no_new_frames or state.frames_no_progress >= cfg.no_progress_frames
        ):
            state.target_cell = None
            state.target_kind = None
            state.hunt_target_id = None
            state.hunt_last_search_center = None
            state.search_state = "unreachable"
            start_scan("stale")
            if state.scan_remaining > 0.0:
                state.search_state = "scanning"
            state.frames_no_new = 0
            state.frames_no_progress = 0

        if (not is_scanning) and (
            state.target_cell is None
            or state.search_state in ("idle", "reached", "unreachable")
            or len(visible_animals) > 0
        ):
            hunger_ratio = state.hunger / max(1e-6, cfg.hunger_max)
            satiety = 1.0 - hunger_ratio
            hunt_drive = max(0.25, hunger_ratio)
            known_bushes = [(bx, by) for (bx, by) in state.bush_positions if state.bush_known[bx][by]]
            home_cell = None
            if state.last_harvest_cell is not None:
                home_cell = state.last_harvest_cell
            elif known_bushes:
                avg_x = sum(b[0] for b in known_bushes) / len(known_bushes)
                avg_y = sum(b[1] for b in known_bushes) / len(known_bushes)
                home_cell = min(
                    known_bushes,
                    key=lambda b: (b[0] - avg_x) ** 2 + (b[1] - avg_y) ** 2
                )
            home_radius = cfg.home_radius + int(cfg.home_radius_extra * satiety)

            forage_cell, forage_score = pick_best_forage(
                cfg, state, dist_known, state.game_time, state.hunger, current_cell
            )
            no_forage = forage_cell is None
            frontier_cell = pick_exploration_target_from_dist(cfg, state, current_cell, dist_known)
            frontier_score = 0.0
            if frontier_cell is not None:
                frontier_score = cfg.frontier_weight / (1.0 + dist_known[frontier_cell[0]][frontier_cell[1]])
                if home_cell is not None:
                    if no_forage:
                        home_radius = cfg.home_radius + cfg.home_radius_extra
                    near_frontier = pick_frontier_near_home(cfg, state, dist_known, home_cell, home_radius)
                    if near_frontier is not None:
                        frontier_cell = near_frontier
                        frontier_score = cfg.frontier_weight / (1.0 + dist_known[frontier_cell[0]][frontier_cell[1]])
                    home_dist = abs(frontier_cell[0] - home_cell[0]) + abs(frontier_cell[1] - home_cell[1])
                    if home_dist > home_radius:
                        if no_forage:
                            far_mult = 1.0
                        else:
                            far_mult = cfg.home_far_mult + (1.0 - cfg.home_far_mult) * satiety
                        frontier_score *= far_mult
                frontier_score *= (1.0 + satiety * cfg.frontier_satiety_bonus)
                if known_bushes:
                    frontier_score *= cfg.explore_when_bushes_mult

            hunt_cell = None
            hunt_score = 0.0
            hunt_kind = None
            hunt_id = None

            for animal in state.animals:
                if not animal.alive:
                    continue
                if animal.id not in visible_animals:
                    continue
                mem = state.animal_memory[animal.id]
                dist = math.hypot(
                    animal.pos[0] - state.player_pos[0],
                    animal.pos[1] - state.player_pos[1],
                )
                tau = clamp(dist / max(cfg.move_speed, 1.0), 0.2, 1.0)
                pred_x = animal.pos[0] + mem.vel_est[0] * tau
                pred_y = animal.pos[1] + mem.vel_est[1] * tau
                pred_cell = world_to_cell(cfg, pred_x, pred_y)
                cx, cy = world_to_cell(cfg, animal.pos[0], animal.pos[1])
                target = None
                if 0 <= pred_cell[0] < cfg.tiles_x and 0 <= pred_cell[1] < cfg.tiles_y:
                    if dist_known[pred_cell[0]][pred_cell[1]] != -1:
                        target = pred_cell
                if target is None and dist_known[cx][cy] != -1:
                    target = (cx, cy)
                if target is None:
                    continue
                dist_steps = dist_known[target[0]][target[1]]
                if dist_steps == -1:
                    continue
                p_catch = clamp(1.0 - dist / (cfg.tile_size * 10.0), 0.0, 1.0)
                score = cfg.hunt_weight * hunt_drive * p_catch / (1.0 + dist_steps)
                if dist <= cfg.tile_size * 2.5:
                    score = max(score, cfg.hunt_weight * 0.5)
                else:
                    score = max(score, cfg.hunt_weight * 0.35)
                if score > hunt_score:
                    hunt_score = score
                    hunt_cell = target
                    hunt_kind = "hunt"
                    hunt_id = animal.id

            for animal in state.animals:
                if not animal.alive:
                    continue
                if animal.id in visible_animals:
                    continue
                mem = state.animal_memory[animal.id]
                if mem.confidence <= 0.0 or mem.belief_cell is None:
                    continue
                if dist_known[mem.belief_cell[0]][mem.belief_cell[1]] == -1:
                    continue
                dist_steps = dist_known[mem.belief_cell[0]][mem.belief_cell[1]]
                score = cfg.hunt_weight * hunt_drive * mem.confidence / (1.0 + dist_steps)
                score = score / (1.0 + max(0.0, mem.belief_radius))
                if score > hunt_score:
                    hunt_score = score
                    hunt_cell = mem.belief_cell
                    hunt_kind = "track"
                    hunt_id = animal.id

            if (
                state.hunt_target_id is not None
                and state.hunt_last_search_center is not None
                and state.hunt_search_timer < cfg.hunt_search_time
            ):
                mem = state.animal_memory[state.hunt_target_id]
                search_cell = pick_random_known_open_in_radius(
                    cfg, state, state.hunt_last_search_center, cfg.hunt_search_radius, animals_rng, current_cell
                )
                if search_cell is not None and dist_known[search_cell[0]][search_cell[1]] != -1:
                    dist_steps = dist_known[search_cell[0]][search_cell[1]]
                    score = cfg.hunt_weight * hunt_drive * mem.confidence / (1.0 + dist_steps)
                    if score > hunt_score:
                        hunt_score = score
                        hunt_cell = search_cell
                        hunt_kind = "search"
                        hunt_id = state.hunt_target_id

            best_alt = max(forage_score, frontier_score)
            force_hunt = False
            if state.target_kind in ("hunt", "track", "search") and hunt_score > 0.0:
                if hunt_score >= cfg.hunt_sticky_ratio * best_alt:
                    force_hunt = True

            if force_hunt and hunt_cell is not None:
                state.target_cell = hunt_cell
                state.target_kind = hunt_kind
                state.hunt_target_id = hunt_id
                state.search_state = "searching"
                state.stuck_frames = 0
            elif hunt_cell is not None and hunt_score > 0.0 and hunt_score >= max(forage_score, frontier_score):
                state.target_cell = hunt_cell
                state.target_kind = hunt_kind
                state.hunt_target_id = hunt_id
                state.search_state = "searching"
                state.stuck_frames = 0
            elif forage_cell is not None and (
                state.hunger >= cfg.forage_priority_hunger or forage_score > frontier_score
            ):
                state.target_cell = forage_cell
                state.target_kind = "forage"
                state.hunt_target_id = None
                state.search_state = "searching"
                state.stuck_frames = 0
            elif frontier_cell is not None:
                state.target_cell = frontier_cell
                state.target_kind = "frontier"
                state.hunt_target_id = None
                state.search_state = "searching"
                state.stuck_frames = 0
            elif home_cell is not None:
                loiter = pick_loiter_target(cfg, state, dist_known, home_cell, cfg.loiter_radius, current_cell)
                if loiter is not None:
                    state.target_cell = loiter
                    state.target_kind = "home"
                    state.hunt_target_id = None
                    state.search_state = "searching"
                    state.stuck_frames = 0
                else:
                    patrol = None
                    if not no_forage:
                        patrol = pick_nearest_known_bush(cfg, state, dist_known, current_cell, home_cell, home_radius)
                    if patrol is None:
                        patrol = pick_nearest_known_open(
                            cfg, state, dist_known, current_cell, home_cell, home_radius, avoid_bushes=True
                        )
                    if patrol is None:
                        patrol = pick_nearest_known_open(
                            cfg, state, dist_known, current_cell, None, None, avoid_bushes=True
                        )
                    if patrol is not None:
                        state.target_cell = patrol
                        state.target_kind = "home"
                        state.hunt_target_id = None
                        state.search_state = "searching"
                        state.stuck_frames = 0
                    else:
                        state.target_cell = None
                        state.target_kind = None
                        state.hunt_target_id = None
                        state.search_state = "idle"
            else:
                roam = pick_nearest_known_open(cfg, state, dist_known, current_cell, None, None, avoid_bushes=False)
                if roam is not None:
                    state.target_cell = roam
                    state.target_kind = "roam"
                    state.hunt_target_id = None
                    state.search_state = "searching"
                    state.stuck_frames = 0
                else:
                    state.target_cell = None
                    state.target_kind = None
                    state.hunt_target_id = None
                    state.search_state = "idle"

        path = []
        if not is_scanning:
            if state.target_cell is not None and state.search_state not in ("reached", "unreachable"):
                if is_confirmed_wall(cfg, state, state.target_cell[0], state.target_cell[1]) and state.ever_seen[
                    state.target_cell[0]
                ][state.target_cell[1]]:
                    state.search_state = "unreachable"
                elif current_cell == state.target_cell and not is_wall_cell(
                    cfg, state, state.target_cell[0], state.target_cell[1]
                ):
                    state.search_state = "reached"
                else:
                    if dist_known[state.target_cell[0]][state.target_cell[1]] != -1:
                        path = reconstruct_path(parent_known, current_cell, state.target_cell)
                    else:
                        exp = None
                        clusters = find_frontier_clusters(cfg, state, dist_known)
                        if clusters:
                            reps = [
                                pick_cluster_representative(c, dist_known, avoid=current_cell) for c in clusters
                            ]
                            reps = [
                                r for r in reps if r != current_cell and dist_known[r[0]][r[1]] != -1
                            ]
                            if reps:
                                exp = min(reps, key=lambda c: dist_known[c[0]][c[1]])

                        if exp is not None:
                            path = reconstruct_path(parent_known, current_cell, exp)
                        else:
                            state.search_state = "unreachable"

        if (not is_scanning) and path and len(path) > 1:
            next_cell = path[1]
            target_pos = cell_center(cfg, next_cell[0], next_cell[1])

            desired_angle = math.atan2(
                target_pos[1] - state.player_pos[1],
                target_pos[0] - state.player_pos[0]
            )
            state.last_move_angle = desired_angle
            state.player_angle = rotate_toward(state.player_angle, desired_angle, cfg.rot_speed * dt)

            state.move_accum += dt
            step_time = cfg.tile_size / max(1e-6, cfg.move_speed)
            attempted_move = False
            moved = False

            if state.move_accum >= step_time:
                state.move_accum -= step_time
                attempted_move = True
                if not is_wall_cell(cfg, state, next_cell[0], next_cell[1]):
                    state.player_pos = target_pos
                    moved = True

            if attempted_move:
                if moved:
                    state.stuck_frames = 0
                    state.last_blocked_cell = None
                else:
                    state.stuck_frames += 1
                    if state.stuck_frames >= cfg.stuck_frames:
                        state.subjective[next_cell[0]][next_cell[1]] = 1.0
                        state.ever_seen[next_cell[0]][next_cell[1]] = True
                        state.last_blocked_cell = next_cell
                        state.stuck_frames = 0
                        state.target_cell = None
                        state.target_kind = None
                        state.search_state = "unreachable"
                        start_scan("stuck")
                        if state.scan_remaining > 0.0:
                            state.search_state = "scanning"
        elif not is_scanning and state.sound_turn_remaining <= 0.0:
            state.player_angle = (state.player_angle + cfg.rot_speed * dt) % (2 * math.pi)

        current_cell_after = world_to_cell(cfg, state.player_pos[0], state.player_pos[1])
        if state.target_kind == "forage" and state.target_cell is not None and current_cell_after == state.target_cell:
            if state.bushes[current_cell_after[0]][current_cell_after[1]] and state.fruit[
                current_cell_after[0]
            ][current_cell_after[1]]:
                state.fruit[current_cell_after[0]][current_cell_after[1]] = False
                state.last_harvest_time[current_cell_after[0]][current_cell_after[1]] = state.game_time
                state.bush_known[current_cell_after[0]][current_cell_after[1]] = True
                state.bush_last_seen_time[current_cell_after[0]][current_cell_after[1]] = state.game_time
                state.bush_last_harvest_time[current_cell_after[0]][current_cell_after[1]] = state.game_time
                state.bush_last_seen_fruit[current_cell_after[0]][current_cell_after[1]] = False
                state.bush_last_empty_time[current_cell_after[0]][current_cell_after[1]] = state.game_time
                state.bush_last_checked_empty_time[current_cell_after[0]][current_cell_after[1]] = state.game_time
                state.hunger = max(0.0, state.hunger - cfg.hunger_eat_amount)
                state.last_harvest_cell = current_cell_after
            state.target_cell = None
            state.target_kind = None
            state.search_state = "idle"

        for animal in state.animals:
            if not animal.alive:
                continue
            dist = math.hypot(
                animal.pos[0] - state.player_pos[0],
                animal.pos[1] - state.player_pos[1],
            )
            if dist <= cfg.hunt_catch_dist * cfg.tile_size:
                mem = state.animal_memory[animal.id]
                mem.last_seen_pos = None
                mem.last_seen_cell = None
                mem.last_seen_time = -1e9
                mem.vel_est = (0.0, 0.0)
                mem.confidence = 0.0

                if state.hunt_target_id == animal.id:
                    state.hunt_target_id = None
                    state.target_cell = None
                    state.target_kind = None
                    state.search_state = "idle"

                animal.alive = False
                state.hunger = max(0.0, state.hunger - cfg.hunt_eat_amount)
                respawn_animal(
                    cfg, state, animal, animals_rng, avoid_cell=current_cell_after, avoid_radius=cfg.home_radius
                )

        cam_x = clamp(state.player_pos[0] - cfg.width / 2, 0, cfg.world_w - cfg.width)
        cam_y = clamp(state.player_pos[1] - cfg.height / 2, 0, cfg.world_h - cfg.height)

        screen.fill(cfg.black)
        draw_world(cfg, state, screen, cam_x, cam_y, seen_cells)
        draw_animals(cfg, state, screen, cam_x, cam_y, seen_cells)
        draw_target(cfg, screen, state.target_cell, cam_x, cam_y)
        draw_player(cfg, screen, state.player_pos, state.player_angle, cam_x, cam_y)
        draw_frontiers(cfg, state, screen, cam_x, cam_y, dist_known)

        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
