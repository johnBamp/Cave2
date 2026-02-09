from __future__ import annotations

import math
import random
import sys

import pygame

from config import load_config
from state import init_state
from grid import clamp, world_to_cell, cell_center, rotate_toward, is_confirmed_wall, is_wall_cell
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
from rendering import draw_world, draw_player, draw_target, draw_frontiers


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


def main() -> None:
    pygame.init()
    cfg = load_config()
    screen = pygame.display.set_mode((cfg.width, cfg.height))
    pygame.display.set_caption("AI Test")

    current_seed = parse_seed()
    state = init_state(cfg, current_seed)

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

        if is_scanning:
            step = cfg.rot_speed * dt
            state.player_angle = (state.player_angle + state.scan_dir * step) % (2 * math.pi)
            state.scan_remaining = max(0.0, state.scan_remaining - step)
            if state.scan_remaining <= 0.0:
                is_scanning = False

        if not is_scanning and state.search_state == "scanning":
            state.search_state = "idle"

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
            state.search_state = "unreachable"
            start_scan("stale")
            if state.scan_remaining > 0.0:
                state.search_state = "scanning"
            state.frames_no_new = 0
            state.frames_no_progress = 0

        if (not is_scanning) and (
            state.target_cell is None or state.search_state in ("idle", "reached", "unreachable")
        ):
            hunger_ratio = state.hunger / max(1e-6, cfg.hunger_max)
            satiety = 1.0 - hunger_ratio
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

            forage_cell, forage_score = pick_best_forage(cfg, state, dist_known, state.game_time, state.hunger, current_cell)
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

            if forage_cell is not None and (
                state.hunger >= cfg.forage_priority_hunger or forage_score > frontier_score
            ):
                state.target_cell = forage_cell
                state.target_kind = "forage"
                state.search_state = "searching"
                state.stuck_frames = 0
            elif frontier_cell is not None:
                state.target_cell = frontier_cell
                state.target_kind = "frontier"
                state.search_state = "searching"
                state.stuck_frames = 0
            elif home_cell is not None:
                loiter = pick_loiter_target(cfg, state, dist_known, home_cell, cfg.loiter_radius, current_cell)
                if loiter is not None:
                    state.target_cell = loiter
                    state.target_kind = "home"
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
                        state.search_state = "searching"
                        state.stuck_frames = 0
                    else:
                        state.target_cell = None
                        state.target_kind = None
                        state.search_state = "idle"
            else:
                state.target_cell = None
                state.target_kind = None
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
        elif not is_scanning:
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

        cam_x = clamp(state.player_pos[0] - cfg.width / 2, 0, cfg.world_w - cfg.width)
        cam_y = clamp(state.player_pos[1] - cfg.height / 2, 0, cfg.world_h - cfg.height)

        screen.fill(cfg.black)
        draw_world(cfg, state, screen, cam_x, cam_y, seen_cells)
        draw_target(cfg, screen, state.target_cell, cam_x, cam_y)
        draw_player(cfg, screen, state.player_pos, state.player_angle, cam_x, cam_y)
        draw_frontiers(cfg, state, screen, cam_x, cam_y, dist_known)

        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
