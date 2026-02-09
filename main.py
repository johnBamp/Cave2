from __future__ import annotations

import math
import random
import sys

import pygame

from config import load_config
from state import init_state
from grid import clamp, world_to_cell, cell_center, rotate_toward, is_wall_cell, is_confirmed_wall
from movement import bfs_known_open, reconstruct_path
from perception import cast_and_update, infer_enclosed_voids, observe_local_3x3
from exploration import (
    find_frontier_clusters,
    pick_cluster_representative,
    pick_frontier_near_home,
    pick_loiter_target,
    pick_nearest_known_bush,
    pick_nearest_known_open,
)
from foraging import observe_bushes_from_cells, update_fruit_respawn, pick_best_forage
from rendering import (
    draw_world,
    draw_player,
    draw_target,
    draw_frontiers,
    draw_subjective_debug,
    draw_animals,
    draw_animals_subjective,
    draw_wolves,
)
from sound import emit_sound, process_hearing
from animals import update_animals, respawn_animal
from wolves import update_wolves, wolf_attack, respawn_wolf
from comm import choose_intent_region, choose_call, update_call_mapping, apply_region_merge, region_from_cell


def parse_seed() -> int:
    if len(sys.argv) > 1:
        try:
            return int(sys.argv[1])
        except ValueError:
            return random.randrange(1 << 30)
    return random.randrange(1 << 30)


def score_unknown_cone(cfg, state, agent, angle: float, probe_rays: int = 9) -> float:
    half_fov = math.radians(cfg.fov_deg) / 2.0
    max_dist = cfg.vision_range
    steps = int(max_dist // cfg.ray_step)
    score = 0.0

    for i in range(probe_rays):
        t = i / (probe_rays - 1) if probe_rays > 1 else 0.5
        ang = angle - half_fov + (2 * half_fov) * t
        dx = math.cos(ang)
        dy = math.sin(ang)

        x, y = agent.pos
        for s in range(steps):
            x += dx * cfg.ray_step
            y += dy * cfg.ray_step
            cx, cy = world_to_cell(cfg, x, y)
            if not (0 <= cx < cfg.tiles_x and 0 <= cy < cfg.tiles_y):
                break
            if state.objective[cx][cy] == 1:
                break
            if not agent.ever_seen[cx][cy]:
                score += 1.0 / (1.0 + s)

    return score


def best_unknown_angle(cfg, state, agent, samples: int = 24):
    best_angle = None
    best_score = 0.0
    for i in range(samples):
        ang = (2 * math.pi) * (i / samples)
        score = score_unknown_cone(cfg, state, agent, ang)
        if score > best_score:
            best_score = score
            best_angle = ang
    if best_score <= 0.0:
        return None
    return best_angle


def pick_frontier_with_comm(cfg, agent, current_cell, dist, other_cell=None, separate_bias=False):
    clusters = find_frontier_clusters(cfg, agent, dist)
    if not clusters:
        return None

    best = None
    best_score = None
    for cluster in clusters:
        rep = pick_cluster_representative(cluster, dist, avoid=current_cell)
        if rep == current_cell:
            continue
        d = dist[rep[0]][rep[1]]
        if d == -1:
            continue
        base = 1.0 / (1.0 + d)
        size_bonus = 0.02 * len(cluster)
        if cfg.comm_enabled:
            region_id = region_from_cell(cfg, rep)
            known = agent.comm_region_known[region_id] if 0 <= region_id < len(agent.comm_region_known) else 0.0
            score = base * (1.0 - known) + size_bonus
        else:
            score = base + size_bonus
        # Bias to spread agents apart
        if other_cell is not None:
            d_other = abs(rep[0] - other_cell[0]) + abs(rep[1] - other_cell[1])
            sep_dist = max(1, cfg.separate_distance_tiles)
            sep_factor = min(1.0, d_other / sep_dist)
            if separate_bias:
                score = (sep_factor * cfg.separate_weight) + (0.15 * base) + size_bonus
            else:
                score *= (1.0 - cfg.separate_weight) + cfg.separate_weight * sep_factor
        # Directional exploration bias to break symmetry
        dx = rep[0] - current_cell[0]
        dy = rep[1] - current_cell[1]
        if dx != 0 or dy != 0:
            ang = math.atan2(dy, dx)
            diff = (ang - agent.explore_bias_angle + math.pi) % (2 * math.pi) - math.pi
            bias_factor = 0.5 * (1.0 + math.cos(diff))
            score += cfg.explore_bias_weight * bias_factor
        if best_score is None or score > best_score:
            best_score = score
            best = rep
    return best


def merge_subjectives(cfg, agent_a, agent_b):
    for gx in range(cfg.tiles_x):
        for gy in range(cfg.tiles_y):
            a_seen = agent_a.ever_seen[gx][gy]
            b_seen = agent_b.ever_seen[gx][gy]
            if not (a_seen or b_seen):
                continue
            if a_seen and b_seen:
                val = 0.5 * (agent_a.subjective[gx][gy] + agent_b.subjective[gx][gy])
            elif a_seen:
                val = agent_a.subjective[gx][gy]
            else:
                val = agent_b.subjective[gx][gy]
            agent_a.subjective[gx][gy] = val
            agent_b.subjective[gx][gy] = val
            agent_a.ever_seen[gx][gy] = True
            agent_b.ever_seen[gx][gy] = True
    agent_a.sync_text_timer = cfg.sync_text_sec
    agent_b.sync_text_timer = cfg.sync_text_sec


def find_shared_meet_cell(cfg, agent_a, agent_b, center_cell, radius: int):
    cx0, cy0 = center_cell
    best = None
    best_dist = None
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            cx = cx0 + dx
            cy = cy0 + dy
            if not (0 <= cx < cfg.tiles_x and 0 <= cy < cfg.tiles_y):
                continue
            if not agent_a.ever_seen[cx][cy] or not agent_b.ever_seen[cx][cy]:
                continue
            if agent_a.subjective[cx][cy] >= cfg.wall_conf_thresh:
                continue
            if agent_b.subjective[cx][cy] >= cfg.wall_conf_thresh:
                continue
            d = abs(dx) + abs(dy)
            if best_dist is None or d < best_dist:
                best_dist = d
                best = (cx, cy)
    return best


def main() -> None:
    pygame.init()
    cfg = load_config()
    screen = pygame.display.set_mode((cfg.width, cfg.height))
    pygame.display.set_caption("AI Test")
    font = pygame.font.SysFont(None, 18)

    current_seed = parse_seed()
    state = init_state(cfg, current_seed)
    rng = random.Random(current_seed + 9999)

    clock = pygame.time.Clock()
    running = True

    def shortest_angle(from_angle: float, to_angle: float) -> float:
        return (to_angle - from_angle + math.pi) % (2 * math.pi) - math.pi

    def start_scan(agent, reason: str, target_angle: float | None = None) -> None:
        agent.scan_reason = reason
        agent.scan_goal = "unknown"

        if target_angle is None:
            target_angle = best_unknown_angle(cfg, state, agent)
            if target_angle is None:
                if agent.last_move_angle is not None:
                    target_angle = agent.last_move_angle + math.pi
                    agent.scan_goal = "behind"
                else:
                    target_angle = agent.angle + math.pi
                    agent.scan_goal = "behind"

        agent.scan_target_angle = target_angle
        delta = shortest_angle(agent.angle, target_angle)
        half_fov = math.radians(cfg.fov_deg) / 2.0
        needed = max(0.0, abs(delta) - half_fov)
        agent.scan_remaining = min(math.pi, needed)
        agent.scan_dir = 1.0 if delta >= 0 else -1.0

    # Seed initial perception
    for agent in state.agents:
        seen_cells, _ = cast_and_update(cfg, state, agent)
        infer_enclosed_voids(cfg, agent)
        observe_local_3x3(cfg, state, agent, world_to_cell(cfg, agent.pos[0], agent.pos[1]), state.game_time)

    while running:
        dt = clock.tick(60) / 1000.0
        state.game_time += dt
        if cfg.enable_animals:
            update_animals(cfg, state, dt, rng, player_pos=state.agents[0].pos if state.agents else None)
        if cfg.enable_wolves:
            update_wolves(cfg, state, dt, rng, agents=state.agents)
        if cfg.enable_animals:
            for animal in state.animals:
                if not animal.alive and animal.respawn_timer > 0.0:
                    animal.respawn_timer = max(0.0, animal.respawn_timer - dt)
                    if animal.respawn_timer <= 0.0:
                        respawn_animal(cfg, state, animal, rng)
        if cfg.enable_wolves:
            for wolf in state.wolves:
                if not wolf.alive and wolf.respawn_timer > 0.0:
                    wolf.respawn_timer = max(0.0, wolf.respawn_timer - dt)
                    if wolf.respawn_timer <= 0.0:
                        respawn_wolf(cfg, state, wolf, rng)
        update_fruit_respawn(cfg, state, state.game_time)

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
                rng = random.Random(current_seed + 9999)

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if state.agents:
                    agent = state.agents[0]
                    cam_x = clamp(agent.pos[0] - cfg.width / 2, 0, cfg.world_w - cfg.width)
                    cam_y = clamp(agent.pos[1] - cfg.height / 2, 0, cfg.world_h - cfg.height)
                    world_x = cam_x + event.pos[0]
                    world_y = cam_y + event.pos[1]
                    cx, cy = world_to_cell(cfg, world_x, world_y)
                    if 0 <= cx < cfg.tiles_x and 0 <= cy < cfg.tiles_y:
                        agent.target_cell = (cx, cy)
                        agent.search_state = "searching"
                        agent.stuck_frames = 0
                        agent.scan_remaining = 0.0
                        agent.scan_dir = 1.0
                        agent.scan_reason = None
                        agent.scan_goal = None
                        agent.scan_target_angle = None
                        agent.last_blocked_cell = None
                        agent.last_target_cell = None
                        agent.last_target_dist = None
                        agent.frames_no_new = 0
                        agent.frames_no_progress = 0
                        agent.move_accum = 0.0

        # Communication emission + creature sounds
        state.sound_events.clear()
        if cfg.sound_enabled and cfg.comm_enabled:
            for agent in state.agents:
                agent.comm_emit_timer += dt
                if agent.comm_emit_timer >= cfg.comm_period_sec:
                    region_id = choose_intent_region(cfg, agent)
                    call_id = choose_call(cfg, agent, region_id, rng)
                    emit_sound(
                        state,
                        agent.pos,
                        cfg.comm_sound_level,
                        source_id=agent.agent_id,
                        kind="utterance",
                        payload=call_id,
                    )
                    agent.comm_emit_timer -= cfg.comm_period_sec
        if cfg.sound_enabled and cfg.enable_animals:
            for animal in state.animals:
                if not animal.alive:
                    continue
                emit_sound(state, animal.pos, cfg.animal_sound_level, animal.id, "animal")
        if cfg.sound_enabled and cfg.enable_wolves:
            for wolf in state.wolves:
                if not wolf.alive:
                    continue
                emit_sound(state, wolf.pos, cfg.wolf_sound_level, wolf.id, "wolf")

        # Hearing / learning
        if cfg.sound_enabled and cfg.comm_enabled:
            for agent in state.agents:
                heard, _, _, sound_event = process_hearing(
                    cfg, state, agent.pos, ignore_source_id=agent.agent_id
                )
                if heard and sound_event is not None and sound_event.kind == "utterance":
                    if sound_event.payload is not None:
                        sender_cell = world_to_cell(cfg, sound_event.pos[0], sound_event.pos[1])
                        region_id = region_from_cell(cfg, sender_cell)
                        update_call_mapping(cfg, agent, sound_event.payload, region_id)
                        apply_region_merge(cfg, agent, region_id)
                        agent.comm_last_heard = (sound_event.payload, region_id, state.game_time)

        # Per-agent update
        agent_seen_cells: list[set[tuple[int, int]]] = []
        regroup_distance = cfg.regroup_distance_tiles * cfg.tile_size
        sync_distance = cfg.sync_distance_tiles * cfg.tile_size
        separate_distance = cfg.separate_distance_tiles * cfg.tile_size
        agent_distance = None
        downed_agents = [agent for agent in state.agents if agent.downed]
        threat_wolf = None
        threat_dist = None
        if cfg.enable_wolves:
            for wolf in state.wolves:
                if not wolf.alive:
                    continue
                for agent in state.agents:
                    if agent.downed:
                        continue
                    dist = math.hypot(wolf.pos[0] - agent.pos[0], wolf.pos[1] - agent.pos[1])
                    if dist <= cfg.wolf_sight_range_tiles * cfg.tile_size:
                        if threat_dist is None or dist < threat_dist:
                            threat_dist = dist
                            threat_wolf = wolf
        if len(state.agents) > 1:
            agent_distance = math.hypot(
                state.agents[0].pos[0] - state.agents[1].pos[0],
                state.agents[0].pos[1] - state.agents[1].pos[1],
            )
        meet_active = state.meet_active_until > state.game_time
        if state.game_time >= state.meet_next_time:
            state.meet_active_until = state.game_time + cfg.meet_window_sec
            state.meet_next_time = state.game_time + cfg.meet_interval_sec
            if len(state.agents) > 1:
                a_cell = world_to_cell(cfg, state.agents[0].pos[0], state.agents[0].pos[1])
                b_cell = world_to_cell(cfg, state.agents[1].pos[0], state.agents[1].pos[1])
                center = ((a_cell[0] + b_cell[0]) // 2, (a_cell[1] + b_cell[1]) // 2)
                shared = find_shared_meet_cell(cfg, state.agents[0], state.agents[1], center, 6)
                state.meet_cell = shared if shared is not None else a_cell
            else:
                state.meet_cell = None
            meet_active = True
        for agent in state.agents:
            if agent.downed:
                agent.sync_text_timer = max(0.0, agent.sync_text_timer - dt)
                agent_seen_cells.append(set())
                continue
            agent.hunger = min(cfg.hunger_max, agent.hunger + cfg.hunger_decay_per_sec * dt)
            agent.attack_cooldown = max(0.0, agent.attack_cooldown - dt)
            seen_cells, newly_seen = cast_and_update(cfg, state, agent)
            agent_seen_cells.append(seen_cells)
            if cfg.enable_animals:
                for animal in state.animals:
                    if not animal.alive:
                        continue
                    acx, acy = world_to_cell(cfg, animal.pos[0], animal.pos[1])
                    if (acx, acy) in seen_cells:
                        agent.animal_last_seen[animal.id] = animal.pos
            observe_bushes_from_cells(cfg, state, seen_cells, state.game_time)
            infer_enclosed_voids(cfg, agent)
            newly_seen += observe_local_3x3(
                cfg, state, agent, world_to_cell(cfg, agent.pos[0], agent.pos[1]), state.game_time
            )
            if newly_seen > 0:
                agent.frames_no_new = 0
            else:
                agent.frames_no_new += 1
            agent.sync_text_timer = max(0.0, agent.sync_text_timer - dt)

            current_cell = world_to_cell(cfg, agent.pos[0], agent.pos[1])
            agent.ever_seen[current_cell[0]][current_cell[1]] = True
            agent.subjective[current_cell[0]][current_cell[1]] = min(
                agent.subjective[current_cell[0]][current_cell[1]], 0.0
            )

            dist_known, parent_known = bfs_known_open(cfg, agent, current_cell)
            is_scanning = agent.scan_remaining > 0.0

            if is_scanning:
                if find_frontier_clusters(cfg, agent, dist_known):
                    agent.scan_remaining = 0.0
                    is_scanning = False

            if (not is_scanning) and agent.search_state in ("reached", "unreachable") and agent.scan_remaining <= 0.0:
                start_scan(agent, "reached_or_unreachable")
                agent.search_state = "scanning"
                agent.target_cell = None
                is_scanning = agent.scan_remaining > 0.0

            if is_scanning:
                step = cfg.rot_speed * dt
                agent.angle = (agent.angle + agent.scan_dir * step) % (2 * math.pi)
                agent.scan_remaining = max(0.0, agent.scan_remaining - step)
                if agent.scan_remaining <= 0.0:
                    is_scanning = False

            if not is_scanning and agent.search_state == "scanning":
                agent.search_state = "idle"

            if agent.target_cell != agent.last_target_cell:
                agent.last_target_cell = agent.target_cell
                agent.last_target_dist = None
                agent.frames_no_progress = 0

            if agent.target_cell is None:
                agent.last_target_dist = None
                agent.frames_no_progress = 0
            else:
                dist_to_target = dist_known[agent.target_cell[0]][agent.target_cell[1]]
                if dist_to_target == -1:
                    agent.frames_no_progress += 1
                else:
                    if agent.last_target_dist is None or dist_to_target < agent.last_target_dist:
                        agent.frames_no_progress = 0
                    else:
                        agent.frames_no_progress += 1
                    agent.last_target_dist = dist_to_target

            if (not is_scanning) and (
                agent.frames_no_new >= cfg.no_new_frames or agent.frames_no_progress >= cfg.no_progress_frames
            ):
                agent.target_cell = None
                agent.search_state = "unreachable"
                start_scan(agent, "stale")
                if agent.scan_remaining > 0.0:
                    agent.search_state = "scanning"
                agent.frames_no_new = 0
                agent.frames_no_progress = 0

            # Tag-team wolves if any are close to either agent
            force_target = False
            if downed_agents:
                nearest_down = None
                nearest_dist = None
                for downed in downed_agents:
                    dist = math.hypot(downed.pos[0] - agent.pos[0], downed.pos[1] - agent.pos[1])
                    if nearest_dist is None or dist < nearest_dist:
                        nearest_dist = dist
                        nearest_down = downed
                if nearest_down is not None:
                    agent.target_cell = world_to_cell(cfg, nearest_down.pos[0], nearest_down.pos[1])
                    agent.target_kind = "revive"
                    agent.search_state = "searching"
                    agent.stuck_frames = 0
                    force_target = True
            if (not force_target) and threat_wolf is not None and not is_scanning:
                agent.target_cell = world_to_cell(cfg, threat_wolf.pos[0], threat_wolf.pos[1])
                agent.target_kind = "fight"
                agent.search_state = "searching"
                agent.stuck_frames = 0
                force_target = True

            if (not force_target) and meet_active and state.meet_cell is not None and not is_scanning:
                agent.target_cell = state.meet_cell
                agent.target_kind = "meet"
                agent.search_state = "searching"
                agent.stuck_frames = 0
            elif (not force_target) and cfg.comm_enabled and agent_distance is not None and agent_distance > regroup_distance and not is_scanning:
                other = state.agents[1 - agent.agent_id]
                agent.target_cell = world_to_cell(cfg, other.pos[0], other.pos[1])
                agent.target_kind = "regroup"
                agent.search_state = "searching"
                agent.stuck_frames = 0
            elif (not force_target) and (not is_scanning) and (
                agent.target_cell is None or agent.search_state in ("idle", "reached", "unreachable")
            ):
                other_cell = None
                separate_bias = False
                if len(state.agents) > 1:
                    other = state.agents[1 - agent.agent_id]
                    other_cell = world_to_cell(cfg, other.pos[0], other.pos[1])
                    if agent_distance is not None and agent_distance < separate_distance and not meet_active:
                        separate_bias = True
                forage_cell, forage_score = pick_best_forage(
                    cfg, state, dist_known, state.game_time, agent.hunger, current_cell
                )
                home_cell = agent.last_harvest_cell
                if home_cell is None:
                    home_cell = pick_nearest_known_bush(cfg, state, agent, dist_known, current_cell)
                frontier_cell = None
                frontier_score = 0.0
                if home_cell is not None:
                    frontier_cell = pick_frontier_near_home(cfg, agent, dist_known, home_cell, cfg.home_radius)
                if frontier_cell is None:
                    frontier_cell = pick_frontier_with_comm(
                        cfg,
                        agent,
                        current_cell,
                        dist_known,
                        other_cell=other_cell,
                        separate_bias=separate_bias,
                    )
                if frontier_cell is not None:
                    frontier_score = 1.0 / (1.0 + dist_known[frontier_cell[0]][frontier_cell[1]])
                force_forage = False
                if forage_cell is not None:
                    force_forage = state.bush_last_seen_fruit[forage_cell[0]][forage_cell[1]]
                if forage_cell is not None and (
                    force_forage or agent.hunger >= cfg.forage_priority_hunger or forage_score > frontier_score
                ):
                    agent.target_cell = forage_cell
                    agent.target_kind = "forage"
                    agent.search_state = "searching"
                    agent.stuck_frames = 0
                elif frontier_cell is not None:
                    agent.target_cell = frontier_cell
                    agent.target_kind = "frontier"
                    agent.search_state = "searching"
                    agent.stuck_frames = 0
                elif home_cell is not None:
                    loiter = pick_loiter_target(cfg, agent, dist_known, home_cell, cfg.loiter_radius, current_cell)
                    if loiter is None:
                        loiter = pick_nearest_known_open(cfg, state, agent, dist_known, current_cell)
                    if loiter is not None:
                        agent.target_cell = loiter
                        agent.target_kind = "home"
                        agent.search_state = "searching"
                        agent.stuck_frames = 0
                    else:
                        agent.target_cell = None
                        agent.target_kind = None
                        agent.search_state = "idle"
                else:
                    agent.target_cell = None
                    agent.target_kind = None
                    agent.search_state = "idle"

            path = []
            if not is_scanning:
                if agent.target_cell is not None and agent.search_state not in ("reached", "unreachable"):
                    if is_confirmed_wall(cfg, agent, agent.target_cell[0], agent.target_cell[1]):
                        agent.search_state = "unreachable"
                    elif current_cell == agent.target_cell and not is_wall_cell(
                        cfg, state, agent.target_cell[0], agent.target_cell[1]
                    ):
                        agent.search_state = "reached"
                    else:
                        if dist_known[agent.target_cell[0]][agent.target_cell[1]] != -1:
                            path = reconstruct_path(parent_known, current_cell, agent.target_cell)
                        else:
                            agent.search_state = "unreachable"

            if (not is_scanning) and path and len(path) > 1:
                next_cell = path[1]
                target_pos = cell_center(cfg, next_cell[0], next_cell[1])

                desired_angle = math.atan2(
                    target_pos[1] - agent.pos[1],
                    target_pos[0] - agent.pos[0],
                )
                agent.last_move_angle = desired_angle
                agent.angle = rotate_toward(agent.angle, desired_angle, cfg.rot_speed * dt)

                agent.move_accum += dt
                step_time = cfg.tile_size / max(1e-6, cfg.move_speed)
                attempted_move = False
                moved = False

                if agent.move_accum >= step_time:
                    agent.move_accum -= step_time
                    attempted_move = True
                    if not is_wall_cell(cfg, state, next_cell[0], next_cell[1]):
                        agent.pos = target_pos
                        moved = True

                if attempted_move:
                    if moved:
                        agent.stuck_frames = 0
                        agent.last_blocked_cell = None
                    else:
                        agent.stuck_frames += 1
                        if agent.stuck_frames >= cfg.stuck_frames:
                            agent.subjective[next_cell[0]][next_cell[1]] = 1.0
                            agent.ever_seen[next_cell[0]][next_cell[1]] = True
                            agent.last_blocked_cell = next_cell
                            agent.stuck_frames = 0
                            agent.target_cell = None
                            agent.search_state = "unreachable"
                            start_scan(agent, "stuck")
                            if agent.scan_remaining > 0.0:
                                agent.search_state = "scanning"
            elif not is_scanning:
                agent.angle = (agent.angle + cfg.rot_speed * dt) % (2 * math.pi)

            # Harvest
            if agent.target_kind == "forage":
                cur = world_to_cell(cfg, agent.pos[0], agent.pos[1])
                if cur == agent.target_cell and state.bushes[cur[0]][cur[1]] and state.fruit[cur[0]][cur[1]]:
                    state.fruit[cur[0]][cur[1]] = False
                    state.last_harvest_time[cur[0]][cur[1]] = state.game_time
                    state.bush_last_seen_time[cur[0]][cur[1]] = state.game_time
                    state.bush_last_seen_fruit[cur[0]][cur[1]] = False
                    state.bush_last_harvest_time[cur[0]][cur[1]] = state.game_time
                    state.bush_last_empty_time[cur[0]][cur[1]] = state.game_time
                    state.bush_last_checked_empty_time[cur[0]][cur[1]] = state.game_time
                    agent.hunger = max(0.0, agent.hunger - cfg.hunger_eat_amount)
                    agent.last_harvest_cell = cur
                    agent.target_cell = None
                    agent.target_kind = None
                    agent.search_state = "idle"
            agent.respawn_estimate_sec = state.respawn_estimate_sec

        if len(state.agents) > 1:
            dist_sync = math.hypot(
                state.agents[0].pos[0] - state.agents[1].pos[0],
                state.agents[0].pos[1] - state.agents[1].pos[1],
            )
            if dist_sync <= sync_distance:
                merge_subjectives(cfg, state.agents[0], state.agents[1])
                state.meet_active_until = state.game_time
                state.meet_cell = None

        # Revive downed agents
        downed_agents = [agent for agent in state.agents if agent.downed]
        if downed_agents:
            for agent in state.agents:
                if agent.downed:
                    continue
                cur = world_to_cell(cfg, agent.pos[0], agent.pos[1])
                for downed in downed_agents:
                    if world_to_cell(cfg, downed.pos[0], downed.pos[1]) == cur:
                        downed.downed = False
                        downed.hp = cfg.player_max_hp * 0.5
                        downed.attack_cooldown = 0.0
                        downed.sync_text_timer = cfg.sync_text_sec
                        agent.sync_text_timer = cfg.sync_text_sec

        # Combat
        if cfg.enable_wolves:
            # Agent attacks
            for agent in state.agents:
                if agent.downed:
                    continue
                if agent.attack_cooldown > 0.0:
                    continue
                nearest_wolf = None
                nearest_dist = None
                for wolf in state.wolves:
                    if not wolf.alive:
                        continue
                    dist = math.hypot(wolf.pos[0] - agent.pos[0], wolf.pos[1] - agent.pos[1])
                    if dist <= cfg.player_attack_range_tiles * cfg.tile_size:
                        if nearest_dist is None or dist < nearest_dist:
                            nearest_dist = dist
                            nearest_wolf = wolf
                if nearest_wolf is not None:
                    damage = cfg.player_dps * cfg.player_attack_cooldown
                    nearest_wolf.hp = max(0.0, nearest_wolf.hp - damage)
                    agent.attack_cooldown = cfg.player_attack_cooldown
                    if nearest_wolf.hp <= 0.0:
                        nearest_wolf.alive = False
                        nearest_wolf.respawn_timer = 0.0

            # Wolf attacks
            for wolf in state.wolves:
                if not wolf.alive:
                    continue
                target_agent = None
                best_dist = None
                for agent in state.agents:
                    if agent.downed:
                        continue
                    dist = math.hypot(wolf.pos[0] - agent.pos[0], wolf.pos[1] - agent.pos[1])
                    if best_dist is None or dist < best_dist:
                        best_dist = dist
                        target_agent = agent
                if target_agent is None:
                    continue
                wolf_attack(cfg, wolf, target_agent)
                if target_agent.hp <= 0.0:
                    target_agent.downed = True
                    target_agent.target_cell = None
                    target_agent.target_kind = "downed"
                    target_agent.search_state = "idle"

        # Render
        if not state.agents:
            screen.fill(cfg.black)
            pygame.display.flip()
            continue

        focus = state.agents[0]
        cam_x = clamp(focus.pos[0] - cfg.width / 2, 0, cfg.world_w - cfg.width)
        cam_y = clamp(focus.pos[1] - cfg.height / 2, 0, cfg.world_h - cfg.height)

        screen.fill(cfg.black)
        # Render using agent 0's subjective view
        focus_seen = agent_seen_cells[0] if agent_seen_cells else set()
        draw_world(cfg, state, focus, screen, cam_x, cam_y, focus_seen)
        agent_b = state.agents[1] if len(state.agents) > 1 else None
        draw_subjective_debug(cfg, focus, agent_b, screen, cam_x, cam_y)

        for idx, agent in enumerate(state.agents):
            color = cfg.agent_colors[idx % len(cfg.agent_colors)]
            draw_target(cfg, screen, agent.target_cell, cam_x, cam_y)
            draw_player(cfg, screen, agent.pos, agent.angle, cam_x, cam_y, color=color)
            if agent.sync_text_timer > 0.0:
                text = font.render("SYNC", True, color)
                tx = int(agent.pos[0] - cam_x)
                ty = int(agent.pos[1] - cam_y - cfg.tile_size)
                screen.blit(text, (tx - text.get_width() // 2, ty))
            if agent.downed:
                px = int(agent.pos[0] - cam_x)
                py = int(agent.pos[1] - cam_y)
                size = max(6, cfg.tile_size)
                pygame.draw.line(screen, (220, 40, 40), (px - size, py - size), (px + size, py + size), 3)
                pygame.draw.line(screen, (220, 40, 40), (px - size, py + size), (px + size, py - size), 3)

        if cfg.enable_animals:
            # Subjective animal rendering per agent (red/blue)
            colors = [(220, 60, 60), (60, 100, 255)]
            for idx, agent in enumerate(state.agents):
                draw_animals_subjective(cfg, agent, screen, cam_x, cam_y, colors[idx % len(colors)])
        if cfg.enable_wolves:
            draw_wolves(cfg, state, screen, cam_x, cam_y)

        # Respawn estimate display
        if state.agents:
            a_est = state.agents[0].respawn_estimate_sec
            b_est = state.agents[1].respawn_estimate_sec if len(state.agents) > 1 else None
            a_text = "A: ?"
            b_text = "B: ?"
            if a_est is not None:
                a_text = f"A: {a_est:.1f}s"
            if b_est is not None:
                b_text = f"B: {b_est:.1f}s"
            est_surface = font.render(f"Respawn {a_text} | {b_text}", True, (220, 220, 220))
            screen.blit(est_surface, (8, 6))

        dist_dbg, _ = bfs_known_open(cfg, focus, world_to_cell(cfg, focus.pos[0], focus.pos[1]))
        draw_frontiers(cfg, focus, screen, cam_x, cam_y, dist_dbg)

        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
