import math
import os
import random
import sys
from collections import deque
from typing import List, Tuple

import pygame
import torch

import settings
from bushes import Bush
from ppo_agent import PPOAgent, PolicyAction, RolloutStorage

# Basic 2D cave demo with manual player control.
# - Move: W/A/S/D
# - Turn: Q/E
# - A full FOV fan of rays is drawn in the world and summarized on the sidebar.

Vec2 = pygame.math.Vector2


class CaveMap:
    def __init__(self, cols: int, rows: int, cell_size: int, fill_prob: float = settings.FILL_PROB):
        self.cols = cols
        self.rows = rows
        self.cell_size = cell_size
        self.grid = self._generate(fill_prob)

    def _generate(self, fill_prob: float) -> List[List[int]]:
        grid = [[1 if random.random() < fill_prob else 0 for _ in range(self.cols)] for _ in range(self.rows)]

        def neighbors(x: int, y: int) -> int:
            count = 0
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if nx < 0 or nx >= self.cols or ny < 0 or ny >= self.rows:
                        count += 1  # Treat out of bounds as wall
                    elif grid[ny][nx] == 1:
                        count += 1
            return count

        for _ in range(settings.SMOOTH_PASSES):
            new_grid = [[0] * self.cols for _ in range(self.rows)]
            for y in range(self.rows):
                for x in range(self.cols):
                    wall_count = neighbors(x, y)
                    if grid[y][x] == 1:
                        new_grid[y][x] = 1 if wall_count >= 4 else 0
                    else:
                        new_grid[y][x] = 1 if wall_count >= 5 else 0
            grid = new_grid
        return grid

    def is_wall(self, world_pos: Vec2) -> bool:
        x_idx = int(world_pos.x // self.cell_size)
        y_idx = int(world_pos.y // self.cell_size)
        if x_idx < 0 or x_idx >= self.cols or y_idx < 0 or y_idx >= self.rows:
            return True
        return self.grid[y_idx][x_idx] == 1

    def draw(self, surface: pygame.Surface):
        for y in range(self.rows):
            for x in range(self.cols):
                if self.grid[y][x] == 1:
                    rect = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
                    surface.fill(settings.WALL_COLOR, rect)


class Player:
    def __init__(self, pos: Tuple[float, float], angle_deg: float = 0.0):
        self.pos = Vec2(pos)
        self.angle = angle_deg  # degrees
        self.radius = settings.PLAYER_RADIUS
        self.move_speed = settings.PLAYER_MOVE_SPEED  # pixels per second
        self.turn_speed = settings.PLAYER_TURN_SPEED  # degrees per second
        self.energy = settings.ENERGY_MAX

    def update(self, dt: float, cave: CaveMap, keys: pygame.key.ScancodeWrapper) -> float:
        is_agent = hasattr(keys, "pressed")
        move_dir = Vec2(0, 0)
        if keys[pygame.K_w]:
            move_dir += Vec2(1, 0).rotate(self.angle)
        if keys[pygame.K_s]:
            move_dir += Vec2(-1, 0).rotate(self.angle)
        if keys[pygame.K_a]:
            move_dir += Vec2(1, 0).rotate(self.angle - 90)
        if keys[pygame.K_d]:
            move_dir += Vec2(1, 0).rotate(self.angle + 90)

        moved = Vec2(0, 0)
        if move_dir.length_squared() > 0:
            if is_agent:
                step = move_dir.normalize() * settings.AGENT_STEP_PX
                moved = self._try_move(step, cave)
            else:
                move_dir = move_dir.normalize() * self.move_speed * dt
                moved = self._try_move(move_dir, cave)

        if keys[pygame.K_q]:
            if is_agent:
                self.angle = (self.angle - settings.AGENT_TURN_DEG) % 360
            else:
                self.angle = (self.angle - self.turn_speed * dt) % 360
        if keys[pygame.K_e]:
            if is_agent:
                self.angle = (self.angle + settings.AGENT_TURN_DEG) % 360
            else:
                self.angle = (self.angle + self.turn_speed * dt) % 360
        return moved.length()

    def _try_move(self, delta: Vec2, cave: CaveMap):
        # Try full move; if blocked, try axis-aligned components to allow sliding along walls.
        if self._can_occupy(cave, self.pos + delta):
            self.pos += delta
            return delta

        # Attempt x then y independently for smoother sliding.
        slide_attempts = [Vec2(delta.x, 0), Vec2(0, delta.y)]
        for attempt in slide_attempts:
            candidate = self.pos + attempt
            if self._can_occupy(cave, candidate):
                self.pos = candidate
                return attempt
        return Vec2(0, 0)

    def _can_occupy(self, cave: CaveMap, pos: Vec2) -> bool:
        # Sample a few points around the circumference to reduce snagging.
        r = self.radius
        skin = r * 0.7
        offsets = [
            Vec2(0, 0),
            Vec2(r, 0),
            Vec2(-r, 0),
            Vec2(0, r),
            Vec2(0, -r),
            Vec2(skin, skin),
            Vec2(-skin, skin),
            Vec2(skin, -skin),
            Vec2(-skin, -skin),
        ]
        return all(not cave.is_wall(pos + off) for off in offsets)

    def _cast_single_ray(self, cave: CaveMap, bush_lookup, angle_deg: float, max_dist: float) -> Tuple[Vec2, str, float]:
        direction = Vec2(1, 0).rotate(angle_deg)
        step = direction.normalize() * (cave.cell_size * settings.RAY_STEP_FRACTION)
        current = Vec2(self.pos)
        traveled = 0.0
        while traveled < max_dist:
            current += step
            traveled += step.length()
            cell_x = int(current.x // cave.cell_size)
            cell_y = int(current.y // cave.cell_size)
            if cave.is_wall(current):
                return current, "wall", traveled
            if (cell_x, cell_y) in bush_lookup:
                bush = bush_lookup[(cell_x, cell_y)]
                hit_type = "bush_fruit" if bush.has_fruit else "bush_empty"
                # Snap hit point to bush center for clarity
                return bush.center, hit_type, traveled
        return self.pos + direction.normalize() * max_dist, "none", max_dist

    def cast_rays(self, cave: CaveMap, bushes, max_dist: float, fov_deg: float, ray_count: int) -> List[Tuple[Vec2, str, float]]:
        if ray_count <= 1:
            bush_lookup = {(b.cell[0], b.cell[1]): b for b in bushes}
            return [self._cast_single_ray(cave, bush_lookup, self.angle, max_dist)]
        bush_lookup = {(b.cell[0], b.cell[1]): b for b in bushes}
        start_angle = self.angle - fov_deg / 2.0
        step = fov_deg / (ray_count - 1)
        hits = []
        for i in range(ray_count):
            ang = start_angle + step * i
            hits.append(self._cast_single_ray(cave, bush_lookup, ang, max_dist))
        return hits

    def draw(self, surface: pygame.Surface):
        pygame.draw.circle(surface, settings.PLAYER_COLOR, self.pos, self.radius)
        nose = self.pos + Vec2(self.radius + 6, 0).rotate(self.angle)
        pygame.draw.line(surface, settings.PLAYER_NOSE_COLOR, self.pos, nose, 3)


class Game:
    def __init__(self):
        pygame.init()
        self.map_width = settings.MAP_WIDTH
        self.map_height = settings.MAP_HEIGHT
        self.sidebar_width = settings.SIDEBAR_WIDTH
        self.screen = pygame.display.set_mode((self.map_width + self.sidebar_width, self.map_height))
        pygame.display.set_caption("Cave Line-of-Sight Demo")
        self.clock = pygame.time.Clock()

        self.cell_size = settings.CELL_SIZE
        self.cave = CaveMap(self.map_width // self.cell_size, self.map_height // self.cell_size, self.cell_size)
        # Try to place player in an open cell near center.
        spawn = self._find_open_spot()
        self.player = Player(spawn, angle_deg=0)

        self.ray_color = settings.RAY_COLOR
        self.bushes: List[Bush] = self._spawn_bushes(settings.BUSH_COUNT)
        self.control_mode = settings.CONTROL_MODE
        # Observation size will be set after first feature build
        self.ppo_agent = None
        self.storage = None
        self.step_count = 0
        self.fast_mode = False
        self.stall_steps = 0
        self.last_pos = Vec2(self.player.pos)
        self.visited = set()
        self._mark_visited(self.player.pos)
        self.last_angle = self.player.angle
        self.prev_move = Vec2(0, 0)
        self.straight_steps = 0
        self.last_fruit_seen = 0
        self.seen_obs = set()
        self.visited_pose = set()
        self.prev_action_idx = 0
        self.recent_cells = deque(maxlen=200)

    def _find_open_spot(self) -> Tuple[float, float]:
        for _ in range(2000):
            x = random.randint(self.cell_size * 5, self.map_width - self.cell_size * 5)
            y = random.randint(self.cell_size * 5, self.map_height - self.cell_size * 5)
            if not self.cave.is_wall(Vec2(x, y)):
                return x, y
        return self.map_width // 2, self.map_height // 2

    def _spawn_bushes(self, count: int) -> List["Bush"]:
        bushes: List[Bush] = []
        tries = 0
        while len(bushes) < count and tries < 5000:
            tries += 1
            cell_x = random.randint(1, self.map_width // self.cell_size - 2)
            cell_y = random.randint(1, self.map_height // self.cell_size - 2)
            pos = Vec2(cell_x * self.cell_size + self.cell_size / 2, cell_y * self.cell_size + self.cell_size / 2)
            if self.cave.is_wall(pos):
                continue
            if any(b.cell == (cell_x, cell_y) for b in bushes):
                continue
            if not self._cell_clear_for_bush(cell_x, cell_y):
                continue
            bushes.append(Bush(cell_x, cell_y, self.cell_size))
        return bushes

    def _cell_clear_for_bush(self, cx: int, cy: int, clearance: int = 1) -> bool:
        # Require surrounding cells to be open so the player can reach the bush (avoid tight corners).
        for dy in range(-clearance, clearance + 1):
            for dx in range(-clearance, clearance + 1):
                nx, ny = cx + dx, cy + dy
                if nx < 0 or ny < 0 or nx >= self.cave.cols or ny >= self.cave.rows:
                    return False
                if self.cave.grid[ny][nx] == 1:
                    return False
        return True

    def run(self):
        running = True
        while running:
            dt = self.clock.tick(60) / 1000.0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_f:
                    self.fast_mode = not self.fast_mode

            steps = settings.FAST_STEPS if self.fast_mode else 1
            step_dt = dt / steps if steps > 0 else dt
            ray_hits = None

            last_hits = None
            for _ in range(steps):
                last_hits = self._simulate_step(step_dt)

            # Drawing (once per frame)
            self._render(last_hits)

        pygame.quit()
        sys.exit()

    def _update_energy(self, dt: float, dist_moved: float):
        decay = settings.ENERGY_IDLE_DRAIN_PER_S * dt + settings.ENERGY_MOVE_DRAIN_PER_PX * dist_moved
        self.player.energy = max(0.0, self.player.energy - decay)

    def _update_bushes(self, dt: float):
        for bush in self.bushes:
            bush.update(dt)

    def _check_bush_collisions(self):
        for bush in self.bushes:
            if bush.has_fruit and bush.contains_point(self.player.pos):
                bush.consume()
                self.player.energy = min(settings.ENERGY_MAX, self.player.energy + settings.FRUIT_ENERGY_GAIN)
                return True
        return False

    def _simulate_step(self, dt: float):
        done = False
        next_feats = None
        # Perception
        ray_hits = self.player.cast_rays(
            self.cave, self.bushes, settings.MAX_VIEW_DIST, settings.FOV_DEG, settings.RAY_COUNT
        )
        tactile_codes = self._sense_adjacent_tiles(for_agent=True)
        fruit_seen = sum(1 for _, t, _ in ray_hits if t == "bush_fruit")
        obs_sig = self._obs_signature(ray_hits)
        pose_new = self._mark_visited_pose(self.player.pos, self.player.angle)
        stall_norm = min(1.0, self.stall_steps / settings.STALL_STEPS)
        avg_open, lr_bias = self._openness(ray_hits)

        if self.control_mode == "nn":
            obs_vec = self._build_feature_vector(ray_hits, tactile_codes, avg_open, lr_bias, stall_norm)
            if self.ppo_agent is None:
                obs_size = len(obs_vec)
                self.ppo_agent = PPOAgent(obs_size, len(self._actions_list()))
                self.storage = RolloutStorage(settings.ROLLOUT_STEPS, obs_size)
            obs_tensor = torch.tensor(obs_vec, dtype=torch.float32)
            action_idx, logprob, value, _ = self.ppo_agent.act(obs_tensor, explore=True)
            keys = self._action_to_keys(action_idx)
            feats = obs_vec
        else:
            keys = pygame.key.get_pressed()
            feats = []
            action_idx = 0
            logprob = torch.tensor(0.0)
            value = torch.tensor(0.0)

        pos_before = Vec2(self.player.pos)
        dist_moved = self.player.update(dt, self.cave, keys)
        pos_after = Vec2(self.player.pos)
        move_vec = pos_after - pos_before
        energy_before = self.player.energy
        self._update_energy(dt, dist_moved)
        self._update_bushes(dt)
        fruit_gained = self._check_bush_collisions()
        energy_after = self.player.energy

        # Reward shaping
        reward = 0.0
        reward += (energy_after - energy_before) * 0.05
        reward += dist_moved * 0.08  # encourage exploration
        # Penalize failed translation only if trying to move (not just turning)
        translating = any(keys[k] for k in (pygame.K_w, pygame.K_a, pygame.K_s, pygame.K_d))
        if dist_moved == 0 and translating:
            reward -= 0.2  # bumping into walls
        if fruit_gained:
            reward += settings.FRUIT_REWARD_BONUS
        # Seeing fruit in FOV is a positive signal
        if fruit_seen > self.last_fruit_seen:
            reward += settings.FRUIT_SEEN_BONUS * (fruit_seen - self.last_fruit_seen)
        self.last_fruit_seen = fruit_seen
        stall_factor = 1.0 + min(3.0, self.stall_steps / 40.0)
        if obs_sig not in self.seen_obs:
            reward += settings.NEW_OBS_REWARD * stall_factor
            self.seen_obs.add(obs_sig)
        if pose_new:
            reward += settings.NEW_POSE_REWARD * stall_factor

        # Coverage bonus
        if self._mark_visited(self.player.pos):
            reward += settings.NEW_CELL_REWARD

        # Move toward fruit globally (not just line-of-sight)
        nearest_fruit_before = self._nearest_fruit_world(pos_before)
        nearest_fruit_after = self._nearest_fruit_world(pos_after)
        if nearest_fruit_before is not None and nearest_fruit_after is not None:
            reward += (nearest_fruit_before - nearest_fruit_after) * settings.FRUIT_APPROACH_GAIN

        nearest_before = self._nearest_fruit_dist(ray_hits)
        nearest_after = self._nearest_fruit_dist(
            self.player.cast_rays(self.cave, self.bushes, settings.MAX_VIEW_DIST, settings.FOV_DEG, settings.RAY_COUNT)
        )
        if nearest_before is not None and nearest_after is not None:
            reward += (nearest_before - nearest_after) * 0.1

        # Wall proximity penalty (but soften when making fruit progress)
        wall_dist = self._nearest_wall_dist(ray_hits)
        if wall_dist is not None and wall_dist < settings.CLOSE_WALL_DIST:
            penalty_scale = 1.0
            if nearest_before is not None and nearest_after is not None and nearest_after < nearest_before:
                penalty_scale = 0.3
            reward -= settings.CLOSE_WALL_PENALTY * (1 + (settings.CLOSE_WALL_DIST - wall_dist) / settings.CLOSE_WALL_DIST) * penalty_scale

        # Stall detection
        if dist_moved < 0.5:
            self.stall_steps += 1
        else:
            self.stall_steps = 0
        if self.stall_steps == settings.STALL_STEPS:
            reward -= settings.STALL_PENALTY
        if self.stall_steps >= settings.STALL_RESPAWN_STEPS:
            reward -= settings.STALL_PENALTY * 2
            done = True
            next_feats = None
            self._respawn_player()
            self.stall_steps = 0

        # Straight-line penalty if not improving
        heading_delta = abs((self.player.angle - self.last_angle + 180) % 360 - 180)
        if heading_delta < settings.TURN_BONUS_THRESHOLD_DEG:
            self.straight_steps += 1
        else:
            self.straight_steps = 0
        if self.straight_steps >= settings.STRAIGHT_STEPS:
            reward -= settings.STRAIGHT_PENALTY
            self.straight_steps = 0

        if energy_after <= 0:
            reward -= 8.0
            done = True
            next_feats = None
            self._respawn_player()

        # Turning bonus when stuck
        angle_delta = abs((self.player.angle - self.last_angle + 180) % 360 - 180)
        if dist_moved < 0.5 and angle_delta > settings.TURN_BONUS_THRESHOLD_DEG:
            reward += settings.TURN_BONUS * (angle_delta / 180.0)
        self.last_angle = self.player.angle

        # Back-and-forth oscillation penalty (only when not already stalled)
        if dist_moved > 0 and self.prev_move.length_squared() > 0:
            if self.stall_steps < 15:
                if self.prev_move.normalize().dot(move_vec.normalize()) < -0.2:
                    reward -= settings.BACKFORTH_PENALTY
        if dist_moved > 0:
            self.prev_move = move_vec

        if self.control_mode == "nn":
            if not self.ppo_agent:
                obs_size = len(feats)
                self.ppo_agent = PPOAgent(obs_size, len(self._actions_list()))
                self.storage = RolloutStorage(settings.ROLLOUT_STEPS, obs_size)
            self.storage.add(
                torch.tensor(feats, dtype=torch.float32),
                action_idx,
                logprob,
                reward,
                done,
                value,
            )
            if len(self.storage.rewards) >= settings.ROLLOUT_STEPS:
                last_value = 0.0 if done else value.item()
                returns, advantages = self.storage.compute_advantages(last_value)
                storage_dict = {
                    "obs": torch.stack(self.storage.obs).unsqueeze(0),  # (1,T,obs)
                    "actions": torch.tensor(self.storage.actions).unsqueeze(0),
                    "logprobs": torch.stack(self.storage.logprobs).unsqueeze(0),
                    "returns": torch.tensor(returns).unsqueeze(0),
                    "advantages": torch.tensor(advantages).unsqueeze(0),
                }
                self.ppo_agent.update(storage_dict)
                self.storage.reset()
            self.prev_action_idx = action_idx
        return ray_hits

    def _nearest_fruit_dist(self, ray_hits):
        dists = [d for _, hit_type, d in ray_hits if hit_type == "bush_fruit"]
        if not dists:
            return None
        return min(dists)

    def _nearest_wall_dist(self, ray_hits):
        dists = [d for _, hit_type, d in ray_hits if hit_type == "wall"]
        if not dists:
            return None
        return min(dists)

    def _openness(self, ray_hits):
        if not ray_hits:
            return 0.0, 0.0
        total = sum(d for _, _, d in ray_hits)
        avg_open = total / (len(ray_hits) * settings.MAX_VIEW_DIST)
        third = max(1, len(ray_hits) // 3)
        left_open = sum(d for _, _, d in ray_hits[:third])
        right_open = sum(d for _, _, d in ray_hits[-third:])
        lr_bias = (right_open - left_open) / (right_open + left_open + 1e-6)
        return avg_open, lr_bias

    def _nearest_fruit_world(self, pos: Vec2):
        # World-distance to closest ripe bush
        ripe = [b for b in self.bushes if b.has_fruit]
        if not ripe:
            return None
        return min((b.center - pos).length() for b in ripe)

    def _mark_visited(self, pos: Vec2):
        cell = (int(pos.x // self.cell_size), int(pos.y // self.cell_size))
        if cell not in self.visited:
            self.visited.add(cell)
            return True
        return False

    def _mark_visited_pose(self, pos: Vec2, angle_deg: float):
        cell = (int(pos.x // self.cell_size), int(pos.y // self.cell_size))
        angle_bin = int(angle_deg // 30) % 12
        key = (cell[0], cell[1], angle_bin)
        if key in self.visited_pose:
            return False
        self.visited_pose.add(key)
        return True

    def _obs_signature(self, ray_hits):
        # coarse signature: sample 12 rays across existing hits
        if not ray_hits:
            return None
        sample = []
        stride = max(1, len(ray_hits) // 12)
        for i in range(0, len(ray_hits), stride):
            pt, hit_type, dist = ray_hits[i]
            dist_bin = max(0, min(10, int((dist / settings.MAX_VIEW_DIST) * 10)))
            sample.append((hit_type, dist_bin))
            if len(sample) >= 12:
                break
        return tuple(sample)

    def _respawn_player(self):
        self.player.pos = Vec2(self._find_open_spot())
        self.player.energy = settings.ENERGY_MAX
        self.player.angle = random.uniform(0, 360)

    def _render(self, ray_hits):
        # Drawing
        self.screen.fill(settings.BACKGROUND_COLOR)
        world_surface = self.screen.subsurface((0, 0, self.map_width, self.map_height))
        world_surface.fill(settings.WORLD_BG_COLOR)
        self.cave.draw(world_surface)
        for bush in self.bushes:
            bush.draw(world_surface)

        for pt, hit_type, _ in ray_hits:
            pygame.draw.line(world_surface, self.ray_color, self.player.pos, pt, 1)
            color = {
                "wall": settings.FOV_WALL_COLOR,
                "bush_fruit": settings.FOV_FRUIT_COLOR,
                "bush_empty": settings.FOV_BUSH_COLOR,
                "none": settings.FOV_CLEAR_COLOR,
            }.get(hit_type, settings.FOV_CLEAR_COLOR)
            pygame.draw.circle(world_surface, color, pt, 2)
        self.player.draw(world_surface)

        sidebar = self.screen.subsurface((self.map_width, 0, self.sidebar_width, self.map_height))
        sidebar.fill(settings.SIDEBAR_BG_COLOR)
        self._draw_sidebar(sidebar, ray_hits)

        pygame.display.flip()

    def _actions_list(self):
        return [
            (),
            (pygame.K_w,),
            (pygame.K_s,),
            (pygame.K_a,),
            (pygame.K_d,),
            (pygame.K_q,),
            (pygame.K_e,),
            (pygame.K_w, pygame.K_q),
            (pygame.K_w, pygame.K_e),
        ]

    def _action_to_keys(self, action_idx: int):
        actions = self._actions_list()
        mapping = {k: True for k in actions[action_idx]} if action_idx < len(actions) else {}
        return PolicyAction(mapping)

    def player_cast_count(self):
        return min(settings.RAY_COUNT, 24)

    def _build_feature_vector(self, ray_hits, tactile_codes, avg_open, lr_bias, stall_norm):
        feats = []
        stride = max(1, len(ray_hits) // self.player_cast_count()) if ray_hits else 1
        sampled = ray_hits[::stride][: self.player_cast_count()] if ray_hits else []
        if len(sampled) < self.player_cast_count() and ray_hits:
            sampled += [ray_hits[-1]] * (self.player_cast_count() - len(sampled))
        for _, hit_type, dist in sampled:
            d_norm = max(0.0, min(1.0, dist / settings.MAX_VIEW_DIST))
            type_code = {"wall": 1.0, "bush_empty": 0.5, "bush_fruit": 1.5, "none": 0.0}.get(hit_type, 0.0)
            feats.extend([d_norm, type_code])
        for code in tactile_codes:
            feats.append({"wall": 1.0, "open": 0.0, "bush_empty": 0.5, "bush_fruit": 1.5, "unknown": 0.0}.get(code, 0.0))
        feats.append(max(0.0, min(1.0, self.player.energy / settings.ENERGY_MAX)))
        feats.append(self.prev_action_idx / max(1, (len(self._actions_list()) - 1)))
        feats.append(stall_norm)
        feats.extend([avg_open, lr_bias])
        return feats
    def _draw_sidebar(self, surface: pygame.Surface, ray_hits: List[Tuple[Vec2, str, float]]):
        w, h = surface.get_size()
        font = pygame.font.SysFont("arial", 14)

        self._draw_fov_bar(surface, ray_hits, font)
        self._draw_energy(surface, font)
        self._draw_tactile(surface, font)

        center = Vec2(w * 0.5, h * 0.5)
        pygame.draw.circle(surface, settings.COMPASS_RING_COLOR, center, 60, 2)
        # Cardinal markers
        markers = [("N", 0), ("E", 90), ("S", 180), ("W", 270)]
        for label, angle in markers:
            offset = Vec2(0, -72).rotate(angle)
            text = font.render(label, True, settings.TEXT_DIM_COLOR)
            text_rect = text.get_rect(center=(center + offset))
            surface.blit(text, text_rect)

        # Use the middle ray as our direction reference.
        mid_hit = ray_hits[len(ray_hits) // 2][0] if ray_hits else self.player.pos
        dir_vec = (mid_hit - self.player.pos)
        if dir_vec.length_squared() > 0:
            dir_vec = dir_vec.normalize() * 80
        end = center + dir_vec
        pygame.draw.line(surface, self.ray_color, center, end, 4)
        pygame.draw.circle(surface, settings.FOV_CLEAR_COLOR, end, 6)

        if self.control_mode == "manual":
            control_lines = ["Controls:", "WASD - Move", "Q/E  - Turn", "Esc  - Quit"]
        else:
            control_lines = ["Mode: Neural agent", "Esc  - Quit"]
        text_lines = control_lines + [f"FOV: {settings.FOV_DEG:.0f}° | Rays: {settings.RAY_COUNT}"]
        for i, line in enumerate(text_lines):
            t = font.render(line, True, settings.TEXT_COLOR)
            surface.blit(t, (16, h - 130 + i * 18))

    def _draw_fov_bar(self, surface: pygame.Surface, ray_hits: List[Tuple[Vec2, str, float]], font):
        if not ray_hits:
            return
        w = surface.get_width()
        margin = settings.FOV_BAR_MARGIN
        bar_width = w - margin * 2
        seg_w = bar_width / len(ray_hits)
        top = settings.FOV_BAR_TOP
        for i, (_, hit_type, dist) in enumerate(ray_hits):
            # distance shading for openness
            ratio = max(0.0, min(1.0, dist / settings.MAX_VIEW_DIST))
            shade = 255 - int(90 * ratio)
            base_clear = (shade, shade, shade)
            if hit_type == "wall":
                color = settings.FOV_WALL_COLOR
            elif hit_type == "bush_fruit":
                color = settings.FOV_FRUIT_COLOR
            elif hit_type == "bush_empty":
                color = settings.FOV_BUSH_COLOR
            else:
                color = base_clear
            rect = pygame.Rect(margin + i * seg_w, top, max(seg_w - 1, 1), settings.FOV_BAR_HEIGHT)
            surface.fill(color, rect)
        label = font.render("FOV scan", True, settings.TEXT_COLOR)
        surface.blit(label, (margin, top + settings.FOV_BAR_HEIGHT + 6))

    def _draw_energy(self, surface: pygame.Surface, font):
        margin = 16
        top = settings.FOV_BAR_TOP + settings.FOV_BAR_HEIGHT + 32
        width = surface.get_width() - margin * 2
        height = 14
        pygame.draw.rect(surface, settings.ENERGY_BAR_BG, pygame.Rect(margin, top, width, height), border_radius=3)
        pct = self.player.energy / settings.ENERGY_MAX if settings.ENERGY_MAX > 0 else 0
        fill_w = max(0, min(width, int(width * pct)))
        pygame.draw.rect(surface, settings.ENERGY_BAR_FILL, pygame.Rect(margin, top, fill_w, height), border_radius=3)
        label = font.render(f"Energy: {self.player.energy:4.0f}/{settings.ENERGY_MAX:.0f}", True, settings.TEXT_COLOR)
        surface.blit(label, (margin, top - 18))

    def _draw_tactile(self, surface: pygame.Surface, font):
        margin = 16
        top = settings.FOV_BAR_TOP + settings.FOV_BAR_HEIGHT + 64
        info = self._sense_adjacent_tiles(for_agent=False)
        label = font.render("Adjacent tiles:", True, settings.TEXT_COLOR)
        surface.blit(label, (margin, top))
        for i, (name, desc, color) in enumerate(info):
            text = font.render(f"{name}: {desc}", True, color)
            surface.blit(text, (margin, top + 18 + i * 16))

    def _sense_adjacent_tiles(self, for_agent: bool = False):
        results = []
        offsets = [("N", (0, -1)), ("E", (1, 0)), ("S", (0, 1)), ("W", (-1, 0))]
        cell_x = int(self.player.pos.x // self.cell_size)
        cell_y = int(self.player.pos.y // self.cell_size)
        bush_lookup = {(b.cell[0], b.cell[1]): b for b in self.bushes}
        for name, (dx, dy) in offsets:
            cx, cy = cell_x + dx, cell_y + dy
            desc = "unknown"
            color = settings.TEXT_DIM_COLOR
            code = "unknown"
            if cx < 0 or cy < 0 or cx >= self.cave.cols or cy >= self.cave.rows:
                desc = "wall"
                color = settings.FOV_WALL_COLOR
                code = "wall"
            elif (cx, cy) in bush_lookup:
                bush = bush_lookup[(cx, cy)]
                if bush.has_fruit:
                    desc = "bush (fruit)"
                    color = settings.FRUIT_COLOR
                    code = "bush_fruit"
                else:
                    desc = "bush (growing)"
                    color = settings.BUSH_COLOR
                    code = "bush_empty"
            elif self.cave.grid[cy][cx] == 1:
                desc = "wall"
                color = settings.FOV_WALL_COLOR
                code = "wall"
            else:
                desc = "open"
                color = settings.FOV_CLEAR_COLOR
                code = "open"
            if for_agent:
                results.append(code)
            else:
                results.append((name, desc, color))
        return results


if __name__ == "__main__":
    Game().run()
