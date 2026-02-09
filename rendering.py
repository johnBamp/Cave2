from __future__ import annotations

import math
import pygame

from config import Config
from grid import clamp, world_to_cell
from exploration import find_frontier_clusters, pick_cluster_representative


def draw_player(cfg: Config, screen, pos, angle, cam_x, cam_y, color=None):
    px = int(pos[0] - cam_x)
    py = int(pos[1] - cam_y)
    draw_color = cfg.player_color if color is None else color
    pygame.draw.circle(screen, draw_color, (px, py), int(cfg.player_radius))
    length = 40
    fx = px + int(math.cos(angle) * length)
    fy = py + int(math.sin(angle) * length)
    pygame.draw.line(screen, draw_color, (px, py), (fx, fy), 3)


def draw_world(cfg: Config, state, agent, screen, cam_x, cam_y, seen_cells):
    start_x = max(0, int(cam_x // cfg.tile_size))
    end_x = min(cfg.tiles_x - 1, int((cam_x + cfg.width) // cfg.tile_size) + 1)
    start_y = max(0, int(cam_y // cfg.tile_size))
    end_y = min(cfg.tiles_y - 1, int((cam_y + cfg.height) // cfg.tile_size) + 1)

    for gx in range(start_x, end_x + 1):
        for gy in range(start_y, end_y + 1):
            sx = int(gx * cfg.tile_size - cam_x)
            sy = int(gy * cfg.tile_size - cam_y)
            rect = pygame.Rect(sx, sy, cfg.tile_size, cfg.tile_size)

            if agent.ever_seen[gx][gy]:
                v = int(clamp(agent.subjective[gx][gy], 0.0, 1.0) * 255)
                fill = (v, v, v)
            else:
                fill = cfg.unknown_fill

            pygame.draw.rect(screen, fill, rect)
            pygame.draw.rect(screen, cfg.grid_color, rect, 1)

            if cfg.show_player_fov and (gx, gy) in seen_cells:
                pygame.draw.rect(screen, cfg.player_color, rect, 2)

            if agent.ever_seen[gx][gy] and state.bushes[gx][gy]:
                pygame.draw.rect(screen, cfg.bush_color, rect, 1)
                if cfg.fruit_enabled and state.fruit[gx][gy]:
                    cx = sx + cfg.tile_size // 2
                    cy = sy + cfg.tile_size // 2
                    pygame.draw.circle(screen, cfg.fruit_color, (cx, cy), max(2, cfg.tile_size // 4))


def draw_subjective_debug(cfg: Config, agent_a, agent_b, screen, cam_x, cam_y):
    if not cfg.show_subjective_debug:
        return
    start_x = max(0, int(cam_x // cfg.tile_size))
    end_x = min(cfg.tiles_x - 1, int((cam_x + cfg.width) // cfg.tile_size) + 1)
    start_y = max(0, int(cam_y // cfg.tile_size))
    end_y = min(cfg.tiles_y - 1, int((cam_y + cfg.height) // cfg.tile_size) + 1)

    red = pygame.Surface((cfg.tile_size, cfg.tile_size), pygame.SRCALPHA)
    blue = pygame.Surface((cfg.tile_size, cfg.tile_size), pygame.SRCALPHA)
    red.fill((255, 0, 0, 80))
    blue.fill((0, 0, 255, 80))

    for gx in range(start_x, end_x + 1):
        for gy in range(start_y, end_y + 1):
            sx = int(gx * cfg.tile_size - cam_x)
            sy = int(gy * cfg.tile_size - cam_y)
            if agent_a is not None and agent_a.ever_seen[gx][gy]:
                screen.blit(red, (sx, sy))
            if agent_b is not None and agent_b.ever_seen[gx][gy]:
                screen.blit(blue, (sx, sy))


def draw_target(cfg: Config, screen, target_cell, cam_x, cam_y):
    if target_cell is None:
        return
    gx, gy = target_cell
    sx = int(gx * cfg.tile_size - cam_x)
    sy = int(gy * cfg.tile_size - cam_y)
    rect = pygame.Rect(sx, sy, cfg.tile_size, cfg.tile_size)
    pygame.draw.rect(screen, cfg.target_color, rect, 2)


def draw_frontiers(cfg: Config, agent, screen, cam_x, cam_y, dist):
    clusters = find_frontier_clusters(cfg, agent, dist)
    for cluster in clusters:
        rep = pick_cluster_representative(cluster, dist)
        gx, gy = rep
        sx = int(gx * cfg.tile_size - cam_x)
        sy = int(gy * cfg.tile_size - cam_y)
        rect = pygame.Rect(sx, sy, cfg.tile_size, cfg.tile_size)
        pygame.draw.rect(screen, (255, 255, 0), rect, 2)


def draw_animals(cfg: Config, state, screen, cam_x, cam_y, seen_cells):
    for animal in state.animals:
        if not animal.alive:
            continue
        cx, cy = world_to_cell(cfg, animal.pos[0], animal.pos[1])
        if cfg.animal_seen_only and (cx, cy) not in seen_cells:
            continue
        sx = int(animal.pos[0] - cam_x)
        sy = int(animal.pos[1] - cam_y)
        radius = max(2, cfg.tile_size // 3)
        pygame.draw.circle(screen, cfg.animal_color, (sx, sy), radius)


def draw_animals_subjective(cfg: Config, agent, screen, cam_x, cam_y, color):
    for pos in agent.animal_last_seen.values():
        sx = int(pos[0] - cam_x)
        sy = int(pos[1] - cam_y)
        radius = max(2, cfg.tile_size // 3)
        pygame.draw.circle(screen, color, (sx, sy), radius, 2)


def draw_wolves(cfg: Config, state, screen, cam_x, cam_y):
    for wolf in state.wolves:
        if not wolf.alive:
            continue
        sx = int(wolf.pos[0] - cam_x)
        sy = int(wolf.pos[1] - cam_y)
        radius = max(3, cfg.tile_size // 3)
        pygame.draw.circle(screen, cfg.wolf_color, (sx, sy), radius)
