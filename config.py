from __future__ import annotations

from dataclasses import dataclass
import math

import settings as cfg


@dataclass(frozen=True)
class Config:
    # Window
    width: int
    height: int

    # World
    tiles_x: int
    tiles_y: int
    tile_size: int
    world_w: int
    world_h: int

    # Colors
    grid_color: tuple[int, int, int]
    black: tuple[int, int, int]
    player_color: tuple[int, int, int]
    unknown_fill: tuple[int, int, int]
    target_color: tuple[int, int, int]
    bush_color: tuple[int, int, int]
    fruit_color: tuple[int, int, int]

    # Raycast
    fov_deg: float
    rays: int
    ray_step: int
    vision_range: int

    # Movement
    rot_speed: float
    move_speed: float
    player_radius: float
    waypoint_eps: float
    stuck_frames: int
    no_new_frames: int
    no_progress_frames: int

    # Foraging
    bush_density: float
    bush_cluster_count: int
    bush_cluster_radius: int
    fruit_respawn_sec: float
    hunger_max: float
    hunger_decay_per_sec: float
    hunger_eat_amount: float
    forage_weight: float
    frontier_weight: float
    likely_fruit_value: float
    forage_priority_hunger: float
    explore_when_bushes_mult: float
    min_bush_attraction: float
    home_radius: int
    home_far_mult: float
    loiter_radius: int
    home_radius_extra: int
    frontier_satiety_bonus: float

    # Cave generation
    fill_prob: float
    smooth_passes: int
    smooth_wall_threshold: int
    smooth_empty_threshold: int
    min_region_size: int
    final_smooth_passes: int
    noise_wall_prob: float
    noise_passes: int
    corridor_radius: int
    spawn_clearance: int

    # Belief update
    empty_alpha: float
    wall_alpha: float
    wall_conf_thresh: float
    min_enclosed_size: int


def load_config() -> Config:
    return Config(
        # Window
        width=cfg.WIDTH,
        height=cfg.HEIGHT,

        # World
        tiles_x=cfg.TILES_X,
        tiles_y=cfg.TILES_Y,
        tile_size=cfg.TILE_SIZE,
        world_w=cfg.TILES_X * cfg.TILE_SIZE,
        world_h=cfg.TILES_Y * cfg.TILE_SIZE,

        # Colors
        grid_color=cfg.GRID_COLOR,
        black=cfg.BLACK,
        player_color=cfg.PLAYER_COLOR,
        unknown_fill=cfg.UNKNOWN_FILL,
        target_color=cfg.TARGET_COLOR,
        bush_color=cfg.BUSH_COLOR,
        fruit_color=cfg.FRUIT_COLOR,

        # Raycast
        fov_deg=cfg.FOV_DEG,
        rays=cfg.RAYS,
        ray_step=cfg.RAY_STEP,
        vision_range=cfg.VISION_RANGE,

        # Movement
        rot_speed=math.radians(cfg.ROT_SPEED_DEG),
        move_speed=cfg.MOVE_SPEED,
        player_radius=cfg.TILE_SIZE * cfg.PLAYER_RADIUS_RATIO,
        waypoint_eps=cfg.TILE_SIZE * cfg.WAYPOINT_EPS_RATIO,
        stuck_frames=cfg.STUCK_FRAMES,
        no_new_frames=cfg.NO_NEW_FRAMES,
        no_progress_frames=cfg.NO_PROGRESS_FRAMES,

        # Foraging
        bush_density=cfg.BUSH_DENSITY,
        bush_cluster_count=cfg.BUSH_CLUSTER_COUNT,
        bush_cluster_radius=cfg.BUSH_CLUSTER_RADIUS,
        fruit_respawn_sec=cfg.FRUIT_RESPAWN_SEC,
        hunger_max=cfg.HUNGER_MAX,
        hunger_decay_per_sec=cfg.HUNGER_DECAY_PER_SEC,
        hunger_eat_amount=cfg.HUNGER_EAT_AMOUNT,
        forage_weight=cfg.FORAGE_WEIGHT,
        frontier_weight=cfg.FRONTIER_WEIGHT,
        likely_fruit_value=cfg.LIKELY_FRUIT_VALUE,
        forage_priority_hunger=cfg.FORAGE_PRIORITY_HUNGER,
        explore_when_bushes_mult=cfg.EXPLORE_WHEN_BUSHES_MULT,
        min_bush_attraction=cfg.MIN_BUSH_ATTRACTION,
        home_radius=cfg.HOME_RADIUS,
        home_far_mult=cfg.HOME_FAR_MULT,
        loiter_radius=cfg.LOITER_RADIUS,
        home_radius_extra=cfg.HOME_RADIUS_EXTRA,
        frontier_satiety_bonus=cfg.FRONTIER_SATIETY_BONUS,

        # Cave generation
        fill_prob=cfg.FILL_PROB,
        smooth_passes=cfg.SMOOTH_PASSES,
        smooth_wall_threshold=cfg.SMOOTH_WALL_THRESHOLD,
        smooth_empty_threshold=cfg.SMOOTH_EMPTY_THRESHOLD,
        min_region_size=cfg.MIN_REGION_SIZE,
        final_smooth_passes=cfg.FINAL_SMOOTH_PASSES,
        noise_wall_prob=cfg.NOISE_WALL_PROB,
        noise_passes=cfg.NOISE_PASSES,
        corridor_radius=cfg.CORRIDOR_RADIUS,
        spawn_clearance=cfg.SPAWN_CLEARANCE,

        # Belief update
        empty_alpha=cfg.EMPTY_ALPHA,
        wall_alpha=cfg.WALL_ALPHA,
        wall_conf_thresh=cfg.WALL_CONF_THRESH,
        min_enclosed_size=cfg.MIN_ENCLOSED_SIZE,
    )
