from __future__ import annotations

from dataclasses import dataclass
import math

import settings as cfg


@dataclass(frozen=True)
class Config:
    # Window
    width: int
    height: int
    cam_speed: float

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
    show_player_fov: bool
    show_subjective_debug: bool

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
    fruit_enabled: bool
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

    # Animals
    enable_animals: bool
    animal_count: int
    animal_speed: float
    animal_turn_chance: float
    animal_stuck_turn: float
    animal_flee_radius: int
    animal_flee_speed_mult: float
    animal_wander_radius: int
    animal_repath_sec: float
    animal_belief_growth: float
    animal_predict_max_sec: float
    animal_search_ring_step: int
    animal_search_bias_deg: float

    # Sound
    sound_enabled: bool
    sound_range_tiles: int
    sound_threshold: float
    sound_turn_sec: float
    sound_turn_speed_mult: float
    animal_sound_level: float
    sound_confidence: float

    # Wolves
    enable_wolves: bool
    wolf_count: int
    wolf_speed: float
    wolf_wander_radius: int
    wolf_repath_sec: float
    wolf_sight_range_tiles: int
    wolf_sound_level: float
    wolf_hearing_range_tiles: int

    # Combat
    player_max_hp: float
    player_dps: float
    player_attack_range_tiles: float
    player_attack_cooldown: float
    wolf_max_hp: float
    wolf_dps: float
    wolf_attack_range_tiles: float
    wolf_attack_cooldown: float

    # Fight/Flee
    flee_hp_threshold: float
    flee_outnumbered: int
    fight_hunger_threshold: float
    flee_duration_sec: float

    # Rendering
    wolf_color: tuple[int, int, int]
    show_wolf_fov: bool
    hunt_weight: float
    hunt_eat_amount: float
    hunt_retarget_sec: float
    hunt_sticky_ratio: float
    hunt_lost_sec: float
    hunt_search_radius: int
    hunt_search_time: float
    hunt_catch_dist: float
    animal_color: tuple[int, int, int]
    animal_seen_only: bool

    # Agents / Communication
    agent_colors: list[tuple[int, int, int]]
    comm_enabled: bool
    comm_calls: int
    comm_grid_x: int
    comm_grid_y: int
    comm_period_sec: float
    comm_eps: float
    comm_lr: float
    comm_merge_alpha: float
    comm_sound_level: float
    show_comm_debug: bool
    regroup_distance_tiles: int
    sync_distance_tiles: int
    separate_distance_tiles: int
    separate_weight: float
    sync_text_sec: float
    meet_interval_sec: float
    meet_window_sec: float
    explore_bias_weight: float

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
        cam_speed=cfg.CAM_SPEED,

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
        show_player_fov=cfg.SHOW_PLAYER_FOV,
        show_subjective_debug=cfg.SHOW_SUBJECTIVE_DEBUG,

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
        fruit_enabled=cfg.FRUIT_ENABLED,
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

        # Animals
        enable_animals=cfg.ENABLE_ANIMALS,
        animal_count=cfg.ANIMAL_COUNT,
        animal_speed=cfg.ANIMAL_SPEED,
        animal_turn_chance=cfg.ANIMAL_TURN_CHANCE,
        animal_stuck_turn=cfg.ANIMAL_STUCK_TURN,
        animal_flee_radius=cfg.ANIMAL_FLEE_RADIUS,
        animal_flee_speed_mult=cfg.ANIMAL_FLEE_SPEED_MULT,
        animal_wander_radius=cfg.ANIMAL_WANDER_RADIUS,
        animal_repath_sec=cfg.ANIMAL_REPATH_SEC,
        animal_belief_growth=cfg.ANIMAL_BELIEF_GROWTH,
        animal_predict_max_sec=cfg.ANIMAL_PREDICT_MAX_SEC,
        animal_search_ring_step=cfg.ANIMAL_SEARCH_RING_STEP,
        animal_search_bias_deg=cfg.ANIMAL_SEARCH_BIAS_DEG,

        # Sound
        sound_enabled=cfg.SOUND_ENABLED,
        sound_range_tiles=cfg.SOUND_RANGE_TILES,
        sound_threshold=cfg.SOUND_THRESHOLD,
        sound_turn_sec=cfg.SOUND_TURN_SEC,
        sound_turn_speed_mult=cfg.SOUND_TURN_SPEED_MULT,
        animal_sound_level=cfg.ANIMAL_SOUND_LEVEL,
        sound_confidence=cfg.SOUND_CONFIDENCE,

        # Wolves
        enable_wolves=cfg.ENABLE_WOLVES,
        wolf_count=cfg.WOLF_COUNT,
        wolf_speed=cfg.WOLF_SPEED,
        wolf_wander_radius=cfg.WOLF_WANDER_RADIUS,
        wolf_repath_sec=cfg.WOLF_REPATH_SEC,
        wolf_sight_range_tiles=cfg.WOLF_SIGHT_RANGE_TILES,
        wolf_sound_level=cfg.WOLF_SOUND_LEVEL,
        wolf_hearing_range_tiles=cfg.WOLF_HEARING_RANGE_TILES,

        # Combat
        player_max_hp=cfg.PLAYER_MAX_HP,
        player_dps=cfg.PLAYER_DPS,
        player_attack_range_tiles=cfg.PLAYER_ATTACK_RANGE_TILES,
        player_attack_cooldown=cfg.PLAYER_ATTACK_COOLDOWN,
        wolf_max_hp=cfg.WOLF_MAX_HP,
        wolf_dps=cfg.WOLF_DPS,
        wolf_attack_range_tiles=cfg.WOLF_ATTACK_RANGE_TILES,
        wolf_attack_cooldown=cfg.WOLF_ATTACK_COOLDOWN,

        # Fight/Flee
        flee_hp_threshold=cfg.FLEE_HP_THRESHOLD,
        flee_outnumbered=cfg.FLEE_OUTNUMBERED,
        fight_hunger_threshold=cfg.FIGHT_HUNGER_THRESHOLD,
        flee_duration_sec=cfg.FLEE_DURATION_SEC,

        # Rendering
        wolf_color=cfg.WOLF_COLOR,
        show_wolf_fov=cfg.SHOW_WOLF_FOV,
        hunt_weight=cfg.HUNT_WEIGHT,
        hunt_eat_amount=cfg.HUNT_EAT_AMOUNT,
        hunt_retarget_sec=cfg.HUNT_RETARGET_SEC,
        hunt_sticky_ratio=cfg.HUNT_STICKY_RATIO,
        hunt_lost_sec=cfg.HUNT_LOST_SEC,
        hunt_search_radius=cfg.HUNT_SEARCH_RADIUS,
        hunt_search_time=cfg.HUNT_SEARCH_TIME,
        hunt_catch_dist=cfg.HUNT_CATCH_DIST,
        animal_color=cfg.ANIMAL_COLOR,
        animal_seen_only=cfg.ANIMAL_SEEN_ONLY,

        # Agents / Communication
        agent_colors=list(cfg.AGENT_COLORS),
        comm_enabled=cfg.COMM_ENABLED,
        comm_calls=cfg.COMM_CALLS,
        comm_grid_x=cfg.COMM_GRID_X,
        comm_grid_y=cfg.COMM_GRID_Y,
        comm_period_sec=cfg.COMM_PERIOD_SEC,
        comm_eps=cfg.COMM_EPS,
        comm_lr=cfg.COMM_LR,
        comm_merge_alpha=cfg.COMM_MERGE_ALPHA,
        comm_sound_level=cfg.COMM_SOUND_LEVEL,
        show_comm_debug=cfg.SHOW_COMM_DEBUG,
        regroup_distance_tiles=cfg.REGROUP_DISTANCE_TILES,
        sync_distance_tiles=cfg.SYNC_DISTANCE_TILES,
        separate_distance_tiles=cfg.SEPARATE_DISTANCE_TILES,
        separate_weight=cfg.SEPARATE_WEIGHT,
        sync_text_sec=cfg.SYNC_TEXT_SEC,
        meet_interval_sec=cfg.MEET_INTERVAL_SEC,
        meet_window_sec=cfg.MEET_WINDOW_SEC,
        explore_bias_weight=cfg.EXPLORE_BIAS_WEIGHT,

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
