from __future__ import annotations

from dataclasses import dataclass
import random

from config import Config
from grid import cell_center
from world_gen import generate_cave, find_spawn_cell
from foraging import init_bushes, reset_bush_memory


@dataclass
class GameState:
    # World
    objective: list[list[int]]
    subjective: list[list[float]]
    ever_seen: list[list[bool]]

    # Foraging world state
    bushes: list[list[bool]]
    fruit: list[list[bool]]
    last_harvest_time: list[list[float]]
    bush_positions: list[tuple[int, int]]

    # Foraging memory
    bush_known: list[list[bool]]
    bush_last_seen_time: list[list[float]]
    bush_last_seen_fruit: list[list[bool]]
    bush_last_harvest_time: list[list[float]]
    bush_last_empty_time: list[list[float]]
    bush_last_checked_empty_time: list[list[float]]
    respawn_estimate_sec: float | None
    respawn_lower_bound: float
    respawn_ema_alpha: float

    # Agent state
    player_pos: tuple[float, float]
    player_angle: float
    target_cell: tuple[int, int] | None
    target_kind: str | None
    search_state: str
    stuck_frames: int
    scan_remaining: float
    scan_dir: float
    scan_reason: str | None
    scan_goal: str | None
    scan_target_angle: float | None
    last_move_angle: float | None
    last_blocked_cell: tuple[int, int] | None
    last_target_cell: tuple[int, int] | None
    last_target_dist: int | None
    frames_no_new: int
    frames_no_progress: int
    move_accum: float
    hunger: float
    game_time: float
    last_harvest_cell: tuple[int, int] | None


def reset_belief(cfg: Config):
    subjective = [[0.5 for _ in range(cfg.tiles_y)] for _ in range(cfg.tiles_x)]
    ever_seen = [[False for _ in range(cfg.tiles_y)] for _ in range(cfg.tiles_x)]
    return subjective, ever_seen


def init_state(cfg: Config, seed: int) -> GameState:
    objective, main_region = generate_cave(cfg, seed)
    subjective, ever_seen = reset_belief(cfg)

    bushes, fruit, last_harvest_time, bush_positions = init_bushes(
        cfg, objective, main_region, random.Random(seed + 1337)
    )
    (
        bush_known,
        bush_last_seen_time,
        bush_last_seen_fruit,
        bush_last_harvest_time,
        bush_last_empty_time,
        bush_last_checked_empty_time,
    ) = reset_bush_memory(cfg)

    rng = random.Random(seed)
    spawn_cell = find_spawn_cell(cfg, objective, main_region, rng, cfg.spawn_clearance)
    player_pos = cell_center(cfg, *spawn_cell)

    return GameState(
        objective=objective,
        subjective=subjective,
        ever_seen=ever_seen,
        bushes=bushes,
        fruit=fruit,
        last_harvest_time=last_harvest_time,
        bush_positions=bush_positions,
        bush_known=bush_known,
        bush_last_seen_time=bush_last_seen_time,
        bush_last_seen_fruit=bush_last_seen_fruit,
        bush_last_harvest_time=bush_last_harvest_time,
        bush_last_empty_time=bush_last_empty_time,
        bush_last_checked_empty_time=bush_last_checked_empty_time,
        respawn_estimate_sec=None,
        respawn_lower_bound=0.0,
        respawn_ema_alpha=0.3,
        player_pos=player_pos,
        player_angle=0.0,
        target_cell=None,
        target_kind=None,
        search_state="idle",
        stuck_frames=0,
        scan_remaining=0.0,
        scan_dir=1.0,
        scan_reason=None,
        scan_goal=None,
        scan_target_angle=None,
        last_move_angle=None,
        last_blocked_cell=None,
        last_target_cell=None,
        last_target_dist=None,
        frames_no_new=0,
        frames_no_progress=0,
        move_accum=0.0,
        hunger=0.0,
        game_time=0.0,
        last_harvest_cell=None,
    )
