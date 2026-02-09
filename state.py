from __future__ import annotations

from dataclasses import dataclass
import math
import random

from config import Config
from grid import cell_center
from world_gen import generate_cave, find_spawn_cell
from foraging import init_bushes, reset_bush_memory
from animals import Animal, spawn_animals
from sound import SoundEvent
from wolves import Wolf, spawn_wolves


@dataclass
class AgentState:
    agent_id: int
    pos: tuple[float, float]
    angle: float
    subjective: list[list[float]]
    ever_seen: list[list[bool]]
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
    comm_emit_timer: float
    comm_region_known: list[float]
    comm_call_probs: list[list[float]]
    comm_last_heard: tuple[int, int, float] | None
    sync_text_timer: float
    explore_bias_angle: float
    hunger: float
    last_harvest_cell: tuple[int, int] | None
    respawn_estimate_sec: float | None
    hp: float
    attack_cooldown: float
    downed: bool
    animal_last_seen: dict[int, tuple[float, float]]


@dataclass
class GameState:
    # World
    objective: list[list[int]]

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

    # Animals
    animals: list[Animal]
    animal_memory: dict[int, "AnimalMemory"]
    hunt_target_id: int | None
    hunt_retarget_timer: float
    hunt_search_timer: float
    hunt_last_search_center: tuple[int, int] | None

    # Sound
    sound_events: list[SoundEvent]

    # Wolves / Combat
    wolves: list[Wolf]
    player_hp: float
    player_attack_cooldown: float
    flee_timer: float
    combat_target_id: int | None

    # Agents
    agents: list[AgentState]
    game_time: float
    meet_next_time: float
    meet_active_until: float
    meet_cell: tuple[int, int] | None


@dataclass
class AnimalMemory:
    last_seen_pos: tuple[float, float] | None
    last_seen_cell: tuple[int, int] | None
    last_seen_time: float
    vel_est: tuple[float, float]
    confidence: float
    belief_pos: tuple[float, float] | None
    belief_cell: tuple[int, int] | None
    belief_radius: float
    belief_heading: float
    belief_last_update: float


def reset_belief(cfg: Config):
    subjective = [[0.5 for _ in range(cfg.tiles_y)] for _ in range(cfg.tiles_x)]
    ever_seen = [[False for _ in range(cfg.tiles_y)] for _ in range(cfg.tiles_x)]
    return subjective, ever_seen


def init_state(cfg: Config, seed: int) -> GameState:
    objective, main_region = generate_cave(cfg, seed)

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
    spawn_cell_a = find_spawn_cell(cfg, objective, main_region, rng, cfg.spawn_clearance)
    spawn_cell_b = spawn_cell_a

    animals = (
        spawn_animals(cfg, objective, main_region, random.Random(seed + 4242))
        if cfg.enable_animals
        else []
    )
    wolves = (
        spawn_wolves(cfg, objective, main_region, random.Random(seed + 6969))
        if cfg.enable_wolves
        else []
    )
    animal_memory: dict[int, AnimalMemory] = {}
    for animal in animals:
        animal_memory[animal.id] = AnimalMemory(
            last_seen_pos=None,
            last_seen_cell=None,
            last_seen_time=-1e9,
            vel_est=(0.0, 0.0),
            confidence=0.0,
            belief_pos=None,
            belief_cell=None,
            belief_radius=0.0,
            belief_heading=0.0,
            belief_last_update=-1e9,
        )

    regions = max(1, cfg.comm_grid_x * cfg.comm_grid_y)
    call_probs = [
        [1.0 / regions for _ in range(regions)] for _ in range(max(1, cfg.comm_calls))
    ]

    agents = []
    base_bias = random.Random(seed + 12345).random() * 2.0 * 3.141592653589793
    for idx, spawn_cell in enumerate([spawn_cell_a, spawn_cell_b]):
        bias = base_bias + (math.pi if idx == 1 else 0.0)
        subjective, ever_seen = reset_belief(cfg)
        agent = AgentState(
            agent_id=idx,
            pos=cell_center(cfg, *spawn_cell),
            angle=0.0,
            subjective=subjective,
            ever_seen=ever_seen,
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
            comm_emit_timer=0.0,
            comm_region_known=[0.0 for _ in range(regions)],
            comm_call_probs=[row[:] for row in call_probs],
            comm_last_heard=None,
            sync_text_timer=0.0,
            explore_bias_angle=bias,
            hunger=0.0,
            last_harvest_cell=None,
            respawn_estimate_sec=None,
            hp=cfg.player_max_hp,
            attack_cooldown=0.0,
            downed=False,
            animal_last_seen={},
        )
        agents.append(agent)

    return GameState(
        objective=objective,
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
        animals=animals,
        animal_memory=animal_memory,
        hunt_target_id=None,
        hunt_retarget_timer=0.0,
        hunt_search_timer=0.0,
        hunt_last_search_center=None,
        sound_events=[],
        wolves=wolves,
        player_hp=cfg.player_max_hp,
        player_attack_cooldown=0.0,
        flee_timer=0.0,
        combat_target_id=None,
        agents=agents,
        game_time=0.0,
        meet_next_time=cfg.meet_interval_sec,
        meet_active_until=-1.0,
        meet_cell=None,
    )
