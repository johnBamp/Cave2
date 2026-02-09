from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass
class SoundEvent:
    pos: tuple[float, float]
    base_volume: float
    source_id: int | None
    kind: str
    payload: int | None = None


def emit_sound(state, pos, base_volume: float, source_id=None, kind="generic", payload=None) -> None:
    state.sound_events.append(
        SoundEvent(pos=pos, base_volume=base_volume, source_id=source_id, kind=kind, payload=payload)
    )


def process_hearing(cfg, state, listener_pos, ignore_source_id=None):
    if not state.sound_events:
        return False, None, 0.0, None

    range_world = cfg.sound_range_tiles * cfg.tile_size
    if range_world <= 0:
        return False, None, 0.0, None

    acc_x = 0.0
    acc_y = 0.0
    max_vol = 0.0
    strongest = None

    for ev in state.sound_events:
        if ignore_source_id is not None and ev.source_id == ignore_source_id:
            continue
        dx = ev.pos[0] - listener_pos[0]
        dy = ev.pos[1] - listener_pos[1]
        dist = math.hypot(dx, dy)
        vol = ev.base_volume * max(0.0, 1.0 - dist / range_world)
        if vol <= 0.0:
            continue
        acc_x += dx * vol
        acc_y += dy * vol
        if vol > max_vol:
            max_vol = vol
            strongest = ev

    if max_vol < cfg.sound_threshold:
        return False, None, max_vol, strongest

    angle = math.atan2(acc_y, acc_x)
    return True, angle, max_vol, strongest
