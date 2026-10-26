"""Frame-to-frame tracking of enemy heads to estimate their velocity.

Positions are relative to our head (screen centre). Because the camera
follows us, the measured velocity is the *relative* velocity, which is exactly
what collision prediction needs.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

from .vision import EnemyHead


@dataclass
class Track:
    id: int
    x: float
    y: float
    radius: float
    last_seen: float
    vx: float = 0.0
    vy: float = 0.0
    hits: int = 1
    body_area: float = 0.0
    meas_x: float = 0.0
    meas_y: float = 0.0

    def __post_init__(self) -> None:
        self.meas_x, self.meas_y = self.x, self.y

    @property
    def dist(self) -> float:
        return math.hypot(self.x, self.y)

    @property
    def closing_speed(self) -> float:
        """Speed at which the head approaches us (positive = coming closer)."""
        d = self.dist
        return 0.0 if d == 0 else -(self.x * self.vx + self.y * self.vy) / d


class HeadTracker:
    def __init__(self, max_speed: float, max_age: float, vel_alpha: float) -> None:
        self.max_speed = max_speed
        self.max_age = max_age
        self.alpha = vel_alpha
        self.tracks: dict[int, Track] = {}
        self.lost: list[Track] = []
        self._ids = itertools.count(1)
        self._t: float | None = None

    def reset(self) -> None:
        self.tracks.clear()
        self.lost = []
        self._t = None

    def update(self, heads: list[EnemyHead], t: float, cx: float, cy: float) -> list[Track]:
        dt = 0.0 if self._t is None else max(t - self._t, 0.0)
        self._t = t
        for tr in self.tracks.values():          # coast all tracks to "now"
            tr.x += tr.vx * dt
            tr.y += tr.vy * dt

        dets = [(hd.x - cx, hd.y - cy, hd.radius) for hd in heads]
        areas = [hd.body_area for hd in heads]
        pairs = []
        for tid, tr in self.tracks.items():
            gate = self.max_speed * max(t - tr.last_seen, 1.0 / 60.0) + 2.0 * tr.radius
            for k, (x, y, _) in enumerate(dets):
                d = math.hypot(x - tr.x, y - tr.y)
                if d <= gate:
                    pairs.append((d, tid, k))
        pairs.sort()

        used_tracks, used_dets = set(), set()
        for _, tid, k in pairs:                   # greedy nearest-neighbour assignment
            if tid in used_tracks or k in used_dets:
                continue
            used_tracks.add(tid)
            used_dets.add(k)
            tr = self.tracks[tid]
            x, y, r = dets[k]
            elapsed = t - tr.last_seen
            if elapsed > 0:
                # measure against the last *measured* position, not the coasted one
                mvx, mvy = (x - tr.meas_x) / elapsed, (y - tr.meas_y) / elapsed
                a = self.alpha if tr.hits > 1 else 1.0
                tr.vx += a * (mvx - tr.vx)
                tr.vy += a * (mvy - tr.vy)
            tr.x, tr.y, tr.radius, tr.last_seen = x, y, r, t
            tr.meas_x, tr.meas_y = x, y
            tr.body_area = areas[k]
            tr.hits += 1

        for k, (x, y, r) in enumerate(dets):
            if k not in used_dets:
                tid = next(self._ids)
                self.tracks[tid] = Track(tid, x, y, r, t, body_area=areas[k])

        # tracks dropped this frame (at their last *measured* position) -> death candidates
        self.lost = [self.tracks.pop(tid) for tid in
                     [tid for tid, tr in self.tracks.items() if t - tr.last_seen > self.max_age]]
        return list(self.tracks.values())
