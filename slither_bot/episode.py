"""Round bookkeeping: when does a round start/end and how good was it?"""
from __future__ import annotations

from .brain import Action
from .hud import HudReader
from .tuner import Episode
from .vision import GameState


class EpisodeTracker:
    START_ALIVE_S = 0.7          # alive this long -> a round has started
    LOST_S = 4.0                 # snake not seen this long (and no menu) -> treat as dead

    def __init__(self, hud: HudReader | None, final_wait_s: float) -> None:
        self.hud = hud
        self.final_wait_s = final_wait_s
        self.phase = "idle"      # idle | playing | ending
        self._alive_since: float | None = None
        self._reset(0.0)

    def _reset(self, now: float) -> None:
        self.started = now
        self.last_alive = now
        self.last_t = now
        self.ended = now
        self.peak_length: int | None = None
        self.best_rank: int | None = None
        self.players: int | None = None
        self.peak_radius = 0.0
        self.modes: dict[str, float] = {}

    def update(self, state: GameState, action: Action, now: float) -> Episode | None:
        hv = self.hud.snapshot() if self.hud is not None and self.hud.available else None

        if self.phase == "idle":
            if state.alive:
                self._alive_since = self._alive_since or now
                if now - self._alive_since >= self.START_ALIVE_S:
                    self._reset(self._alive_since)
                    self.phase = "playing"
            else:
                self._alive_since = None
            return None

        if self.phase == "playing":
            dt = now - self.last_t
            self.last_t = now
            if state.alive:
                self.last_alive = now
                self.peak_radius = max(self.peak_radius, state.self_radius)
                key = (action.debug or {}).get("phase") or action.mode.value
                self.modes[key] = self.modes.get(key, 0.0) + dt
            if hv is not None and hv.timestamp > self.started:
                if hv.length is not None:
                    self.peak_length = max(self.peak_length or 0, hv.length)
                if hv.rank is not None and (self.best_rank is None or hv.rank < self.best_rank):
                    self.best_rank, self.players = hv.rank, hv.players
            if state.menu_button is not None or now - self.last_alive > self.LOST_S:
                self.phase = "ending"
                self.ended = now
            return None

        # ending: wait for the death screen "Deine endgültige Länge war N"
        final = None
        if hv is not None and hv.final_timestamp >= self.ended - 0.2:
            final = hv.final_length
        if final is not None or now - self.ended > self.final_wait_s or hv is None or state.alive:
            ep = Episode(
                started=self.started, duration_s=self.last_alive - self.started,
                final_length=final, peak_length=self.peak_length, best_rank=self.best_rank,
                players=self.players, peak_radius=self.peak_radius,
                modes={k: round(v, 1) for k, v in self.modes.items()},
            )
            self.phase = "idle"
            self._alive_since = None
            return ep
        return None
