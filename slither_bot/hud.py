"""Reads the in-game HUD with the built-in Windows OCR (package ``winocr``).

  bottom left during play:  "Deine Länge: 1234" / "Dein rang: 5 von 537"
  death screen:             "Deine endgültige Länge war 1234"

The length is the real score of a round, so the self-tuner optimises it
directly. Without winocr the reader is disabled and the bot falls back to a
vision-based proxy (own thickness and survival time).
"""
from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass

import cv2

from .capture import LatestFrameBuffer
from .config import HudConfig


@dataclass
class HudValues:
    length: int | None = None
    rank: int | None = None
    players: int | None = None
    final_length: int | None = None
    timestamp: float = 0.0          # grab time of the crops these values came from
    final_timestamp: float = 0.0


def parse_stats(text: str, cfg: HudConfig) -> tuple[int | None, int | None, int | None]:
    length = rank = players = None
    m = re.search(cfg.length_regex, text, re.IGNORECASE)
    if m:
        length = int(m.group(1))
    m = re.search(cfg.rank_regex, text, re.IGNORECASE)
    if m:
        rank, players = int(m.group(1)), int(m.group(2))
    return length, rank, players


def parse_final(text: str, cfg: HudConfig) -> int | None:
    m = re.search(cfg.final_regex, text, re.IGNORECASE)
    return int(m.group(1)) if m else None


class PlausibleValue:
    """Filters OCR numbers.

    * A value is only accepted when two consecutive readings agree (±10 %).
    * With ``max_rise_per_s`` set, an accepted value may then only grow as fast as
      the game allows; a bigger jump needs confirmation again (new round, misread).
    """

    def __init__(self, max_rise_per_s: float | None, slack: float = 30.0) -> None:
        self.max_rise = max_rise_per_s
        self.slack = slack
        self.value: int | None = None
        self._t = 0.0
        self._pending: int | None = None

    @staticmethod
    def _close(a: int, b: int) -> bool:
        return abs(a - b) <= max(3, 0.1 * max(a, b))

    def update(self, x: int | None, t: float) -> int | None:
        if x is None or not (0 < x < 1_000_000):
            return self.value
        if self.value is not None and self.max_rise is not None:
            dt = max(t - self._t, 0.0)
            if 0.5 * self.value - self.slack <= x <= self.value + self.max_rise * dt + self.slack:
                self.value, self._t, self._pending = x, t, None
                return x
        elif self.value is not None and self._close(x, self.value):
            self.value, self._t, self._pending = x, t, None
            return x
        if self._pending is not None and self._close(x, self._pending):
            self.value, self._t, self._pending = x, t, None   # confirmed jump / new value
            return x
        self._pending = x
        return self.value


class HudReader(threading.Thread):
    def __init__(self, buffer: LatestFrameBuffer, cfg: HudConfig) -> None:
        super().__init__(daemon=True, name="hud-ocr")
        self.buffer = buffer
        self.cfg = cfg
        self.values = HudValues()
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        # overnight logs showed OCR misreads like 29301 for a length of 391: every
        # value must be physically plausible and confirmed before it counts
        self._length = PlausibleValue(max_rise_per_s=cfg.max_growth_per_s)
        self._rank = PlausibleValue(max_rise_per_s=None)
        self._final = PlausibleValue(max_rise_per_s=None)
        self.available = False
        self.error: str | None = None
        try:
            import winocr  # noqa: F401
            self.available = cfg.enabled
        except ImportError:
            self.error = "winocr not installed (pip install winocr) - using vision proxy as score"

    def snapshot(self) -> HudValues:
        with self._lock:
            return HudValues(**vars(self.values))

    def stop(self) -> None:
        self._stop_event.set()

    def _ocr(self, img) -> str:
        import winocr

        big = cv2.resize(img, None, fx=self.cfg.upscale, fy=self.cfg.upscale, interpolation=cv2.INTER_CUBIC)
        res = winocr.recognize_cv2_sync(big, self.cfg.lang)
        return res["text"] if isinstance(res, dict) else str(getattr(res, "text", res))

    def run(self) -> None:
        if not self.available:
            return
        last_ts = 0.0
        while not self._stop_event.is_set():
            hud = self.buffer.latest_hud()
            if hud is None or hud[0] == last_ts:
                time.sleep(0.05)
                continue
            ts, crops = hud
            last_ts = ts
            if "stats" not in crops:
                continue
            try:
                length, rank, players = parse_stats(self._ocr(crops["stats"]), self.cfg)
                final = parse_final(self._ocr(crops["final"]), self.cfg)
            except Exception as exc:          # OCR engine missing language etc.
                self.error = f"OCR failed: {exc!r}"
                self.available = False
                return
            length = self._length.update(length, ts)
            rank_ok = rank is not None and players is not None and 1 <= rank <= players and players >= 2
            rank = self._rank.update(rank if rank_ok else None, ts)
            final = self._final.update(final, ts)
            with self._lock:
                v = self.values
                if length is not None:
                    v.length, v.timestamp = length, ts
                if rank is not None:
                    v.rank, v.players = rank, players
                if final is not None:
                    v.final_length, v.final_timestamp = final, ts
            rest = self.cfg.interval_s - (time.perf_counter() - ts)
            if rest > 0:
                time.sleep(min(rest, self.cfg.interval_s))
