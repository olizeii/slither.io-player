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


class HudReader(threading.Thread):
    def __init__(self, buffer: LatestFrameBuffer, cfg: HudConfig) -> None:
        super().__init__(daemon=True, name="hud-ocr")
        self.buffer = buffer
        self.cfg = cfg
        self.values = HudValues()
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
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
            try:
                length, rank, players = parse_stats(self._ocr(crops["stats"]), self.cfg)
                final = parse_final(self._ocr(crops["final"]), self.cfg)
            except Exception as exc:          # OCR engine missing language etc.
                self.error = f"OCR failed: {exc!r}"
                self.available = False
                return
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
