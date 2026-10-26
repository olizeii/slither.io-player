"""Low-latency screen capture running in its own thread.

The capture thread grabs the game region, downsamples it to the analysis
resolution and publishes it into a single-slot buffer. The consumer always
gets the *newest* frame; stale frames are dropped instead of queued, so a
slow analysis step never builds up latency.

Backends (fastest first):
  * dxcam / bettercam - DXGI Desktop Duplication, 100-240 FPS, Windows only
  * mss               - GDI/X11/Quartz, ~30-60 FPS for a 1080p region
"""
from __future__ import annotations

import importlib
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np

from . import win32
from .config import CaptureConfig


@dataclass
class Frame:
    image: np.ndarray     # BGR, analysis resolution
    frame_id: int
    timestamp: float      # time.perf_counter() at grab time


class LatestFrameBuffer:
    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._frame: Frame | None = None
        self._next_id = 1
        self._hud: tuple[float, dict[str, np.ndarray]] | None = None

    def publish_hud(self, crops: dict[str, np.ndarray], timestamp: float) -> None:
        with self._cond:
            self._hud = (timestamp, crops)

    def latest_hud(self) -> tuple[float, dict[str, np.ndarray]] | None:
        """Full-resolution HUD crops (for OCR) and their grab time."""
        with self._cond:
            return self._hud

    def publish(self, image: np.ndarray, timestamp: float) -> None:
        with self._cond:
            self._frame = Frame(image, self._next_id, timestamp)
            self._next_id += 1
            self._cond.notify_all()

    def wait_newer(self, last_id: int, timeout: float) -> Frame | None:
        with self._cond:
            self._cond.wait_for(
                lambda: self._frame is not None and self._frame.frame_id > last_id, timeout
            )
            frame = self._frame
        return frame if frame is not None and frame.frame_id > last_id else None


def pick_backend(name: str) -> str:
    candidates = ["dxcam", "bettercam", "mss"] if name == "auto" else [name]
    for candidate in candidates:
        try:
            importlib.import_module(candidate)
            return candidate
        except ImportError:
            continue
    raise RuntimeError(f"no capture backend available (tried {candidates})")


def _on_primary_monitor(region: tuple[int, int, int, int]) -> bool:
    size = win32.primary_screen_size()
    if size is None:
        return True
    left, top, w, h = region
    return left >= 0 and top >= 0 and left + w <= size[0] and top + h <= size[1]


class ScreenCapture(threading.Thread):
    def __init__(self, region: tuple[int, int, int, int], cfg: CaptureConfig,
                 buffer: LatestFrameBuffer, hud_rects: dict[str, tuple] | None = None,
                 hud_interval_s: float = 0.5) -> None:
        super().__init__(daemon=True, name="capture")
        self.region = region            # left, top, width, height (screen pixels)
        self.cfg = cfg
        self.buffer = buffer
        self.hud_rects = hud_rects or {}
        self.hud_interval_s = hud_interval_s
        self._last_hud = 0.0
        self.backend = pick_backend(cfg.backend)
        if self.backend in ("dxcam", "bettercam") and not _on_primary_monitor(region):
            # dxcam captures the primary output only; mss handles the whole virtual desktop
            print("[capture] game is not on the primary monitor -> using mss "
                  "(move it to the main screen for dxcam speed)")
            self.backend = pick_backend("mss")
        self.error: BaseException | None = None
        self._stop_event = threading.Event()
        w, h = region[2], region[3]
        self.out_size = (max(1, int(w * cfg.scale)), max(1, int(h * cfg.scale)))

    def stop(self) -> None:
        self._stop_event.set()

    def _publish(self, bgr: np.ndarray, ts: float) -> None:
        if self.hud_rects and ts - self._last_hud >= self.hud_interval_s:
            # small text needs full resolution, so crop before downscaling
            h, w = bgr.shape[:2]
            crops = {name: bgr[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)].copy()
                     for name, (x0, y0, x1, y1) in self.hud_rects.items()}
            self.buffer.publish_hud(crops, ts)
            self._last_hud = ts
        if self.cfg.scale != 1.0:
            # INTER_AREA averages pixels: small food stays visible after downscaling
            bgr = cv2.resize(bgr, self.out_size, interpolation=cv2.INTER_AREA)
        self.buffer.publish(bgr, ts)

    def run(self) -> None:
        try:
            if self.backend in ("dxcam", "bettercam"):
                self._run_dxgi()
            else:
                self._run_mss()
        except BaseException as exc:  # surfaced by the main loop
            self.error = exc

    def _run_dxgi(self) -> None:
        lib = importlib.import_module(self.backend)
        left, top, w, h = self.region
        # dxcam regions are relative to the output (primary monitor)
        camera = lib.create(output_color="BGR")
        camera.start(target_fps=self.cfg.target_fps, region=(left, top, left + w, top + h),
                     video_mode=True)
        try:
            while not self._stop_event.is_set():
                img = camera.get_latest_frame()   # blocks until the next frame
                if img is None:
                    continue
                self._publish(img, time.perf_counter())
        finally:
            camera.stop()

    def _run_mss(self) -> None:
        import mss

        left, top, w, h = self.region
        monitor = {"left": left, "top": top, "width": w, "height": h}
        period = 1.0 / self.cfg.target_fps
        with mss.mss() as sct:           # mss instances are not thread safe: create here
            while not self._stop_event.is_set():
                t0 = time.perf_counter()
                bgra = np.asarray(sct.grab(monitor))
                self._publish(cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR), t0)
                rest = period - (time.perf_counter() - t0)
                if rest > 0:
                    time.sleep(rest)
