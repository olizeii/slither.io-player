"""Entry point: wires capture -> vision -> brain -> controller together.

Threads:
  capture  grab + downscale, publishes the newest frame (drops stale ones)
  main     vision + brain + hotkeys + debug window  (~8-12 ms per frame)
  mouse    rotates the cursor towards the target angle at 240 Hz

Hotkeys (global): F8 start/pause, F9 debug window on/off, F12 quit.

    python -m slither_bot.main --debug
"""
from __future__ import annotations

import argparse
import sys
import time

import cv2

from . import debug as debug_view
from . import win32
from .brain import Brain, Mode
from .capture import LatestFrameBuffer, ScreenCapture
from .config import BotConfig, load_config
from .controller import MouseController
from .vision import Vision


def locate_game(cfg: BotConfig) -> tuple[int | None, tuple[int, int, int, int]]:
    """Find the browser window running slither.io and return (hwnd, region)."""
    if cfg.capture.region:
        left, top, w, h = cfg.capture.region
        return None, (int(left), int(top), int(w), int(h))
    if not win32.IS_WINDOWS:
        raise SystemExit("window detection needs Windows - pass --region L,T,W,H")
    hwnd = win32.find_window(cfg.capture.window_title_regex)
    if hwnd is None:
        raise SystemExit("no slither.io browser window found. Open http://slither.io, "
                         "press F11 for fullscreen, or pass --region L,T,W,H")
    left, top, w, h = win32.client_rect_on_screen(hwnd)
    crop = cfg.capture.crop_top_px
    return hwnd, (left, top + crop, w, h - crop)


class Respawner:
    def __init__(self, cfg: BotConfig, region, controller: MouseController) -> None:
        self.cfg = cfg.runtime
        self.region = region
        self.controller = controller
        self.dead_since: float | None = None
        self.last_try = 0.0

    def update(self, alive: bool, now: float) -> None:
        if alive:
            self.dead_since = None
            return
        if self.dead_since is None:
            self.dead_since = now
            return
        if now - self.dead_since < self.cfg.respawn_delay_s or now - self.last_try < self.cfg.respawn_retry_s:
            return
        self.last_try = now
        left, top, w, h = self.region
        rx, ry = self.cfg.respawn_click_rel
        print("[respawn] clicking play")
        self.controller.click_at(left + rx * w, top + ry * h)
        if self.cfg.respawn_press_enter:
            time.sleep(0.05)
            self.controller.input.enter()


class Stats:
    def __init__(self, interval: float) -> None:
        self.interval = interval
        self.t0 = time.perf_counter()
        self.frames = 0
        self.work = 0.0
        self.latency = 0.0
        self.fps = 0.0
        self.text = ""

    def tick(self, work_s: float, latency_s: float, mode: Mode) -> None:
        self.frames += 1
        self.work += work_s
        self.latency += latency_s
        now = time.perf_counter()
        if now - self.t0 >= self.interval:
            self.fps = self.frames / (now - self.t0)
            self.text = (f"fps={self.fps:5.1f} vision+brain={1000 * self.work / self.frames:5.1f}ms "
                         f"latency={1000 * self.latency / self.frames:5.1f}ms")
            print(f"[stats] {self.text} mode={mode.value}")
            self.t0, self.frames, self.work, self.latency = now, 0, 0.0, 0.0


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="slither.io computer-vision bot")
    ap.add_argument("--config", default="config.json", help="JSON overrides (optional)")
    ap.add_argument("--region", help="capture region L,T,W,H in screen pixels")
    ap.add_argument("--backend", choices=["auto", "dxcam", "bettercam", "mss"])
    ap.add_argument("--fps", type=int)
    ap.add_argument("--debug", action="store_true", help="show the debug overlay window")
    ap.add_argument("--auto-respawn", action="store_true")
    ap.add_argument("--no-foreground-check", action="store_true")
    ap.add_argument("--start", action="store_true", help="start immediately instead of paused")
    return ap.parse_args(argv)


def build_config(args: argparse.Namespace) -> BotConfig:
    cfg = load_config(args.config)
    if args.region:
        cfg.capture.region = [int(v) for v in args.region.split(",")]
    if args.backend:
        cfg.capture.backend = args.backend
    if args.fps:
        cfg.capture.target_fps = args.fps
    cfg.runtime.debug |= args.debug
    cfg.runtime.auto_respawn |= args.auto_respawn
    if args.no_foreground_check:
        cfg.runtime.foreground_check = False
    if args.start:
        cfg.runtime.start_paused = False
    return cfg


def run(cfg: BotConfig) -> None:
    win32.enable_dpi_awareness()
    win32.set_timer_resolution(1)
    hwnd, region = locate_game(cfg)
    left, top, w, h = region
    print(f"[init] game region {w}x{h} at ({left},{top})")

    buffer = LatestFrameBuffer()
    capture = ScreenCapture(region, cfg.capture, buffer)
    print(f"[init] capture backend: {capture.backend}, analysis size {capture.out_size}")
    vision = Vision(cfg.vision)
    brain = Brain(cfg.brain)
    controller = MouseController((left + w / 2.0, top + h / 2.0), cfg.control)
    respawner = Respawner(cfg, region, controller)
    hotkeys = win32.Hotkeys({"pause": win32.VK_F8, "debug": win32.VK_F9, "quit": win32.VK_F12})
    stats = Stats(cfg.runtime.stats_interval_s)

    paused = cfg.runtime.start_paused
    show_debug = cfg.runtime.debug
    print("[init] F8 = start/pause, F9 = debug window, F12 = quit"
          + ("  (paused - focus the game and press F8)" if paused else ""))

    capture.start()
    controller.start()
    last_id = 0
    frame_no = 0
    try:
        while True:
            frame = buffer.wait_newer(last_id, timeout=0.5)
            keys = hotkeys.poll()
            if "quit" in keys:
                break
            if "pause" in keys:
                paused = not paused
                print("[run] paused" if paused else "[run] running")
            if "debug" in keys:
                show_debug = not show_debug
                if not show_debug:
                    cv2.destroyAllWindows()
            if capture.error is not None:
                raise RuntimeError("capture thread died") from capture.error
            if frame is None:
                continue
            last_id = frame.frame_id
            frame_no += 1

            t0 = time.perf_counter()
            state = vision.analyze(frame.image, frame.timestamp)
            action = brain.decide(state)
            t1 = time.perf_counter()

            focused = (not cfg.runtime.foreground_check or hwnd is None
                       or win32.foreground_window() == hwnd)
            controller.command(action.angle, action.boost)
            controller.set_active(not paused and focused and state.alive)
            if cfg.runtime.auto_respawn and not paused and focused:
                respawner.update(state.alive, t1)

            stats.tick(t1 - t0, t1 - frame.timestamp, action.mode)
            if show_debug and frame_no % cfg.runtime.debug_every == 0:
                info = stats.text + (" PAUSED" if paused else "")
                cv2.imshow(debug_view.WINDOW, debug_view.render(frame.image, state, action, info))
                cv2.waitKey(1)
    except KeyboardInterrupt:
        pass
    finally:
        controller.stop()
        capture.stop()
        cv2.destroyAllWindows()
        win32.reset_timer_resolution(1)
        print("[exit] bye")


def main(argv=None) -> int:
    run(build_config(parse_args(argv)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
