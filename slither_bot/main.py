"""Entry point: wires capture -> vision -> brain -> controller together.

Threads:
  capture  grab + downscale, publishes the newest frame (drops stale ones)
  hud-ocr  reads length / rank / final length with Windows OCR (2 Hz)
  main     vision + brain + hotkeys + training bookkeeping + debug window
  mouse    rotates the cursor towards the target angle at 240 Hz

Hotkeys (global): F8 start/pause, F9 debug window on/off, F12 quit.

    python -m slither_bot.main --debug             # play with the default parameters
    python -m slither_bot.main --train             # play aggressively and self-tune forever
    python -m slither_bot.main --use-best          # play with the best parameters found so far
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

from . import debug as debug_view
from . import win32
from .brain import Brain, Mode
from .capture import LatestFrameBuffer, ScreenCapture
from .config import AGGRESSIVE, PROFILE_VERSION, BotConfig, apply_brain_params, load_config
from .controller import MouseController
from .episode import EpisodeTracker
from .hud import HudReader
from .minimap import analyze_minimap, apply_minimap
from .tuner import SPACE, Tuner, style, style_line
from .vision import GameState, Vision


def short_style(params: dict) -> str:
    st = style(params)
    return f"Aggr {st['Aggression']}% Vors {st['Vorsicht']}% Gier {st['Gier']}% Loot {st['Loot']}%"


def locate_game(cfg: BotConfig) -> tuple[int | None, tuple[int, int, int, int]]:
    """Find the browser window running slither.io and return (hwnd, region)."""
    if cfg.capture.region:
        left, top, w, h = cfg.capture.region
        return None, (int(left), int(top), int(w), int(h))
    if not win32.IS_WINDOWS:
        raise SystemExit("window detection needs Windows - pass --region L,T,W,H")
    hwnd = win32.find_window(cfg.capture.window_title_regex)
    if hwnd is None:
        browsers = ("chrome", "edge", "firefox", "opera", "brave", "vivaldi")
        seen = [t for t in win32.visible_window_titles() if any(b in t.lower() for b in browsers)]
        raise SystemExit("no slither.io browser window found. The slither.io tab must be the "
                         "active tab of its window (title starts with 'slither.io'); press F11 "
                         "for fullscreen, or pass --region L,T,W,H.\n  browser windows seen: "
                         + (", ".join(repr(t) for t in seen) or "none"))
    left, top, w, h = win32.client_rect_on_screen(hwnd)
    crop = cfg.capture.crop_top_px
    return hwnd, (left, top + crop, w, h - crop)


class Respawner:
    """Clicks the green Play / Play Again button once the menu is visible."""

    FALLBACK_AFTER_S = 8.0       # no button detected but dead this long -> click the default spot

    def __init__(self, cfg: BotConfig, region, controller: MouseController) -> None:
        self.cfg = cfg.runtime
        self.region = region
        self.controller = controller
        self.menu_since: float | None = None
        self.dead_since: float | None = None
        self.last_try = 0.0

    def update(self, state: GameState, now: float) -> None:
        left, top, w, h = self.region
        if state.alive:
            self.menu_since = self.dead_since = None
            return
        self.dead_since = self.dead_since or now
        if now - self.last_try < self.cfg.respawn_retry_s:
            return
        if state.menu_button is not None:
            self.menu_since = self.menu_since or now
            if now - self.menu_since < self.cfg.respawn_delay_s:
                return
            bx, by = state.menu_button
            x, y = left + bx * w / state.width, top + by * h / state.height
        elif now - self.dead_since > self.FALLBACK_AFTER_S:
            rx, ry = self.cfg.respawn_click_rel
            x, y = left + rx * w, top + ry * h
        else:
            return
        self.last_try = now
        print(f"[respawn] clicking play at ({x:.0f},{y:.0f})")
        self.controller.click_at(x, y)
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
        self.modes: dict[str, int] = {}
        self.style = ""              # short play-style tag, set by run()

    def tick(self, work_s: float, latency_s: float, mode: Mode, extra: str = "") -> None:
        self.frames += 1
        self.work += work_s
        self.latency += latency_s
        if mode is not Mode.DEAD:
            self.modes[mode.value] = self.modes.get(mode.value, 0) + 1
        now = time.perf_counter()
        if now - self.t0 >= self.interval:
            self.fps = self.frames / (now - self.t0)
            self.text = (f"fps={self.fps:4.0f} {1000 * self.work / self.frames:4.1f}ms "
                         f"lat={1000 * self.latency / self.frames:4.1f}ms")
            total = sum(self.modes.values())
            doing = " ".join(f"{k} {100 * v / total:.0f}%" for k, v in
                             sorted(self.modes.items(), key=lambda kv: -kv[1])[:3]) if total else "-"
            print(f"[stats] {self.text} | {extra} | gerade: {doing} | {self.style}")
            self.t0, self.frames, self.work, self.latency, self.modes = now, 0, 0.0, 0.0, {}


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
    ap.add_argument("--aggressive", action="store_true", help="start from the aggressive profile")
    ap.add_argument("--train", action="store_true",
                    help="self-tuning: aggressive start, auto respawn, learns between rounds")
    ap.add_argument("--use-best", action="store_true", help="play with tuning/best_params.json")
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
    cfg.runtime.auto_respawn |= args.auto_respawn or args.train
    if args.no_foreground_check:
        cfg.runtime.foreground_check = False
    if args.start:
        cfg.runtime.start_paused = False
    if args.aggressive or args.train:
        apply_brain_params(cfg, AGGRESSIVE)
    if args.use_best:
        best = Path(cfg.train.state_dir) / "best_params.json"
        if best.exists():
            apply_brain_params(cfg, json.loads(best.read_text(encoding="utf-8")))
            print(f"[init] using {best}")
        else:
            print(f"[init] {best} not found yet - run --train first")
    return cfg


def run(cfg: BotConfig, train: bool = False) -> None:
    win32.enable_dpi_awareness()
    win32.set_timer_resolution(1)
    hwnd, region = locate_game(cfg)
    left, top, w, h = region
    print(f"[init] game region {w}x{h} at ({left},{top})")

    buffer = LatestFrameBuffer()
    # full-resolution crops for OCR and the minimap population map
    hud_rects = {"minimap": cfg.vision.minimap_rect}
    if cfg.hud.enabled:
        hud_rects.update(stats=cfg.hud.stats_rect, final=cfg.hud.final_rect)
    capture = ScreenCapture(region, cfg.capture, buffer, hud_rects, cfg.hud.interval_s)
    print(f"[init] capture backend: {capture.backend}, analysis size {capture.out_size}")
    hud = HudReader(buffer, cfg.hud)
    print("[init] HUD OCR: " + ("on" if hud.available else f"off ({hud.error})"))

    tuner = tracker = None
    role = "-"
    if train:
        tuner = Tuner(cfg.train, {p.name: getattr(cfg.brain, p.name) for p in SPACE},
                      version=PROFILE_VERSION)
        if tuner.notice:
            print(tuner.notice)
        role, params = tuner.next_params()
        apply_brain_params(cfg, params)
        tracker = EpisodeTracker(hud, cfg.train.final_length_wait_s)
        print(f"[train] state in {tuner.state_path} - {tuner.state['episodes']} rounds so far, "
              f"best length {tuner.state['best_length']}, next round plays the {role}")
        print(f"        {tuner.trend_line()}")
    current = {p.name: getattr(cfg.brain, p.name) for p in SPACE}
    print(f"[init] {style_line(current)}")

    vision = Vision(cfg.vision)
    brain = Brain(cfg.brain)
    controller = MouseController((left + w / 2.0, top + h / 2.0), cfg.control)
    respawner = Respawner(cfg, region, controller)
    hotkeys = win32.Hotkeys({"pause": win32.VK_F8, "debug": win32.VK_F9, "quit": win32.VK_F12})
    stats = Stats(cfg.runtime.stats_interval_s)
    stats.style = short_style(current)

    paused = cfg.runtime.start_paused
    show_debug = cfg.runtime.debug
    print("[init] F8 = start/pause, F9 = debug window, F12 = quit"
          + ("  (paused - focus the game and press F8)" if paused else ""))

    capture.start()
    hud.start()
    controller.start()
    last_id = 0
    frame_no = 0
    minimap_info = None
    minimap_ts = 0.0
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
            crops = buffer.latest_hud()
            if crops is not None and crops[0] != minimap_ts:     # new minimap crop (2 Hz)
                minimap_ts = crops[0]
                minimap_info = analyze_minimap(crops[1].get("minimap"), cfg.vision, minimap_ts)
            apply_minimap(state, minimap_info, t0)
            action = brain.decide(state)
            t1 = time.perf_counter()

            focused = (not cfg.runtime.foreground_check or hwnd is None
                       or win32.foreground_window() == hwnd)
            running = not paused and focused
            controller.command(action.angle, action.boost)
            controller.set_active(running and state.alive)

            if tracker is not None and running:
                episode = tracker.update(state, action, t1)
                if episode is not None:
                    print(tuner.report(episode))
                    role, params = tuner.next_params()
                    apply_brain_params(cfg, params)
                    brain = Brain(cfg.brain)          # fresh brain with the next parameter set
                    stats.style = short_style(params)
                    print(f"[train] next round: {tuner.describe(role)}")
            # never click "Play Again" before the final length has been read
            if cfg.runtime.auto_respawn and running and (tracker is None or tracker.phase != "ending"):
                respawner.update(state, t1)

            hv = hud.snapshot()
            extra = f"Länge {hv.length} Rang {hv.rank}/{hv.players}" if hud.available else "kein OCR"
            stats.tick(t1 - t0, t1 - frame.timestamp, action.mode, extra)
            if show_debug and frame_no % cfg.runtime.debug_every == 0:
                info = stats.text + (" PAUSED" if paused else "") + (f" [{role}]" if train else "")
                cv2.imshow(debug_view.WINDOW, debug_view.render(frame.image, state, action, info))
                cv2.waitKey(1)
    except KeyboardInterrupt:
        pass
    finally:
        controller.stop()
        hud.stop()
        capture.stop()
        cv2.destroyAllWindows()
        win32.reset_timer_resolution(1)
        print("[exit] bye")


def main(argv=None) -> int:
    args = parse_args(argv)
    run(build_config(args), train=args.train)
    return 0


if __name__ == "__main__":
    sys.exit(main())
