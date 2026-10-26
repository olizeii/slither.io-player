"""Mouse output running in its own high-rate thread.

slither.io steers towards the cursor position relative to the screen centre,
so only the *angle* matters. The brain sets a target angle at frame rate
(30-60 Hz); this thread rotates the cursor towards it at ``rate_hz`` with a
bounded angular speed, which gives smooth, continuous steering.

On Windows the cursor is placed with SetCursorPos and the boost button is sent
via SendInput (both ~microseconds). Elsewhere pynput is used.
"""
from __future__ import annotations

import math
import threading
import time

from . import win32
from .config import ControlConfig


class _Win32Input:
    def move(self, x: float, y: float) -> None:
        win32.set_cursor_pos(x, y)

    def button(self, down: bool) -> None:
        win32.left_button(down)

    def enter(self) -> None:
        win32.key_tap(win32.VK_RETURN)


class _PynputInput:
    def __init__(self) -> None:
        from pynput import keyboard, mouse

        self._mouse = mouse.Controller()
        self._kbd = keyboard.Controller()
        self._btn = mouse.Button.left
        self._enter = keyboard.Key.enter

    def move(self, x: float, y: float) -> None:
        self._mouse.position = (int(x), int(y))

    def button(self, down: bool) -> None:
        (self._mouse.press if down else self._mouse.release)(self._btn)

    def enter(self) -> None:
        self._kbd.press(self._enter)
        self._kbd.release(self._enter)


def make_input():
    return _Win32Input() if win32.IS_WINDOWS else _PynputInput()


class MouseController(threading.Thread):
    def __init__(self, center: tuple[float, float], cfg: ControlConfig, backend=None) -> None:
        super().__init__(daemon=True, name="mouse")
        self.center = center              # screen pixels
        self.cfg = cfg
        self.input = backend or make_input()
        self._lock = threading.Lock()
        self._target = 0.0
        self._want_boost = False
        self._active = False
        self._angle = 0.0
        self._boost_down = False
        self._stop_event = threading.Event()

    def command(self, angle: float, boost: bool) -> None:
        with self._lock:
            self._target = angle
            self._want_boost = boost

    def set_active(self, active: bool) -> None:
        with self._lock:
            self._active = active

    def stop(self) -> None:
        self._stop_event.set()
        self.join(timeout=1.0)

    def click_at(self, x: float, y: float) -> None:
        """Used for respawning; only call while the controller is inactive."""
        self.input.move(x, y)
        time.sleep(0.03)
        self.input.button(True)
        time.sleep(0.03)
        self.input.button(False)

    def run(self) -> None:
        period = 1.0 / self.cfg.rate_hz
        last = time.perf_counter()
        try:
            while not self._stop_event.is_set():
                now = time.perf_counter()
                dt, last = now - last, now
                with self._lock:
                    target, boost, active = self._target, self._want_boost, self._active
                if active:
                    diff = (target - self._angle + math.pi) % (2 * math.pi) - math.pi
                    step = self.cfg.max_angular_speed * dt
                    self._angle += max(-step, min(step, diff))
                    r = self.cfg.cursor_radius_px
                    self.input.move(self.center[0] + r * math.cos(self._angle),
                                    self.center[1] + r * math.sin(self._angle))
                else:
                    boost = False
                if boost != self._boost_down:      # only send button edges
                    self.input.button(boost)
                    self._boost_down = boost
                rest = period - (time.perf_counter() - now)
                if rest > 0:
                    time.sleep(rest)
        finally:
            if self._boost_down:
                self.input.button(False)
