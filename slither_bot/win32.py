"""Thin ctypes wrappers around the Win32 API (no pywin32 dependency).

Everything here is a no-op or raises on non-Windows platforms; the rest of
the bot falls back to ``pynput`` there.
"""
from __future__ import annotations

import ctypes
import re
import sys

IS_WINDOWS = sys.platform == "win32"

VK_RETURN = 0x0D
VK_F8 = 0x77
VK_F9 = 0x78
VK_F12 = 0x7B

INPUT_MOUSE = 0
INPUT_KEYBOARD = 1
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
KEYEVENTF_KEYUP = 0x0002

if IS_WINDOWS:
    from ctypes import wintypes

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    class MOUSEINPUT(ctypes.Structure):
        _fields_ = [
            ("dx", wintypes.LONG),
            ("dy", wintypes.LONG),
            ("mouseData", wintypes.DWORD),
            ("dwFlags", wintypes.DWORD),
            ("time", wintypes.DWORD),
            ("dwExtraInfo", ctypes.c_size_t),
        ]

    class KEYBDINPUT(ctypes.Structure):
        _fields_ = [
            ("wVk", wintypes.WORD),
            ("wScan", wintypes.WORD),
            ("dwFlags", wintypes.DWORD),
            ("time", wintypes.DWORD),
            ("dwExtraInfo", ctypes.c_size_t),
        ]

    class _INPUTUNION(ctypes.Union):
        _fields_ = [("mi", MOUSEINPUT), ("ki", KEYBDINPUT)]

    class INPUT(ctypes.Structure):
        _fields_ = [("type", wintypes.DWORD), ("u", _INPUTUNION)]

    WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    user32.SendInput.argtypes = (wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int)
    user32.SendInput.restype = wintypes.UINT
    user32.SetCursorPos.argtypes = (ctypes.c_int, ctypes.c_int)
    user32.GetAsyncKeyState.argtypes = (ctypes.c_int,)
    user32.GetAsyncKeyState.restype = ctypes.c_short
    user32.GetForegroundWindow.restype = wintypes.HWND
    kernel32.GetConsoleWindow.restype = wintypes.HWND


def _require_windows() -> None:
    if not IS_WINDOWS:
        raise OSError("this function needs Windows")


def enable_dpi_awareness() -> None:
    """Make capture coordinates and cursor coordinates both physical pixels.

    Without this, a 125 % / 150 % display scale makes SetCursorPos and the
    screen grab disagree and the bot steers in slightly wrong directions.
    """
    if not IS_WINDOWS:
        return
    try:  # Windows 10 1703+: PER_MONITOR_AWARE_V2
        if user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4)):
            return
    except AttributeError:
        pass
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except (AttributeError, OSError):
        user32.SetProcessDPIAware()


def set_timer_resolution(ms: int = 1) -> None:
    """Raise the scheduler tick so short sleeps in the mouse thread are accurate."""
    if IS_WINDOWS:
        ctypes.windll.winmm.timeBeginPeriod(ms)


def reset_timer_resolution(ms: int = 1) -> None:
    if IS_WINDOWS:
        ctypes.windll.winmm.timeEndPeriod(ms)


def find_window(title_regex: str) -> int | None:
    """Return the first visible top-level window whose title matches."""
    _require_windows()
    pattern = re.compile(title_regex, re.IGNORECASE)
    own_console = kernel32.GetConsoleWindow()
    found: list[int] = []

    def callback(hwnd, _lparam):
        if hwnd == own_console or not user32.IsWindowVisible(hwnd):
            return True
        length = user32.GetWindowTextLengthW(hwnd)
        if length == 0:
            return True
        buf = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buf, length + 1)
        if pattern.search(buf.value):
            found.append(hwnd)
            return False
        return True

    user32.EnumWindows(WNDENUMPROC(callback), 0)
    return found[0] if found else None


def client_rect_on_screen(hwnd: int) -> tuple[int, int, int, int]:
    """(left, top, width, height) of the window's client area in screen pixels."""
    _require_windows()
    rect = wintypes.RECT()
    user32.GetClientRect(hwnd, ctypes.byref(rect))
    origin = wintypes.POINT(0, 0)
    user32.ClientToScreen(hwnd, ctypes.byref(origin))
    return origin.x, origin.y, rect.right - rect.left, rect.bottom - rect.top


def foreground_window() -> int | None:
    return user32.GetForegroundWindow() if IS_WINDOWS else None


def set_cursor_pos(x: float, y: float) -> None:
    user32.SetCursorPos(int(round(x)), int(round(y)))


def _send(*inputs: "INPUT") -> None:
    arr = (INPUT * len(inputs))(*inputs)
    user32.SendInput(len(inputs), arr, ctypes.sizeof(INPUT))


def left_button(down: bool) -> None:
    flags = MOUSEEVENTF_LEFTDOWN if down else MOUSEEVENTF_LEFTUP
    _send(INPUT(type=INPUT_MOUSE, u=_INPUTUNION(mi=MOUSEINPUT(0, 0, 0, flags, 0, 0))))


def key_tap(vk: int) -> None:
    down = INPUT(type=INPUT_KEYBOARD, u=_INPUTUNION(ki=KEYBDINPUT(vk, 0, 0, 0, 0)))
    up = INPUT(type=INPUT_KEYBOARD, u=_INPUTUNION(ki=KEYBDINPUT(vk, 0, KEYEVENTF_KEYUP, 0, 0)))
    _send(down, up)


def is_key_down(vk: int) -> bool:
    return bool(user32.GetAsyncKeyState(vk) & 0x8000) if IS_WINDOWS else False


class Hotkeys:
    """Edge-triggered global hotkeys by polling GetAsyncKeyState (~1 us per key)."""

    def __init__(self, bindings: dict[str, int]):
        self._bindings = bindings
        self._state = {name: False for name in bindings}

    def poll(self) -> set[str]:
        pressed = set()
        for name, vk in self._bindings.items():
            down = is_key_down(vk)
            if down and not self._state[name]:
                pressed.add(name)
            self._state[name] = down
        return pressed
