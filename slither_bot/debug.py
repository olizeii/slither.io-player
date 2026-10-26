"""Debug overlay: shows what the bot sees and why it steers where it does."""
from __future__ import annotations

import math

import cv2
import numpy as np

from .brain import Action
from .vision import GameState

WINDOW = "bot-debug"   # must not contain "slither.io", or window detection finds it


def render(frame: np.ndarray, state: GameState, action: Action | None = None,
           info: str = "") -> np.ndarray:
    img = frame.copy()
    cx, cy = state.center
    obstacle = state.debug.get("obstacle")
    if obstacle is not None:                       # red tint on dangerous pixels
        m = obstacle > 0
        img[m] = (0.4 * img[m] + 0.6 * np.array([40, 40, 255])).astype(np.uint8)

    for x, y, wgt in state.food:
        cv2.circle(img, (int(x), int(y)), 2 if wgt < 100 else 5, (80, 255, 80), -1)

    if action is not None and action.debug:
        dbg = action.debug
        clear, danger = dbg.get("clear"), dbg.get("danger")
        if clear is not None:
            n = len(clear)
            for i in range(n):
                a = -math.pi + (i + 0.5) * 2 * math.pi / n
                r = float(min(clear[i], 400))
                d = float(min(danger[i], 1.0))
                color = (0, int(255 * (1 - d)), int(255 * d))
                cv2.line(img, (int(cx), int(cy)),
                         (int(cx + r * math.cos(a)), int(cy + r * math.sin(a))), color, 1)
        for tr in dbg.get("tracks", []):
            p = (int(cx + tr.x), int(cy + tr.y))
            cv2.circle(img, p, int(max(tr.radius, 4) * 2), (0, 255, 255), 2)
            cv2.arrowedLine(img, p, (int(p[0] + tr.vx * 0.3), int(p[1] + tr.vy * 0.3)), (0, 255, 255), 2)
        for z in dbg.get("zones", []):           # loot zones: magenta = confirmed, grey = pending
            color = (255, 0, 255) if z.confirmed else (160, 160, 160)
            cv2.circle(img, (int(cx + z.x), int(cy + z.y)), 40, color, 2)
            cv2.putText(img, f"{z.source} {z.mass:.0f}", (int(cx + z.x) - 30, int(cy + z.y) - 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        if state.crowd_angle is not None:          # minimap crowd direction: orange arrow
            a = state.crowd_angle
            cv2.arrowedLine(img, (int(cx), int(cy)), (int(cx + 90 * math.cos(a)), int(cy + 90 * math.sin(a))),
                            (0, 165, 255), 2, tipLength=0.25)
        h = dbg.get("heading")
        if h is not None:
            cv2.line(img, (int(cx), int(cy)),
                     (int(cx + 60 * math.cos(h)), int(cy + 60 * math.sin(h))), (255, 120, 0), 2)
        cv2.line(img, (int(cx), int(cy)),
                 (int(cx + 120 * math.cos(action.angle)), int(cy + 120 * math.sin(action.angle))),
                 (255, 255, 255), 3)

    if state.border_angle is not None:
        info += f" border={state.border_proximity:.2f}"
    lines = [info,
             f"mode={action.mode.value if action else '-'} boost={action.boost if action else '-'} "
             f"alive={state.alive} r={state.self_radius:.1f} area={state.self_area} "
             f"food={len(state.food)} heads={len(state.heads)}"]
    for i, text in enumerate(lines):
        cv2.putText(img, text, (8, 18 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img
