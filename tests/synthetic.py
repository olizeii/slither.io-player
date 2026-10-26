"""Synthetic slither.io-like frames for offline tests (analysis resolution)."""
from __future__ import annotations

import cv2
import numpy as np

W, H = 960, 540
CX, CY = W // 2, H // 2


def background(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.empty((H, W, 3), np.uint8)
    img[:] = (40, 30, 25)                          # dark blue-grey, like the hex grid
    noise = rng.integers(-8, 9, size=(H, W, 1), dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def draw_snake(img, points, radius, color, eyes_dir=None) -> None:
    for x, y in points:
        cv2.circle(img, (int(x), int(y)), radius, color, -1)
    if eyes_dir is not None:
        hx, hy = points[0]
        dx, dy = eyes_dir
        px, py = -dy, dx                          # perpendicular
        for s in (-1, 1):
            ex, ey = hx + 0.4 * radius * dx + s * 0.5 * radius * px, hy + 0.4 * radius * dy + s * 0.5 * radius * py
            cv2.circle(img, (int(ex), int(ey)), 3, (255, 255, 255), -1)
            cv2.circle(img, (int(ex), int(ey)), 1, (0, 0, 0), -1)


def draw_food(img, center, n, spread, seed=1) -> None:
    rng = np.random.default_rng(seed)
    colors = [(0, 0, 255), (0, 255, 255), (255, 0, 255), (255, 128, 0), (0, 255, 0)]
    for i in range(n):
        x = center[0] + rng.uniform(-spread, spread)
        y = center[1] + rng.uniform(-spread, spread)
        cv2.circle(img, (int(x), int(y)), 3, colors[i % len(colors)], -1)


def scene(enemy_head=(600, 270), food_center=(480, 130), self_dir=(1, 0)) -> np.ndarray:
    """Own green snake at the centre, a magenta enemy, a food cluster."""
    img = background()
    sx, sy = self_dir
    own = [(CX - sx * k * 4, CY - sy * k * 4) for k in range(45)]
    draw_snake(img, own, 9, (60, 200, 60), eyes_dir=self_dir)
    if enemy_head is not None:
        ex, ey = enemy_head
        enemy = [(ex + k * 3, ey - k * 2) for k in range(40)]   # body trails up-right
        draw_snake(img, enemy, 9, (200, 60, 200), eyes_dir=(-0.83, 0.55))
    if food_center is not None:
        draw_food(img, food_center, 15, 45)
    return img
