"""Minimap analysis on the full-resolution crop (bottom right, 136x136 px at 1080p).

What slither.io draws there (read from the page's canvases):
  * population map: white pixels at alpha ~0.42 wherever snakes are, so they
    show up mid-grey (V ~100-135, low S) over the dark background
  * own position: a separate 14x14 marker, pure white (255) with a dark rim

From that we get our position in the circular world (-> border distance) and
the direction of the most attractive crowded area: dense population, weighted
by how close it is. Crowds mean kills and loot pits, so the brain is biased
towards them when nothing better is in view.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .config import VisionConfig


@dataclass
class MinimapInfo:
    timestamp: float
    pos: tuple[float, float]          # own position, -1..1 in map radii
    border_proximity: float
    border_angle: float
    crowd_angle: float | None = None
    crowd_dist: float = 0.0           # map radii from us to the crowd peak
    crowd_mass: float = 0.0           # population pixels around the peak


def analyze_minimap(crop: np.ndarray, cfg: VisionConfig, timestamp: float) -> MinimapInfo | None:
    if crop is None or crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h, w = hsv.shape[:2]
    s, v = hsv[:, :, 1], hsv[:, :, 2]
    mcx, mcy = w / 2.0, h / 2.0
    radius = min(w, h) / 2.0 * cfg.minimap_radius_frac
    yy, xx = np.mgrid[0:h, 0:w]
    inside = np.hypot(xx - mcx, yy - mcy) <= radius

    dot = (v >= cfg.minimap_self_min_v) & (s <= cfg.minimap_self_max_s) & inside
    if not dot.any():
        return None
    # the marker is a solid blob; the brightest blob's centroid is our position
    n, _, stats, cents = cv2.connectedComponentsWithStats(dot.astype(np.uint8), connectivity=8)
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    px, py = cents[k]
    dx, dy = (px - mcx) / radius, (py - mcy) / radius
    info = MinimapInfo(timestamp, (dx, dy), float(min(math.hypot(dx, dy), 1.5)), math.atan2(dy, dx))

    pop = ((v >= cfg.crowd_min_v) & (v < cfg.minimap_self_min_v) & (s <= cfg.crowd_max_s) & inside)
    near_self = np.hypot(xx - px, yy - py) < cfg.crowd_self_exclusion * radius   # that is us
    pop &= ~near_self
    if np.count_nonzero(pop) < cfg.crowd_min_pixels:
        return info
    sigma = cfg.crowd_sigma * radius
    density = cv2.GaussianBlur(pop.astype(np.float32), (0, 0), sigma)
    dist = np.hypot(xx - px, yy - py) / radius
    attract = density * np.exp(-dist / cfg.crowd_decay) * inside
    iy, ix = np.unravel_index(int(np.argmax(attract)), attract.shape)
    info.crowd_angle = math.atan2(iy - py, ix - px)
    info.crowd_dist = float(dist[iy, ix])
    info.crowd_mass = float(density[iy, ix] * 2 * math.pi * sigma ** 2)   # ~pixels in the blob
    return info


def apply_minimap(state, info: MinimapInfo | None, now: float, max_age_s: float = 2.0) -> None:
    """Copy fresh minimap results into the GameState (overrides the low-res reading)."""
    if info is None or now - info.timestamp > max_age_s:
        return
    state.border_proximity = info.border_proximity
    state.border_angle = info.border_angle
    state.crowd_angle = info.crowd_angle
    state.crowd_dist = info.crowd_dist
    state.crowd_mass = info.crowd_mass
