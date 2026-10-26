"""Computer-vision layer: turns one BGR frame into a compact ``GameState``.

Pipeline (all on the downscaled analysis frame, ~6-10 ms at 960x540):

  1. BGR -> HSV, HUD regions masked out
  2. colour thresholding: "lit" = glowing/coloured pixels OR white pixels.
     The background hex grid is dark (low V), so a V threshold separates it.
  3. connected components; each component is classified by area and shape:
       small blob                  -> food pellet
       medium, round blob          -> big food (remains of a dead snake)
       large / elongated blob      -> snake body
       component under the centre  -> our own snake (camera follows our head)
  4. enemy heads = pairs of white eyes with a dark pupil on a snake body
  5. world border = large dark-red zone, plus our position on the minimap
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from .config import VisionConfig


def _u8(values) -> np.ndarray:
    return np.array(values, dtype=np.uint8)


@dataclass
class EnemyHead:
    x: float
    y: float
    radius: float


@dataclass
class GameState:
    timestamp: float
    width: int
    height: int
    alive: bool
    self_radius: float = 0.0
    self_area: int = 0
    self_heading: float | None = None          # radians, screen coords (y down)
    food: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.float32))      # x, y, weight
    obstacles: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), np.float32))  # x, y
    heads: list[EnemyHead] = field(default_factory=list)
    border_angle: float | None = None          # direction from us towards the nearest border
    border_proximity: float = 0.0              # 0 = world centre, 1 = on the border
    debug: dict = field(default_factory=dict)

    @property
    def center(self) -> tuple[float, float]:
        return self.width / 2.0, self.height / 2.0


class Vision:
    def __init__(self, cfg: VisionConfig) -> None:
        self.cfg = cfg
        self._lit_lo, self._lit_hi = _u8(cfg.lit_lower), _u8(cfg.lit_upper)
        self._white_lo, self._white_hi = _u8(cfg.white_lower), _u8(cfg.white_upper)
        self._eye_lo, self._eye_hi = _u8(cfg.eye_lower), _u8(cfg.eye_upper)
        self._b1_lo, self._b1_hi = _u8(cfg.border_lower1), _u8(cfg.border_upper1)
        self._b2_lo, self._b2_hi = _u8(cfg.border_lower2), _u8(cfg.border_upper2)
        self._dot_lo, self._dot_hi = _u8(cfg.minimap_dot_lower), _u8(cfg.minimap_dot_upper)
        k = cfg.close_kernel
        self._close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)) if k > 1 else None
        self._k3 = np.ones((3, 3), np.uint8)
        self._play_cache: tuple[tuple[int, int], np.ndarray] | None = None

    # ------------------------------------------------------------------ helpers
    def _play_mask(self, h: int, w: int) -> np.ndarray:
        """255 where game content is, 0 on HUD overlays (leaderboard, minimap ...)."""
        if self._play_cache is None or self._play_cache[0] != (h, w):
            mask = np.full((h, w), 255, np.uint8)
            for x0, y0, x1, y1 in self.cfg.ui_rects:
                mask[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)] = 0
            self._play_cache = ((h, w), mask)
        return self._play_cache[1]

    def _self_label(self, labels: np.ndarray, cx: int, cy: int) -> int:
        r = self.cfg.self_search_radius
        window = labels[cy - r:cy + r + 1, cx - r:cx + r + 1].ravel()
        window = window[window > 0]
        return int(np.bincount(window).argmax()) if window.size else 0

    # ------------------------------------------------------------------ main
    def analyze(self, frame: np.ndarray, timestamp: float) -> GameState:
        c = self.cfg
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        play = self._play_mask(h, w)

        # 1) colour thresholding ------------------------------------------------
        lit = cv2.inRange(hsv, self._lit_lo, self._lit_hi)
        lit = cv2.bitwise_or(lit, cv2.inRange(hsv, self._white_lo, self._white_hi))
        lit = cv2.bitwise_and(lit, play)
        if self._close is not None:
            lit = cv2.morphologyEx(lit, cv2.MORPH_CLOSE, self._close)

        # 2) components + classification (vectorised over all components) -----
        n, labels, stats, centroids = cv2.connectedComponentsWithStats(lit, connectivity=8)
        self_label = self._self_label(labels, cx, cy)
        area = stats[:, cv2.CC_STAT_AREA].astype(np.float32)
        bw = stats[:, cv2.CC_STAT_WIDTH].astype(np.float32)
        bh = stats[:, cv2.CC_STAT_HEIGHT].astype(np.float32)
        fill = area / np.maximum(bw * bh, 1.0)
        aspect = np.maximum(bw, bh) / np.maximum(np.minimum(bw, bh), 1.0)

        small_food = (area >= c.food_min_area) & (area <= c.food_max_area)
        big_food = ((area > c.food_max_area) & (area <= c.big_food_max_area)
                    & (fill >= c.big_food_min_fill) & (aspect <= c.big_food_max_aspect))
        snake = (area > c.food_max_area) & ~big_food
        for arr in (small_food, big_food, snake):
            arr[0] = False                       # label 0 = background
            if self_label:
                arr[self_label] = False

        food_idx = np.flatnonzero(small_food | big_food)
        weights = area[food_idx] * np.where(big_food[food_idx], c.big_food_weight, 1.0)
        food = np.column_stack([centroids[food_idx], weights]).astype(np.float32)

        # obstacle mask via lookup table: one fancy-index op instead of a loop
        lut = np.where(snake, 255, 0).astype(np.uint8)
        obstacle = lut[labels]

        # 3) world border (dark red area outside the map) ----------------------
        border = cv2.bitwise_or(cv2.inRange(hsv, self._b1_lo, self._b1_hi),
                                cv2.inRange(hsv, self._b2_lo, self._b2_hi))
        border = cv2.bitwise_and(border, play)
        border_visible = cv2.countNonZero(border) >= c.border_min_area
        if border_visible:
            obstacle = cv2.bitwise_or(obstacle, border)

        # only the outline matters for distance queries -> far fewer points
        outline = cv2.subtract(obstacle, cv2.erode(obstacle, self._k3))
        ys, xs = np.nonzero(outline)
        if xs.size > c.max_obstacle_points:
            step = math.ceil(xs.size / c.max_obstacle_points)
            xs, ys = xs[::step], ys[::step]
        obstacles = np.column_stack([xs, ys]).astype(np.float32)

        state = GameState(
            timestamp=timestamp, width=w, height=h,
            alive=bool(self_label) and area[self_label] >= c.min_self_area,
            food=food, obstacles=obstacles,
        )
        if state.alive:
            state.self_area = int(area[self_label])
            self._analyze_self(state, labels, self_label, cx, cy)
        state.heads = self._find_heads(hsv, play, obstacle, cx, cy, state.self_radius)
        if c.minimap_enabled:
            self._read_minimap(state, hsv)

        state.debug.update(lit=lit, obstacle=obstacle, border_visible=border_visible,
                           components=n - 1)
        return state

    # ------------------------------------------------------------------ own snake
    def _analyze_self(self, state: GameState, labels: np.ndarray, label: int,
                      cx: int, cy: int) -> None:
        c = self.cfg
        h, w = labels.shape
        R = c.self_roi_radius
        x0, x1 = max(cx - R, 0), min(cx + R + 1, w)
        y0, y1 = max(cy - R, 0), min(cy + R + 1, h)
        roi = (labels[y0:y1, x0:x1] == label).astype(np.uint8)

        # own thickness = largest inscribed circle near the head
        dist = cv2.distanceTransform(roi, cv2.DIST_L2, 3)
        hw = c.head_window
        local = dist[max(cy - y0 - hw, 0):cy - y0 + hw + 1, max(cx - x0 - hw, 0):cx - x0 + hw + 1]
        radius = float(local.max()) if local.size else float(dist.max())
        state.self_radius = radius

        # heading: the neck lies behind the head; heading points neck -> head
        yy, xx = np.nonzero(roi)
        dx = xx + x0 - cx
        dy = yy + y0 - cy
        d = np.hypot(dx, dy)
        ring = (d > 2.0 * radius) & (d < 4.5 * radius)
        if np.count_nonzero(ring) >= 10:
            state.self_heading = math.atan2(-dy[ring].mean(), -dx[ring].mean())

    # ------------------------------------------------------------------ enemy heads
    def _find_heads(self, hsv: np.ndarray, play: np.ndarray, obstacle: np.ndarray,
                    cx: int, cy: int, self_radius: float) -> list[EnemyHead]:
        c = self.cfg
        eye = cv2.bitwise_and(cv2.inRange(hsv, self._eye_lo, self._eye_hi), play)
        m, _, stats, cents = cv2.connectedComponentsWithStats(eye, connectivity=8)
        if m <= 1:
            return []
        a = stats[:, cv2.CC_STAT_AREA]
        valid = np.flatnonzero((a >= c.eye_min_area) & (a <= c.eye_max_area))
        valid = valid[valid > 0]
        exclusion = max(c.self_eye_exclusion, 3.0 * self_radius)
        h, w = obstacle.shape
        value = hsv[:, :, 2]

        eyes: list[tuple[float, float]] = []
        for i in valid:
            ex, ey = cents[i]
            if math.hypot(ex - cx, ey - cy) < exclusion:
                continue
            xi, yi = int(ex), int(ey)
            if not obstacle[max(yi - 3, 0):yi + 4, max(xi - 3, 0):xi + 4].any():
                continue                          # white food etc., not on a snake
            x, y = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
            bw, bh = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            pupil = np.count_nonzero(value[y:y + bh, x:x + bw] <= c.eye_pupil_max_v)
            if pupil < c.eye_min_pupil_px:
                continue                          # white skin patch, no pupil
            eyes.append((float(ex), float(ey)))

        heads: list[EnemyHead] = []
        used = [False] * len(eyes)
        for i, (ex, ey) in enumerate(eyes):
            if used[i]:
                continue
            group = [i]
            for j in range(i + 1, len(eyes)):
                if not used[j] and math.hypot(eyes[j][0] - ex, eyes[j][1] - ey) <= c.eye_pair_max_dist:
                    group.append(j)
            for j in group:
                used[j] = True
            if len(group) > 2:
                continue                          # stripes of a white skin, not eyes
            hx = sum(eyes[j][0] for j in group) / len(group)
            hy = sum(eyes[j][1] for j in group) / len(group)
            r = int(c.eye_pair_max_dist)
            xi, yi = int(hx), int(hy)
            roi = obstacle[max(yi - r, 0):min(yi + r + 1, h), max(xi - r, 0):min(xi + r + 1, w)]
            dist = cv2.distanceTransform((roi > 0).astype(np.uint8), cv2.DIST_L2, 3)
            heads.append(EnemyHead(hx, hy, float(dist.max()) if dist.size else 0.0))
        return heads

    # ------------------------------------------------------------------ minimap
    def _read_minimap(self, state: GameState, hsv: np.ndarray) -> None:
        c = self.cfg
        h, w = hsv.shape[:2]
        x0, y0, x1, y1 = c.minimap_rect
        x0, x1 = int(x0 * w), int(x1 * w)
        y0, y1 = int(y0 * h), int(y1 * h)
        roi = hsv[y0:y1, x0:x1]
        if roi.size == 0:
            return
        dot = cv2.inRange(roi, self._dot_lo, self._dot_hi)
        if cv2.countNonZero(dot) == 0:
            return
        # brightest pixel inside the dot mask = our position marker
        _, _, _, (px, py) = cv2.minMaxLoc(roi[:, :, 2], mask=dot)
        mcx, mcy = (x1 - x0) / 2.0, (y1 - y0) / 2.0
        radius = min(x1 - x0, y1 - y0) / 2.0 * c.minimap_radius_frac
        dx, dy = (px - mcx) / radius, (py - mcy) / radius
        state.border_proximity = float(min(math.hypot(dx, dy), 1.5))
        state.border_angle = math.atan2(dy, dx)
