"""Decision layer: sector-based potential field plus a tactical state machine.

The 360 degrees around our head are split into N sectors. For every sector we
compute, fully vectorised with numpy (< 1 ms):

  clearance  free distance along that direction, taking our own body width
             into account (an obstacle beside the ray blocks it as well)
  danger     0..1, grows quadratically when clearance gets small, plus
             predicted enemy-head positions and the world border
  food       distance-weighted food mass, smoothed over neighbouring sectors

  score = w_food * food - w_danger * danger - w_turn * turn - w_switch * switch

Priorities from the spec map onto the code like this:
  Prio 1 survival   hard constraint (forbidden sectors) + ESCAPE mode
  Prio 2 growth     food term of the score + FEAST boost
  Prio 3 tactics    CUTOFF (boost in front of an enemy head) and ENCIRCLE
Tactics only run while survival is not at stake and are aborted the moment
their direction becomes unsafe.
"""
from __future__ import annotations

import enum
import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from .config import BrainConfig
from .tracker import HeadTracker, Track
from .vision import GameState

TAU = 2.0 * math.pi


def wrap(a):
    """Wrap an angle (scalar or array) to [-pi, pi)."""
    return (a + math.pi) % TAU - math.pi


class Mode(str, enum.Enum):
    DEAD = "dead"
    FORAGE = "forage"
    FEAST = "feast"
    ESCAPE = "escape"
    CUTOFF = "cutoff"
    ENCIRCLE = "encircle"
    HUNT = "hunt"
    LOOT = "loot"


@dataclass
class Action:
    angle: float
    boost: bool
    mode: Mode
    debug: dict = field(default_factory=dict)


@dataclass
class _Maneuver:
    mode: Mode
    target_id: int
    started: float
    lead: float = 0.0
    phase: str = "approach"
    phase_started: float = 0.0
    hold_angle: float = 0.0
    orbit_sign: float = 1.0


@dataclass
class LootZone:
    x: float                # relative to our head, kept up to date with our own motion
    y: float
    t0: float
    source: str             # "death" (a head vanished in view) or "pit" (big food pile)
    confirmed: bool = False
    mass: float = 0.0


@dataclass
class _Profiles:
    clear: np.ndarray       # clearance incl. all head predictions
    danger: np.ndarray
    raw: np.ndarray         # raw per-sector minimum distance (no corridor)
    static: np.ndarray      # raw distances without head predictions
    preds: dict             # track id -> per-sector predicted head distance


class Brain:
    def __init__(self, cfg: BrainConfig) -> None:
        self.cfg = cfg
        self.n = cfg.sectors
        self.dtheta = TAU / self.n
        self.angles = -math.pi + (np.arange(self.n) + 0.5) * self.dtheta
        self.unit = np.stack([np.cos(self.angles), np.sin(self.angles)], axis=1)
        self.tracker = HeadTracker(cfg.tracker_max_speed_px_s, cfg.tracker_max_age_s,
                                   cfg.tracker_vel_alpha)
        self.heading = 0.0
        self.last_angle = 0.0
        self.maneuver: _Maneuver | None = None
        self.zones: list[LootZone] = []
        self._last_t: float | None = None
        self._last_boost = False
        # per-frame views of the scene after loot handling (see _apply_loot)
        self._obstacles: np.ndarray | None = None
        self._food: np.ndarray | None = None

        half = max(1, int(math.ceil(3 * cfg.food_sigma_sectors)))
        k = np.arange(-half, half + 1)
        self._food_kernel = np.exp(-(k ** 2) / (2 * cfg.food_sigma_sectors ** 2))
        self._food_half = half
        self._corr_k = np.arange(1, self.n // 4 + 1)
        self._corr_sin = np.sin(self._corr_k * self.dtheta)
        self._corr_cos = np.cos(self._corr_k * self.dtheta)

    # ================================================================ profiles
    def sector_of(self, angle):
        return (((np.asarray(angle) + math.pi) / self.dtheta).astype(int)) % self.n

    def _min_into(self, target: np.ndarray, dx, dy, dist) -> None:
        idx = self.sector_of(np.arctan2(dy, dx))
        np.minimum.at(target, idx, dist)

    def _corridor(self, raw: np.ndarray, margin: float) -> np.ndarray:
        """Free distance along each ray for a snake of half-width ``margin``.

        An obstacle k sectors beside the ray, at distance d, intrudes into the
        corridor if d*sin(k*dθ) < margin; it is then hit after d*cos(k*dθ).
        """
        eff = raw.copy()
        for k, s, c in zip(self._corr_k, self._corr_sin, self._corr_cos):
            for shifted in (np.roll(raw, k), np.roll(raw, -k)):
                hit = shifted * s < margin
                eff = np.where(hit, np.minimum(eff, shifted * c), eff)
        return eff

    def _profiles(self, s: GameState, tracks: list[Track], cx: float, cy: float,
                  margin: float, exclude_id: int | None = None) -> _Profiles:
        cfg = self.cfg
        static = np.full(self.n, cfg.view_radius, np.float64)
        obstacles = s.obstacles if self._obstacles is None else self._obstacles
        if len(obstacles):
            dx = obstacles[:, 0] - cx
            dy = obstacles[:, 1] - cy
            self._min_into(static, dx, dy, np.hypot(dx, dy))

        # predicted head positions are extra, inflated obstacles
        preds = {}
        taus = np.linspace(0.0, cfg.head_predict_horizon_s, 7)
        for tr in tracks:
            px = tr.x + tr.vx * taus
            py = tr.y + tr.vy * taus
            d = np.maximum(np.hypot(px, py) - cfg.head_inflate_factor * max(tr.radius, 4.0), 0.0)
            arr = np.full(self.n, cfg.view_radius, np.float64)
            self._min_into(arr, px, py, d)
            preds[tr.id] = arr

        raw = static.copy()
        for tid, arr in preds.items():
            if tid != exclude_id:
                np.minimum(raw, arr, out=raw)
        clear = self._corridor(raw, margin)

        R = max(cfg.danger_radius, cfg.danger_radius_per_self_r * s.self_radius)
        danger = np.clip((R - clear) / R, 0.0, 1.0) ** 2
        if s.border_angle is not None and s.border_proximity > cfg.border_warn:
            f = min((s.border_proximity - cfg.border_warn) / (1.0 - cfg.border_warn), 1.0)
            danger = danger + cfg.border_weight * f * np.clip(np.cos(self.angles - s.border_angle), 0, None)
        return _Profiles(clear, danger, raw, static, preds)

    def _food_profile(self, s: GameState, raw: np.ndarray, cx: float, cy: float):
        cfg = self.cfg
        food = s.food if self._food is None else self._food
        if len(food) == 0:
            return np.zeros(self.n), np.zeros(self.n)
        dx = food[:, 0] - cx
        dy = food[:, 1] - cy
        d = np.hypot(dx, dy)
        idx = self.sector_of(np.arctan2(dy, dx))
        visible = d < raw[idx] - cfg.food_obstacle_margin       # not behind / inside a snake
        w = food[:, 2] * np.exp(-d / cfg.food_dist_scale)
        prof = np.bincount(idx[visible], weights=w[visible], minlength=self.n)
        near = visible & (d < cfg.feast_radius)
        mass = np.bincount(idx[near], weights=food[near, 2], minlength=self.n)

        hk = self._food_half                                       # circular smoothing
        padded = np.concatenate([prof[-hk:], prof, prof[:hk]])
        prof = np.convolve(padded, self._food_kernel, mode="valid")
        padded = np.concatenate([mass[-hk:], mass, mass[:hk]])
        mass = np.convolve(padded, np.ones(5), mode="same")[hk:-hk]
        peak = prof.max()
        return (prof / peak if peak > 0 else prof), mass

    # ================================================================ decide
    def decide(self, s: GameState) -> Action:
        cfg = self.cfg
        t = s.timestamp
        dt = 0.0 if self._last_t is None else min(max(t - self._last_t, 0.0), 0.1)
        self._last_t = t

        if not s.alive:
            self.maneuver = None
            self.zones = []
            self.tracker.reset()
            return Action(self.last_angle, False, Mode.DEAD)

        cx, cy = s.center
        tracks = self.tracker.update(s.heads, t, cx, cy)
        self._update_heading(s, dt)
        self._update_loot(s, tracks, cx, cy, t, dt)
        self._obstacles, self._food = self._apply_loot(s, cx, cy)
        self_r = max(s.self_radius, cfg.min_self_radius)
        margin = cfg.clearance_factor * self_r + cfg.clearance_pad

        p = self._profiles(s, tracks, cx, cy, margin)
        food, mass = self._food_profile(s, p.raw, cx, cy)
        turn = np.abs(wrap(self.angles - self.heading)) / math.pi
        switch = np.abs(wrap(self.angles - self.last_angle)) / math.pi
        blocked = p.clear < cfg.hard_min_clearance
        can_boost = s.self_area >= cfg.boost_min_self_area
        dbg = dict(clear=p.clear, danger=p.danger, food=food, heading=self.heading,
                   tracks=tracks, margin=margin, zones=list(self.zones))

        # ---- Prio 1: survival ------------------------------------------------
        ahead = int(self.sector_of(self.heading))
        threat = self._closest_threat(tracks)
        if blocked.all() or p.danger[ahead] >= cfg.escape_danger or threat is not None:
            self.maneuver = None
            esc = -cfg.w_danger * p.danger - 0.5 * cfg.w_turn * turn + 1e-3 * p.clear
            esc[blocked] = -np.inf
            best = int(np.argmax(p.clear)) if blocked.all() else int(np.argmax(esc))
            boost = (can_boost and threat is not None
                     and p.clear[best] > cfg.escape_boost_clearance
                     and turn[best] < 0.25)          # boosting widens the turning circle
            dbg.update(score=esc, threat=threat.id if threat else None)
            return self._act(float(self.angles[best]), boost, Mode.ESCAPE, dbg)

        # ---- loot first: remains of a kill / big food piles -------------------
        zone = self._best_zone(tracks)
        if zone is not None:
            dist = math.hypot(zone.x, zone.y)
            angle = math.atan2(zone.y, zone.x)
            if dist > 0.5 * cfg.loot_radius and self._safe(p, angle, min(dist, 150.0)):
                self.maneuver = None
                k = int(self.sector_of(angle))
                worth = zone.source == "death" or zone.mass >= 2 * cfg.pit_min_mass
                boost = can_boost and worth and dist > cfg.loot_boost_dist and turn[k] < 0.35
                dbg["score"] = None
                return self._act(angle, boost, Mode.LOOT, dbg)
            # inside the zone: FORAGE eats it, the orbs are food now

        # ---- Prio 3: running or new tactical maneuver ------------------------
        if self.maneuver is None:
            self.maneuver = self._find_maneuver(s, tracks, cx, cy, margin, t)
        if self.maneuver is not None:
            res = self._run_maneuver(s, tracks, cx, cy, margin, t, can_boost)
            if res is not None:
                dbg["score"] = None
                return self._act(res[0], res[1], self.maneuver.mode if self.maneuver else Mode.FORAGE, dbg)

        # ---- Prio 2: growth ----------------------------------------------------
        score = cfg.w_food * food - cfg.w_danger * p.danger - cfg.w_turn * turn - cfg.w_switch * switch
        if s.crowd_angle is not None and s.crowd_mass >= cfg.crowd_min_mass:
            # minimap: head for the populated areas, where kills and loot happen
            pull = cfg.w_crowd * min(1.0, 3.0 * s.crowd_dist)
            score += pull * np.clip(np.cos(self.angles - s.crowd_angle), 0, None)
        elif food.max() == 0 and s.border_angle is not None:
            # nothing to eat in view: drift towards the (food-rich) world centre
            score += cfg.w_center * s.border_proximity * np.clip(np.cos(self.angles - s.border_angle - math.pi), 0, None)
        score[blocked] = -np.inf
        best = int(np.argmax(score))
        mode = Mode.FORAGE
        boost = False
        if (can_boost and mass[best] >= cfg.feast_threshold and p.danger[best] < 0.2
                and p.clear[best] > cfg.feast_radius and turn[best] < 0.25):
            mode, boost = Mode.FEAST, True               # grab dead-snake remains first
        dbg["score"] = score
        return self._act(float(self.angles[best]), boost, mode, dbg)

    def _act(self, angle: float, boost: bool, mode: Mode, dbg: dict) -> Action:
        self.last_angle = angle
        self._last_boost = boost
        return Action(angle, boost, mode, dbg)

    # ================================================================ loot
    def _loot_items(self, s: GameState, cx: float, cy: float) -> np.ndarray:
        """Relative (x, y, mass) of everything that could be remains of a snake."""
        big = s.food[s.food[:, 2] >= self.cfg.loot_item_min_weight] if len(s.food) else s.food
        items = np.concatenate([big, s.eyeless_blobs]) if len(s.eyeless_blobs) else big
        if len(items) == 0:
            return np.zeros((0, 3))
        return np.column_stack([items[:, 0] - cx, items[:, 1] - cy, items[:, 2]])

    def _update_loot(self, s: GameState, tracks: list[Track], cx: float, cy: float,
                     t: float, dt: float) -> None:
        cfg = self.cfg
        # the camera follows us: world points drift opposite to our own motion
        speed = cfg.boost_speed_px_s if self._last_boost else cfg.base_speed_px_s
        mx, my = speed * math.cos(self.heading) * dt, speed * math.sin(self.heading) * dt
        for z in self.zones:
            z.x -= mx
            z.y -= my

        if cfg.loot_enabled:
            m = cfg.death_edge_margin
            for tr in self.tracker.lost:          # a stable head vanished in the middle of the view
                ax, ay = tr.x + cx, tr.y + cy
                if (tr.hits >= cfg.death_min_hits and m < ax < s.width - m and m < ay < s.height - m
                        and tr.dist <= cfg.loot_max_dist):
                    self.zones.append(LootZone(tr.x, tr.y, t, "death"))

        loot_items = self._loot_items(s, cx, cy)
        all_food = (np.column_stack([s.food[:, 0] - cx, s.food[:, 1] - cy, s.food[:, 2]])
                    if len(s.food) else np.zeros((0, 3)))
        keep = []
        for z in self.zones:
            # deaths are confirmed by orbs/remains, pits by any food
            items = loot_items if z.source == "death" else all_food
            if len(items):
                d = np.hypot(items[:, 0] - z.x, items[:, 1] - z.y)
                near = d < cfg.loot_radius
                z.mass = float(items[near, 2].sum())
                if z.mass > 0:                     # re-centre on what is actually there
                    w = items[near, 2]
                    z.x += 0.5 * (float(np.dot(items[near, 0], w) / w.sum()) - z.x)
                    z.y += 0.5 * (float(np.dot(items[near, 1], w) / w.sum()) - z.y)
            else:
                z.mass = 0.0
            age = t - z.t0
            if not z.confirmed:
                if z.mass >= cfg.death_confirm_mass:
                    z.confirmed = True             # it really died: remains appeared
                elif age > cfg.death_confirm_s:
                    continue                       # just left our sight / eye detection flicker
            elif age > cfg.loot_zone_s or (age > 1.0 and z.mass < cfg.loot_min_mass):
                continue                           # eaten up or gone
            elif z.source == "pit" and age > 1.0 and z.mass < 0.5 * cfg.pit_min_mass:
                continue                           # pile got eaten by someone
            keep.append(z)
        self.zones = keep

        if cfg.pit_enabled and not any(z.confirmed for z in self.zones):
            pit = self._find_pit(s, cx, cy, t)
            if pit is not None:
                self.zones.append(pit)

    def _find_pit(self, s: GameState, cx: float, cy: float, t: float) -> LootZone | None:
        """Densest pile of food in view (grid + 3x3 box sum)."""
        cfg = self.cfg
        if len(s.food) < 5:
            return None
        cell = cfg.pit_cell
        gw, gh = int(math.ceil(s.width / cell)), int(math.ceil(s.height / cell))
        ix = np.clip((s.food[:, 0] / cell).astype(int), 0, gw - 1)
        iy = np.clip((s.food[:, 1] / cell).astype(int), 0, gh - 1)
        grid = np.zeros((gh, gw), np.float32)
        np.add.at(grid, (iy, ix), s.food[:, 2])
        box = cv2.boxFilter(grid, -1, (3, 3), normalize=False, borderType=cv2.BORDER_CONSTANT)
        j = int(np.argmax(box))
        average = 9.0 * float(grid.mean())         # what a 3x3 block holds on average
        if box.flat[j] < max(cfg.pit_min_mass, cfg.pit_density_ratio * average):
            return None                            # just normal food, no pile
        by, bx = divmod(j, gw)
        px, py = (bx + 0.5) * cell, (by + 0.5) * cell
        near = np.hypot(s.food[:, 0] - px, s.food[:, 1] - py) < 1.5 * cell
        w = s.food[near, 2]
        zx = float(np.dot(s.food[near, 0], w) / w.sum()) - cx
        zy = float(np.dot(s.food[near, 1], w) / w.sum()) - cy
        if math.hypot(zx, zy) > cfg.loot_max_dist:
            return None
        return LootZone(zx, zy, t, "pit", confirmed=True, mass=float(box.flat[j]))

    def _best_zone(self, tracks: list[Track]) -> LootZone | None:
        cfg = self.cfg
        best, best_score = None, 0.0
        for z in self.zones:
            if not z.confirmed:
                continue
            dist = math.hypot(z.x, z.y)
            if z.source == "pit":                  # don't race a snake that is much closer
                if any(math.hypot(tr.x - z.x, tr.y - z.y) < cfg.pit_compete_ratio * dist for tr in tracks):
                    continue
            score = (z.mass + (500.0 if z.source == "death" else 0.0)) / (dist + 100.0)
            if score > best_score:
                best, best_score = z, score
        return best

    def _apply_loot(self, s: GameState, cx: float, cy: float):
        """Inside confirmed loot zones, eyeless blobs are remains: food, not walls."""
        zones = [z for z in self.zones if z.confirmed]
        if not zones or (len(s.obstacle_eyeless) == 0 and len(s.eyeless_blobs) == 0):
            return s.obstacles, s.food
        r = self.cfg.loot_radius

        def in_zone(x, y):
            inside = np.zeros(len(x), bool)
            for z in zones:
                inside |= np.hypot(x - cx - z.x, y - cy - z.y) < r
            return inside

        obstacles = s.obstacles
        if len(s.obstacles) and len(s.obstacle_eyeless) == len(s.obstacles):
            drop = s.obstacle_eyeless & in_zone(s.obstacles[:, 0], s.obstacles[:, 1])
            obstacles = s.obstacles[~drop]
        food = s.food
        if len(s.eyeless_blobs):
            loot = s.eyeless_blobs[in_zone(s.eyeless_blobs[:, 0], s.eyeless_blobs[:, 1])]
            if len(loot):
                food = np.concatenate([s.food, loot]).astype(np.float32)
        return obstacles, food

    def _update_heading(self, s: GameState, dt: float) -> None:
        cfg = self.cfg
        step = float(np.clip(wrap(self.last_angle - self.heading), -cfg.turn_rate * dt, cfg.turn_rate * dt))
        self.heading = float(wrap(self.heading + step))
        if s.self_heading is not None:
            self.heading = float(wrap(self.heading + cfg.heading_vision_alpha * wrap(s.self_heading - self.heading)))

    def _closest_threat(self, tracks: list[Track]) -> Track | None:
        cfg = self.cfg
        threats = [tr for tr in tracks
                   if tr.dist < cfg.head_threat_radius and tr.closing_speed > cfg.head_threat_closing_speed]
        return min(threats, key=lambda tr: tr.dist) if threats else None

    # ================================================================ tactics
    def _our_velocity(self) -> np.ndarray:
        return self.cfg.base_speed_px_s * np.array([math.cos(self.heading), math.sin(self.heading)])

    def _safe(self, p: _Profiles, angle: float, need: float) -> bool:
        cfg = self.cfg
        k = int(self.sector_of(angle))
        return p.clear[k] >= max(need, cfg.maneuver_min_clearance) and p.danger[k] < cfg.maneuver_max_danger

    def _cutoff_point(self, tr: Track, lead: float):
        """Point ``lead`` px ahead of the enemy head, in world-velocity terms."""
        ve = np.array([tr.vx, tr.vy]) + self._our_velocity()   # relative -> world
        speed = float(np.hypot(*ve))
        if speed < 1e-6:
            return None, 0.0, None
        vhat = ve / speed
        return np.array([tr.x, tr.y]) + vhat * lead, speed, vhat

    def _find_maneuver(self, s, tracks, cx, cy, margin, t) -> _Maneuver | None:
        cfg = self.cfg
        cut = self._find_cutoff(s, tracks, cx, cy, margin, t)
        if cut is not None:
            return cut

        if cfg.encircle_enabled and s.self_area >= cfg.encircle_min_self_area:
            prey = [tr for tr in tracks if tr.dist <= cfg.encircle_start_dist
                    and tr.radius <= s.self_radius * cfg.encircle_size_ratio]
            if prey:
                tr = min(prey, key=lambda x: x.dist)
                radial = np.array([tr.x, tr.y]) / max(tr.dist, 1e-6)
                perp = np.array([-radial[1], radial[0]])
                heading_vec = np.array([math.cos(self.heading), math.sin(self.heading)])
                sign = 1.0 if float(np.dot(perp, heading_vec)) >= 0 else -1.0
                return _Maneuver(Mode.ENCIRCLE, tr.id, t, orbit_sign=sign)

        if cfg.hunt_enabled:
            prey = [tr for tr in tracks if tr.dist <= cfg.hunt_radius
                    and tr.radius <= s.self_radius * cfg.hunt_size_ratio]
            if prey:
                return _Maneuver(Mode.HUNT, min(prey, key=lambda x: x.dist).id, t)
        return None

    def _find_cutoff(self, s, tracks, cx, cy, margin, t) -> _Maneuver | None:
        cfg = self.cfg
        if cfg.cutoff_enabled and s.self_area >= cfg.boost_min_self_area:
            best = None
            for tr in tracks:
                if not (cfg.cutoff_min_dist <= tr.dist <= cfg.cutoff_max_dist):
                    continue
                if tr.radius > s.self_radius * cfg.cutoff_size_ratio:
                    continue
                for lead in cfg.cutoff_leads:
                    P, speed, vhat = self._cutoff_point(tr, lead)
                    if P is None or speed < cfg.cutoff_min_enemy_speed:
                        break
                    # we must be beside the enemy: not in front (head-on) nor behind it
                    side = float(np.dot(vhat, -np.array([tr.x, tr.y]) / tr.dist))
                    if not (-0.5 < side < 0.5):
                        break
                    t_enemy = lead / speed
                    t_us = float(np.hypot(*P)) / cfg.boost_speed_px_s
                    if t_us >= t_enemy * cfg.cutoff_time_margin:
                        continue
                    p = self._profiles(s, tracks, cx, cy, margin, exclude_id=tr.id)
                    if self._safe(p, math.atan2(P[1], P[0]), 0.9 * float(np.hypot(*P))):
                        if best is None or t_us < best[0]:
                            best = (t_us, tr.id, lead)
                    break
            if best is not None:
                return _Maneuver(Mode.CUTOFF, best[1], t, lead=best[2])
        return None

    def _run_maneuver(self, s, tracks, cx, cy, margin, t, can_boost):
        """Returns (angle, boost) or None (maneuver aborted / finished)."""
        cfg = self.cfg
        m = self.maneuver
        tr = next((x for x in tracks if x.id == m.target_id), None)
        p = self._profiles(s, tracks, cx, cy, margin, exclude_id=m.target_id)
        heading_vec = np.array([math.cos(self.heading), math.sin(self.heading)])

        if m.mode is Mode.CUTOFF:
            if t - m.started > cfg.cutoff_max_s:
                self.maneuver = None
                return None
            if m.phase == "approach":
                if tr is None:
                    self.maneuver = None
                    return None
                P, _, _ = self._cutoff_point(tr, m.lead)
                if P is None:
                    self.maneuver = None
                    return None
                if np.hypot(*P) < cfg.cutoff_arrive_dist or float(np.dot(P, heading_vec)) < 0:
                    # we are across its path: hold course, our body is now the wall
                    m.phase, m.phase_started, m.hold_angle = "hold", t, self.heading
                else:
                    angle = math.atan2(P[1], P[0])
                    if not self._safe(p, angle, 0.9 * float(np.hypot(*P))):
                        self.maneuver = None
                        return None
                    return angle, can_boost
            if t - m.phase_started > cfg.cutoff_hold_s or not self._safe(p, m.hold_angle, 0.0):
                self.maneuver = None
                return None
            return m.hold_angle, False

        if m.mode is Mode.ENCIRCLE:
            if tr is None or t - m.started > cfg.encircle_max_s or tr.dist > cfg.encircle_abort_dist:
                self.maneuver = None
                return None
            orbit = max(cfg.encircle_orbit_min, cfg.encircle_orbit_start - cfg.encircle_shrink_px_s * (t - m.started))
            radial = np.array([tr.x, tr.y]) / max(tr.dist, 1e-6)
            tangent = m.orbit_sign * np.array([-radial[1], radial[0]])
            direction = tangent + cfg.encircle_gain * (tr.dist - orbit) / orbit * radial
            angle = math.atan2(direction[1], direction[0])
            if p.clear[int(self.sector_of(angle))] < cfg.hard_min_clearance:
                self.maneuver = None
                return None
            return angle, can_boost and tr.dist > 1.4 * orbit   # prey is escaping the ring

        if m.mode is Mode.HUNT:
            if tr is None or t - m.started > cfg.hunt_max_s or tr.dist > 1.3 * cfg.hunt_radius:
                self.maneuver = None
                return None
            cut = self._find_cutoff(s, tracks, cx, cy, margin, t)
            if cut is not None:                   # in position: strike
                self.maneuver = cut
                return self._run_maneuver(s, tracks, cx, cy, margin, t, can_boost)
            ve = np.array([tr.vx, tr.vy]) + self._our_velocity()
            speed = float(np.hypot(*ve))
            if speed < 1e-6:
                self.maneuver = None
                return None
            vhat = ve / speed
            perp = np.array([-vhat[1], vhat[0]])
            e = np.array([tr.x, tr.y])
            side = 1.0 if float(np.dot(perp, -e)) >= 0 else -1.0   # stay on our side of its path
            target = e + vhat * cfg.hunt_lead + side * perp * cfg.hunt_offset
            dist = float(np.hypot(*target))
            angle = math.atan2(target[1], target[0])
            if not self._safe(p, angle, min(dist, 150.0)):
                self.maneuver = None
                return None
            return angle, can_boost and dist > 2.5 * cfg.hunt_offset
        self.maneuver = None
        return None
