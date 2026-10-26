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
    TRAVEL = "travel"


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
    # circling
    kind: str = ""                  # "snake" or "pit"
    orbit: float = 0.0              # current ring radius
    inner: float = 0.0              # never tighter than this (prey body / own thickness)
    coverage: float = 0.0           # how much of the ring our body already closes
    zone: "LootZone | None" = None
    last_t: float = 0.0
    center: tuple = (0.0, 0.0)      # ring centre (relative), for the debug overlay


@dataclass
class LootZone:
    x: float                # relative to our head, kept up to date with our own motion
    y: float
    t0: float
    source: str             # "death" (a head vanished in view) or "pit" (big food pile)
    confirmed: bool = False
    mass: float = 0.0
    track_id: int = -1      # which head vanished (kill credit)


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
        self.stats = {"kills": 0, "deaths_seen": 0, "loot_runs": 0}
        self._recent_targets: dict[int, float] = {}   # track id -> last time we attacked it
        self._last_mode = Mode.FORAGE
        self._ego = np.zeros(2)                    # our own integrated movement (prey detection)
        self._food_hist: list[tuple[float, np.ndarray]] = []
        # per-frame views of the scene used by the profiles (food after prey discount)
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
            self._food_hist = []
            self.tracker.reset()
            return Action(self.last_angle, False, Mode.DEAD)

        cx, cy = s.center
        tracks = self.tracker.update(s.heads, t, cx, cy)
        self._update_heading(s, dt)
        s.food = self._discount_movers(s, cx, cy, t, dt)
        self._update_loot(s, tracks, cx, cy, t, dt)
        self._obstacles, self._food = s.obstacles, s.food
        if self.maneuver is not None:
            self._recent_targets[self.maneuver.target_id] = t
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

        # ======== the pro loop: hotspot -> kill -> loot -> (circle) -> repeat ====
        small = self._is_small(s)

        # ---- 2. LOOT: remains of a confirmed kill / a big pile, before others --
        zone = self._best_zone(tracks)
        if zone is not None:
            dist = math.hypot(zone.x, zone.y)
            angle = math.atan2(zone.y, zone.x)
            # long enough to put a ring around the pile? then we circle it (protects it)
            if (zone.source == "pit" and self.maneuver is None and not small
                    and dist < 2 * cfg.loot_radius):
                self.maneuver = self._circle_pit(s, zone, t)
            if self.maneuver is None or self.maneuver.mode is not Mode.ENCIRCLE:
                if dist > 0.5 * cfg.loot_radius and self._safe(p, angle, min(dist, 150.0)):
                    self.maneuver = None
                    k = int(self.sector_of(angle))
                    worth = zone.source == "death" or zone.mass >= 2 * cfg.pit_min_mass
                    boost = can_boost and worth and dist > cfg.loot_boost_dist and turn[k] < 0.35
                    dbg.update(score=None, phase="LOOT")
                    return self._act(angle, boost, Mode.LOOT, dbg)
                # inside the zone: FARM below eats the orbs

        # ---- 3./4. CIRCLE, then ATTACK (cutoff / hunt): running or new ---------
        if self.maneuver is None:
            self.maneuver = self._find_maneuver(s, tracks, cx, cy, margin, t, small)
        if self.maneuver is not None:
            res = self._run_maneuver(s, tracks, cx, cy, margin, t, can_boost)
            if res is not None:
                mode = self.maneuver.mode if self.maneuver else Mode.FORAGE
                dbg.update(score=None, phase="CIRCLE" if mode is Mode.ENCIRCLE else "ATTACK")
                if mode is Mode.ENCIRCLE and self.maneuver is not None:
                    dbg["ring"] = (self.maneuver.center, self.maneuver.orbit, self.maneuver.coverage)
                return self._act(res[0], res[1], mode, dbg)

        # ---- 5. TRAVEL to the minimap hotspot (no fight in view, big enough) --
        base = -cfg.w_danger * p.danger - cfg.w_turn * turn - cfg.w_switch * switch
        hotspot = (s.crowd_angle is not None and s.crowd_mass >= cfg.crowd_min_mass
                   and s.crowd_dist > cfg.hotspot_arrive)
        if hotspot and not small and not tracks:
            pull = np.clip(np.cos(self.angles - s.crowd_angle), 0, None)
            score = base + cfg.w_travel * pull + cfg.travel_food_share * cfg.w_food * food
            score[blocked] = -np.inf
            dbg.update(score=score, phase="TRAVEL")
            return self._act(float(self.angles[int(np.argmax(score))]), False, Mode.TRAVEL, dbg)

        # ---- 6. FARM: grow (start of a round) or patrol inside the hotspot -----
        score = base + cfg.w_food * food
        if s.crowd_angle is not None and s.crowd_mass >= cfg.crowd_min_mass:
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
            mode, boost = Mode.FEAST, True
        dbg.update(score=score, phase="FARM" if small else "PATROL")
        return self._act(float(self.angles[best]), boost, mode, dbg)

    def _is_small(self, s: GameState) -> bool:
        """Self-awareness: still in the growing phase of a round?"""
        if s.length is not None:
            return s.length < self.cfg.grow_until_length
        return s.self_area < self.cfg.grow_until_area

    def _act(self, angle: float, boost: bool, mode: Mode, dbg: dict) -> Action:
        self.last_angle = angle
        self._last_boost = boost
        if mode is Mode.LOOT and self._last_mode is not Mode.LOOT:
            self.stats["loot_runs"] += 1
        self._last_mode = mode
        dbg.setdefault("phase", "SURVIVE" if mode is Mode.ESCAPE else mode.value.upper())
        return Action(angle, boost, mode, dbg)

    # ================================================================ prey
    def _discount_movers(self, s: GameState, cx: float, cy: float, t: float, dt: float) -> np.ndarray:
        """Big glowing things that fly across the screen (slither.io 'prey') are hard
        to catch and pull the bot off course. Food does not move in the world, so
        anything big that does gets only a fraction of its weight. A weight cap
        keeps single giant blobs from dominating the food map as well.
        """
        cfg = self.cfg
        speed = cfg.boost_speed_px_s if self._last_boost else cfg.base_speed_px_s
        self._ego += speed * dt * np.array([math.cos(self.heading), math.sin(self.heading)])
        food = s.food.copy() if len(s.food) else s.food
        movers = np.zeros(len(food), bool)
        if len(food):
            np.minimum(food[:, 2], cfg.food_weight_cap, out=food[:, 2])
            big = np.flatnonzero(food[:, 2] >= cfg.loot_item_min_weight)
            world = food[big, :2] - (cx, cy) + self._ego        # food position in the world
            past = min((e for e in self._food_hist if 0.15 <= t - e[0] <= 0.5),
                       key=lambda e: abs(t - e[0] - 0.25), default=None)
            if past is not None and len(past[1]) and len(big):
                d = np.hypot(world[:, None, 0] - past[1][None, :, 0],
                             world[:, None, 1] - past[1][None, :, 1]).min(axis=1)
                # moved a bit = prey; nothing close at all = new (e.g. fresh remains)
                movers[big[(d > cfg.prey_min_shift) & (d < cfg.prey_max_shift)]] = True
                food[movers, 2] *= cfg.prey_weight
            self._food_hist.append((t, world))
        self._food_hist = [e for e in self._food_hist if t - e[0] <= 0.6]
        s.debug["movers"] = movers
        return food

    # ================================================================ loot
    def _loot_items(self, s: GameState, cx: float, cy: float) -> np.ndarray:
        """Relative (x, y, mass) of separate big orbs = what a dead snake leaves behind.

        Eyeless blobs are deliberately NOT counted: overnight data showed ~25 % of the
        time in LOOT, mostly because a living snake whose eyes flickered out of the
        detection looked like 'head gone + eyeless blob = dead'.
        """
        big = s.food[s.food[:, 2] >= self.cfg.loot_item_min_weight] if len(s.food) else s.food
        if len(big) == 0:
            return np.zeros((0, 3))
        return np.column_stack([big[:, 0] - cx, big[:, 1] - cy, big[:, 2]])

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
                    self.zones.append(LootZone(tr.x, tr.y, t, "death", track_id=tr.id))

        loot_items = self._loot_items(s, cx, cy)
        all_food = (np.column_stack([s.food[:, 0] - cx, s.food[:, 1] - cy, s.food[:, 2]])
                    if len(s.food) else np.zeros((0, 3)))
        keep = []
        for z in self.zones:
            # deaths are confirmed by orbs/remains, pits by any food
            items = loot_items if z.source == "death" else all_food
            n_items = 0
            if len(items):
                d = np.hypot(items[:, 0] - z.x, items[:, 1] - z.y)
                near = d < cfg.loot_radius
                n_items = int(np.count_nonzero(near))
                z.mass = float(items[near, 2].sum())
                if z.mass > 0:                     # re-centre on what is actually there
                    w = items[near, 2]
                    z.x += 0.5 * (float(np.dot(items[near, 0], w) / w.sum()) - z.x)
                    z.y += 0.5 * (float(np.dot(items[near, 1], w) / w.sum()) - z.y)
            else:
                z.mass = 0.0
            age = t - z.t0
            if not z.confirmed:
                if (z.mass >= cfg.death_confirm_mass and n_items >= cfg.death_min_orbs
                        and self._body_points_near(s, cx, cy, z.x, z.y) <= cfg.death_max_body_pts):
                    z.confirmed = True             # it really died: orbs appeared, body is gone
                    self._credit_death(s, z, cx, cy, t)
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

    def _body_points_near(self, s: GameState, cx: float, cy: float, zx: float, zy: float) -> int:
        """Obstacle pixels right where a snake supposedly died: a living body = no death."""
        if len(s.obstacles) == 0:
            return 0
        d = np.hypot(s.obstacles[:, 0] - cx - zx, s.obstacles[:, 1] - cy - zy)
        return int(np.count_nonzero(d < self.cfg.death_clear_radius))

    def _credit_death(self, s: GameState, z: LootZone, cx: float, cy: float, t: float) -> None:
        cfg = self.cfg
        self.stats["deaths_seen"] += 1
        ours = t - self._recent_targets.get(z.track_id, -1e9) < cfg.kill_credit_s
        if not ours and math.hypot(z.x, z.y) < cfg.kill_credit_dist:
            ours = True                               # died right next to our head
        if not ours and len(s.self_points):          # ... or against our body
            d = np.hypot(s.self_points[:, 0] - cx - z.x, s.self_points[:, 1] - cy - z.y)
            ours = float(d.min()) < cfg.kill_credit_dist
        if ours:
            self.stats["kills"] += 1

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
        count = np.zeros((gh, gw), np.float32)    # separate big items: remains are many orbs,
        big = s.food[:, 2] >= cfg.loot_item_min_weight   # one giant blob is not a pile
        np.add.at(count, (iy[big], ix[big]), 1.0)
        box = cv2.boxFilter(grid, -1, (3, 3), normalize=False, borderType=cv2.BORDER_CONSTANT)
        nbox = cv2.boxFilter(count, -1, (3, 3), normalize=False, borderType=cv2.BORDER_CONSTANT)
        box[nbox < cfg.pit_min_items] = 0
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

    def _find_maneuver(self, s, tracks, cx, cy, margin, t, small: bool = False) -> _Maneuver | None:
        """Attack priority: circle (if we are long enough) > cutoff > hunt."""
        cfg = self.cfg
        if not small:
            circle = self._find_circle(s, tracks, cx, cy, margin, t)
            if circle is not None:
                return circle
        cut = self._find_cutoff(s, tracks, cx, cy, margin, t)
        if cut is not None:
            return cut
        if cfg.hunt_enabled and not small:
            prey = [tr for tr in tracks if tr.dist <= cfg.hunt_radius
                    and tr.radius <= s.self_radius * cfg.hunt_size_ratio]
            if prey:
                return _Maneuver(Mode.HUNT, min(prey, key=lambda x: x.dist).id, t)
        return None

    # ================================================================ circling
    def _long_enough(self, s: GameState) -> bool:
        if s.length is not None:
            return s.length >= self.cfg.circle_min_length
        return s.self_area >= self.cfg.encircle_min_self_area

    def _ring_radius_max(self, s: GameState) -> float:
        """Largest ring our (visible) body can close: circumference must fit, with slack."""
        return s.self_length_px / (2 * math.pi * self.cfg.ring_slack)

    def _orbit_sign(self, ox: float, oy: float) -> float:
        d = max(math.hypot(ox, oy), 1e-6)
        perp = np.array([-oy / d, ox / d])
        heading_vec = np.array([math.cos(self.heading), math.sin(self.heading)])
        return 1.0 if float(np.dot(perp, heading_vec)) >= 0 else -1.0   # keep turning the way we go

    def _ring_coverage(self, s: GameState, cx: float, cy: float, ox: float, oy: float, R: float) -> float:
        """How much of the circle around (ox, oy) our own body already covers (0..1).

        This is the 'does the ring close?' check: our body points are binned by angle
        around the prey; 1.0 = it is completely walled in by us.
        """
        if len(s.self_points) == 0:
            return 0.0
        dx = s.self_points[:, 0] - cx - ox
        dy = s.self_points[:, 1] - cy - oy
        d = np.hypot(dx, dy)
        band = (d > 0.5 * R) & (d < 1.6 * R)
        if not band.any():
            return 0.0
        bins = (((np.arctan2(dy[band], dx[band]) + math.pi) / TAU) * 36).astype(int) % 36
        return len(np.unique(bins)) / 36.0

    def _ring_path_free(self, s: GameState, cx: float, cy: float, ox: float, oy: float,
                        R: float, inner: float, margin: float) -> float:
        """Fraction of the planned ring that no *other* snake blocks (the prey inside is fine).
        Our own body never kills us in slither.io, so only third parties matter here."""
        if len(s.obstacles) == 0:
            return 1.0
        obs = s.obstacles[::3] - np.array([cx + ox, cy + oy], np.float32)
        obs = obs[np.hypot(obs[:, 0], obs[:, 1]) > inner]          # drop the prey's coil
        if len(obs) == 0:
            return 1.0
        ang = np.linspace(-math.pi, math.pi, 48, endpoint=False)
        path = np.stack([R * np.cos(ang), R * np.sin(ang)], axis=1)
        d = np.hypot(path[:, None, 0] - obs[None, :, 0], path[:, None, 1] - obs[None, :, 1]).min(axis=1)
        return float(np.mean(d >= margin))

    def _find_circle(self, s, tracks, cx, cy, margin, t) -> _Maneuver | None:
        cfg = self.cfg
        if not cfg.encircle_enabled or not self._long_enough(s):
            return None
        r_max = min(self._ring_radius_max(s), cfg.encircle_orbit_start)
        best = None
        for tr in tracks:
            if tr.dist > cfg.encircle_start_dist or tr.radius > s.self_radius * cfg.encircle_size_ratio:
                continue
            # the prey's visible body must fit inside our ring
            inner = max(cfg.encircle_orbit_min, math.sqrt(max(tr.body_area, 1.0) / math.pi) + 2 * margin)
            if inner > r_max:
                continue
            if self._ring_path_free(s, cx, cy, tr.x, tr.y, r_max, inner, margin) < cfg.ring_path_min_free:
                continue                                   # someone else is in the way
            if best is None or tr.dist < best[0].dist:
                best = (tr, inner)
        if best is None:
            return None
        tr, inner = best
        return _Maneuver(Mode.ENCIRCLE, tr.id, t, orbit_sign=self._orbit_sign(tr.x, tr.y),
                         kind="snake", orbit=r_max, inner=inner, last_t=t)

    def _circle_pit(self, s: GameState, zone: LootZone, t: float) -> _Maneuver | None:
        """Ring around a loot pile, spiralling inwards: eats it and keeps others out."""
        cfg = self.cfg
        if not cfg.encircle_enabled or not self._long_enough(s):
            return None
        r = min(self._ring_radius_max(s), cfg.loot_radius)
        if r < cfg.encircle_orbit_min:
            return None
        return _Maneuver(Mode.ENCIRCLE, -1, t, orbit_sign=self._orbit_sign(zone.x, zone.y),
                         kind="pit", orbit=r, inner=2.5 * max(s.self_radius, cfg.min_self_radius),
                         zone=zone, last_t=t)

    def _find_cutoff(self, s, tracks, cx, cy, margin, t) -> _Maneuver | None:
        cfg = self.cfg
        if cfg.cutoff_enabled:
            # small snakes cannot boost but can still cut off, just at normal speed
            our_speed = cfg.boost_speed_px_s if s.self_area >= cfg.boost_min_self_area else cfg.base_speed_px_s
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
                    t_us = float(np.hypot(*P)) / our_speed
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
            dtm, m.last_t = t - m.last_t, t
            if m.kind == "pit":
                if m.zone not in self.zones or t - m.started > cfg.encircle_max_s:
                    self.maneuver = None             # pile eaten / gone
                    return None
                ox, oy = m.zone.x, m.zone.y
                m.orbit -= cfg.pit_spiral_px_s * dtm  # spiral inwards over the orbs
                if m.orbit < m.inner:
                    self.maneuver = None
                    return None
            else:
                if tr is None or t - m.started > cfg.encircle_max_s or tr.dist > cfg.encircle_abort_dist:
                    self.maneuver = None             # prey died (LOOT takes over) or escaped
                    return None
                ox, oy = tr.x, tr.y
                m.coverage = self._ring_coverage(s, cx, cy, ox, oy, m.orbit)
                if m.coverage >= cfg.ring_closed:    # walled in: tighten slowly until it dies
                    m.orbit = max(m.inner, m.orbit - cfg.encircle_shrink_px_s * dtm)
            m.center = (ox, oy)
            dist = max(math.hypot(ox, oy), 1e-6)
            radial = np.array([ox, oy]) / dist
            tangent = m.orbit_sign * np.array([-radial[1], radial[0]])
            direction = tangent + cfg.encircle_gain * (dist - m.orbit) / m.orbit * radial
            angle = math.atan2(direction[1], direction[0])
            if p.clear[int(self.sector_of(angle))] < cfg.hard_min_clearance:
                self.maneuver = None
                return None
            # boost to close the ring before the prey slips out; never while tightening
            closing = m.kind == "snake" and m.coverage < cfg.ring_closed
            return angle, can_boost and (closing or dist > 1.4 * m.orbit)

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
