import math

import numpy as np

from slither_bot.brain import Brain, Mode
from slither_bot.config import BrainConfig
from slither_bot.hud import PlausibleValue
from slither_bot.tracker import Track
from slither_bot.vision import GameState

W, H = 960, 540
CX, CY = W / 2, H / 2


def ring_points(ox, oy, R, frac=1.0, n=400):
    """Own body laid as (part of) a circle of radius R around (ox, oy), relative coords."""
    a = np.linspace(-math.pi, -math.pi + 2 * math.pi * frac, n)
    return np.stack([CX + ox + R * np.cos(a), CY + oy + R * np.sin(a)], axis=1).astype(np.float32)


def long_state(**kw):
    s = GameState(0.0, W, H, alive=True, self_radius=9.0, self_area=20000, length=1500, **kw)
    s.self_length_px = 2000.0
    return s


def test_ring_coverage_detects_closed_and_open_rings():
    brain = Brain(BrainConfig())
    full = long_state(self_points=ring_points(150, 0, 100))
    half = long_state(self_points=ring_points(150, 0, 100, frac=0.5))
    assert brain._ring_coverage(full, CX, CY, 150, 0, 100) > 0.95
    assert 0.4 < brain._ring_coverage(half, CX, CY, 150, 0, 100) < 0.6


def test_circle_starts_on_small_prey_when_long_enough():
    brain = Brain(BrainConfig())
    prey = Track(7, 150.0, 0.0, 5.0, 0.0, hits=10, body_area=600.0)
    m = brain._find_circle(long_state(), [prey], CX, CY, 18.0, 0.0)
    assert m is not None and m.mode is Mode.ENCIRCLE and m.kind == "snake"
    assert m.inner < m.orbit <= BrainConfig().encircle_orbit_start


def test_no_circle_when_too_short_or_ring_blocked():
    brain = Brain(BrainConfig())
    prey = Track(7, 150.0, 0.0, 5.0, 0.0, hits=10, body_area=600.0)
    short = long_state()
    short.length = 100                                         # below circle_min_length
    assert brain._find_circle(short, [prey], CX, CY, 18.0, 0.0) is None
    # a third snake lies right on the planned ring
    blocker = np.array([[CX + 150 + 160 * math.cos(a), CY + 160 * math.sin(a)]
                        for a in np.linspace(0, 1.5, 60)], np.float32)
    blocked = long_state(obstacles=blocker)
    assert brain._find_circle(blocked, [prey], CX, CY, 18.0, 0.0) is None


def test_ring_tightens_only_once_closed():
    brain = Brain(BrainConfig(ring_closed=0.75, encircle_shrink_px_s=50.0))
    prey = Track(7, 150.0, 0.0, 5.0, 0.0, hits=10, body_area=600.0)
    s = long_state()
    brain.maneuver = brain._find_circle(s, [prey], CX, CY, 18.0, 0.0)
    r0 = brain.maneuver.orbit
    s.self_points = ring_points(150, 0, r0, frac=0.3)          # ring still open
    s.timestamp = 1.0
    brain._run_maneuver(s, [prey], CX, CY, 18.0, 1.0, True)
    assert brain.maneuver.orbit == r0
    s.self_points = ring_points(150, 0, r0)                    # closed
    brain._run_maneuver(s, [prey], CX, CY, 18.0, 2.0, True)
    assert brain.maneuver.orbit < r0


def test_ocr_outliers_are_rejected():
    f = PlausibleValue(max_rise_per_s=40.0)
    assert f.update(391, 0.0) is None                          # not confirmed yet
    assert f.update(392, 0.5) == 392
    assert f.update(29301, 1.0) == 392                         # misread: impossible jump
    assert f.update(395, 1.5) == 395
    assert f.update(10, 2.0) == 395                            # new round: needs confirmation
    assert f.update(11, 2.5) == 11
