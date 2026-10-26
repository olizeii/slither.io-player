import math

import numpy as np

from slither_bot.brain import Brain, Mode, wrap
from slither_bot.config import BrainConfig, VisionConfig
from slither_bot.tracker import HeadTracker
from slither_bot.vision import EnemyHead, Vision

from . import synthetic as syn


def analyze(img, t=0.0):
    cfg = VisionConfig(minimap_enabled=False)
    return Vision(cfg).analyze(img, t)


def test_vision_finds_self_food_enemy():
    s = analyze(syn.scene())
    assert s.alive
    assert 7 <= s.self_radius <= 11
    assert s.self_heading is not None and abs(wrap(s.self_heading)) < math.radians(15)
    # food cluster above the head
    assert len(s.food) >= 5
    assert np.all(s.food[:, 1] < syn.CY - 50)
    # own body is not an obstacle, the enemy is
    d = np.hypot(s.obstacles[:, 0] - syn.CX, s.obstacles[:, 1] - syn.CY)
    assert d.min() > 80
    assert len(s.heads) == 1
    assert math.hypot(s.heads[0].x - 600, s.heads[0].y - 270) < 12


def test_brain_goes_for_food_and_avoids_enemy():
    brain = Brain(BrainConfig())
    s = analyze(syn.scene(enemy_head=(640, 300)))
    action = brain.decide(s)
    assert action.mode is Mode.FORAGE
    assert abs(wrap(action.angle + math.pi / 2)) < math.radians(40)   # up, towards food


def test_brain_escapes_from_close_enemy():
    brain = Brain(BrainConfig())
    s = analyze(syn.scene(enemy_head=(520, 270), food_center=None))
    action = brain.decide(s)
    assert action.mode is Mode.ESCAPE
    assert abs(wrap(action.angle)) > math.radians(60)                 # not into the enemy


def test_corridor_blocks_neighbouring_rays():
    brain = Brain(BrainConfig(sectors=72))
    raw = np.full(72, 600.0)
    raw[int(brain.sector_of(0.0))] = 100.0
    clear = brain._corridor(raw, margin=20.0)
    assert clear[int(brain.sector_of(math.radians(10)))] < 100
    assert clear[int(brain.sector_of(math.radians(30)))] == 600.0


def test_tracker_estimates_velocity():
    tr = HeadTracker(max_speed=900, max_age=0.3, vel_alpha=0.5)
    for i in range(10):
        t = i / 60
        tracks = tr.update([EnemyHead(500 + 120 * t, 270, 8)], t, 480, 270)
    assert len(tracks) == 1
    assert abs(tracks[0].vx - 120) < 10 and abs(tracks[0].vy) < 5
