import math

import cv2
import numpy as np

from slither_bot.brain import Brain, Mode, wrap
from slither_bot.config import BrainConfig, VisionConfig
from slither_bot.minimap import analyze_minimap
from slither_bot.vision import Vision

from . import synthetic as syn

VCFG = VisionConfig(minimap_enabled=False)


def own_snake(img):
    syn.draw_snake(img, [(syn.CX - k * 4, syn.CY) for k in range(45)], 9, (60, 200, 60), eyes_dir=(1, 0))


def scene_enemy_alive():
    img = syn.background()
    own_snake(img)
    syn.draw_snake(img, [(610 + k * 3, 270 - k * 2) for k in range(40)], 9, (200, 60, 200), eyes_dir=(-0.83, 0.55))
    return img


def scene_enemy_dead():
    """Same place, the snake is gone and left a trail of big glowing orbs."""
    img = syn.background()
    own_snake(img)
    for k in range(0, 40, 6):
        cv2.circle(img, (610 + k * 3, 270 - k * 2), 7, (255, 120, 255), -1)
    return img


def test_round_blob_without_eyes_is_loot_with_eyes_is_snake():
    img = syn.background()
    own_snake(img)
    cv2.circle(img, (700, 150), 12, (255, 60, 200), -1)                       # orb, no eyes
    syn.draw_snake(img, [(300, 420)], 12, (60, 200, 255), eyes_dir=(1, 0))    # tiny round snake
    s = Vision(VCFG).analyze(img, 0.0)
    food_xy = s.food[:, :2]
    assert np.min(np.hypot(food_xy[:, 0] - 700, food_xy[:, 1] - 150)) < 3
    assert np.min(np.hypot(food_xy[:, 0] - 300, food_xy[:, 1] - 420)) > 20
    assert any(math.hypot(h.x - 300, h.y - 420) < 15 for h in s.heads)


def test_kill_leads_to_loot_mode():
    vision, brain = Vision(VCFG), Brain(BrainConfig())
    t = 0.0
    for _ in range(12):                                   # enemy visible, stable track
        brain.decide(vision.analyze(scene_enemy_alive(), t))
        t += 1 / 30
    modes = []
    for _ in range(20):                                   # it died: head gone, orbs appear
        a = brain.decide(vision.analyze(scene_enemy_dead(), t))
        modes.append(a.mode)
        t += 1 / 30
    assert Mode.LOOT in modes
    assert any(z.source == "death" and z.confirmed for z in brain.zones)
    assert abs(wrap(a.angle - math.atan2(-30, 160))) < math.radians(30)   # towards the remains


def test_vanished_head_without_remains_is_ignored():
    vision, brain = Vision(VCFG), Brain(BrainConfig(pit_enabled=False))
    t = 0.0
    for _ in range(12):
        brain.decide(vision.analyze(scene_enemy_alive(), t))
        t += 1 / 30
    empty = syn.background()
    own_snake(empty)
    for _ in range(45):                                   # 1.5 s: nothing appears there
        a = brain.decide(vision.analyze(empty, t))
        t += 1 / 30
    assert a.mode is not Mode.LOOT and not brain.zones


def test_merged_orbs_in_zone_become_food_not_wall():
    vision, brain = Vision(VCFG), Brain(BrainConfig())
    img = syn.background()
    own_snake(img)
    for k in range(10):                                   # overlapping orbs -> one long eyeless blob
        cv2.circle(img, (600 + k * 9, 200), 8, (255, 120, 255), -1)
    s = vision.analyze(img, 0.0)
    assert len(s.eyeless_blobs) == 1 and s.obstacle_eyeless.any()
    from slither_bot.brain import LootZone
    brain.zones = [LootZone(160.0, -70.0, 0.0, "death", confirmed=True)]
    obstacles, food = brain._apply_loot(s, *s.center)
    assert len(obstacles) < len(s.obstacles)
    assert len(food) == len(s.food) + 1


def test_food_pile_becomes_pit_target():
    vision, brain = Vision(VCFG), Brain(BrainConfig(pit_min_mass=300))
    img = syn.background()
    own_snake(img)
    syn.draw_food(img, (480, 90), 40, 30, seed=4)        # dense pile straight up
    a = brain.decide(vision.analyze(img, 0.0))
    assert a.mode is Mode.LOOT
    assert abs(wrap(a.angle + math.pi / 2)) < math.radians(20)


def minimap_crop(dot, crowd):
    img = np.full((136, 136, 3), (34, 28, 22), np.uint8)        # dark background
    cv2.circle(img, crowd, 9, (118, 118, 118), -1)              # population: mid grey
    cv2.circle(img, dot, 3, (255, 255, 255), -1)                # own marker: pure white
    return img


def test_minimap_crowd_and_position():
    info = analyze_minimap(minimap_crop(dot=(40, 90), crowd=(100, 40)), VisionConfig(), 1.0)
    assert info is not None
    assert info.pos[0] < 0 and info.pos[1] > 0                  # left, below centre
    assert abs(wrap(info.crowd_angle - math.atan2(40 - 90, 100 - 40))) < math.radians(12)
    assert info.crowd_mass > 20


def test_crowd_bias_steers_when_nothing_to_eat():
    vision, brain = Vision(VCFG), Brain(BrainConfig(w_crowd=1.0))
    img = syn.background()
    own_snake(img)
    s = vision.analyze(img, 0.0)
    s.crowd_angle, s.crowd_dist, s.crowd_mass = math.pi / 2, 0.6, 100.0   # crowd is below us
    a = brain.decide(s)
    assert a.mode is Mode.FORAGE
    assert abs(wrap(a.angle - math.pi / 2)) < math.radians(35)
