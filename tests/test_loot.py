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
    assert brain.stats["deaths_seen"] == 1 and brain.stats["loot_runs"] >= 1
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


def test_living_snake_with_flickering_eyes_is_not_a_kill():
    """The overnight bug: eyes drop out of the detection for a moment, body stays ->
    must NOT become a loot zone (the bot used to drive straight into such bodies)."""
    vision, brain = Vision(VCFG), Brain(BrainConfig(pit_enabled=False))
    t = 0.0
    for _ in range(12):
        brain.decide(vision.analyze(scene_enemy_alive(), t))
        t += 1 / 30
    no_eyes = syn.background()
    own_snake(no_eyes)
    syn.draw_snake(no_eyes, [(610 + k * 3, 270 - k * 2) for k in range(40)], 9, (200, 60, 200))
    for k in range(5):                                    # some big food lying around it
        cv2.circle(no_eyes, (600 + 14 * k, 300), 6, (0, 200, 255), -1)
    modes = []
    for _ in range(45):
        a = brain.decide(vision.analyze(no_eyes, t))
        modes.append(a.mode)
        t += 1 / 30
    assert Mode.LOOT not in modes
    assert not any(z.source == "death" and z.confirmed for z in brain.zones)
    assert brain.stats["deaths_seen"] == 0


def test_food_pile_becomes_pit_target():
    vision, brain = Vision(VCFG), Brain(BrainConfig())
    img = syn.background()
    own_snake(img)
    rng = np.random.default_rng(4)
    for _ in range(12):                                   # pile of big orbs straight up
        x, y = 480 + rng.uniform(-45, 45), 90 + rng.uniform(-30, 30)
        cv2.circle(img, (int(x), int(y)), 6, (255, 120, 255), -1)
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


def test_pro_loop_travels_to_hotspot_when_big_enough():
    vision, brain = Vision(VCFG), Brain(BrainConfig())
    img = syn.background()
    own_snake(img)
    s = vision.analyze(img, 0.0)
    s.crowd_angle, s.crowd_dist, s.crowd_mass = math.pi / 2, 0.6, 100.0   # hotspot is below us
    s.length = 300
    a = brain.decide(s)
    assert a.mode is Mode.TRAVEL and a.debug["phase"] == "TRAVEL"
    assert abs(wrap(a.angle - math.pi / 2)) < math.radians(35)


def test_pro_loop_farms_first_when_small():
    vision, brain = Vision(VCFG), Brain(BrainConfig())
    img = syn.background()
    own_snake(img)
    syn.draw_food(img, (480, 130), 15, 45)                # food up, hotspot down
    s = vision.analyze(img, 0.0)
    s.crowd_angle, s.crowd_dist, s.crowd_mass = math.pi / 2, 0.6, 100.0
    s.length = 20
    a = brain.decide(s)
    assert a.debug["phase"] == "FARM"
    assert abs(wrap(a.angle + math.pi / 2)) < math.radians(45)          # goes for the food
