import math

import numpy as np

from slither_bot.brain import Brain, Mode, wrap
from slither_bot.config import AGGRESSIVE, BrainConfig, TrainConfig
from slither_bot.tuner import SPACE, Tuner, style, style_line
from slither_bot.vision import GameState

W, H = 960, 540
CX, CY = W / 2, H / 2


def state(t, food):
    s = GameState(t, W, H, alive=True, self_radius=9.0, self_area=3000, self_heading=0.0)
    s.food = np.array(food, np.float32).reshape(-1, 3)
    return s


def test_prey_is_discounted_but_static_food_is_not():
    """Heading 0 at 110 px/s: static food drifts left on screen, prey flies upwards."""
    brain = Brain(BrainConfig(heading_vision_alpha=1.0, pit_enabled=False))
    t, dt = 0.0, 1 / 30
    for i in range(30):
        static = (CX + 200 - 110 * t, CY + 60, 150.0)      # drifts with the camera only
        prey = (CX + 150, CY - 40 - 300 * t, 150.0)        # moves in the world
        s = state(t, [static, prey])
        brain.decide(s)
        t += dt
    movers = s.debug["movers"]
    assert not movers[0] and movers[1]
    assert s.food[1, 2] < 0.2 * 150 and s.food[0, 2] == 150


def test_single_giant_blob_is_not_a_pit_but_many_orbs_are():
    brain = Brain(BrainConfig())
    giant = state(0.0, [(CX, CY - 150, 5000.0)])
    assert brain._find_pit(giant, CX, CY, 0.0) is None
    orbs = [(CX + 10 * k, CY - 150 + 6 * (k % 3), 150.0) for k in range(8)]
    pit = brain._find_pit(state(0.0, orbs), CX, CY, 0.0)
    assert pit is not None and abs(wrap(math.atan2(pit.y, pit.x) + math.pi / 2)) < 0.4


def test_cutoff_possible_without_boost():
    cfg = BrainConfig()
    for k, v in AGGRESSIVE.items():
        setattr(cfg, k, v)
    brain = Brain(cfg)
    from slither_bot.tracker import Track
    # enemy 150 px to our right, slowly moving "up" in the world (we move right at 110 px/s,
    # so on screen it drifts left and up): we are beside it, not in front of it
    tr = Track(1, 150.0, 0.0, 8.0, 0.0, vx=-110.0, vy=-60.0, hits=10)
    s = GameState(0.0, W, H, alive=True, self_radius=9.0, self_area=300)   # too small to boost
    m = brain._find_cutoff(s, [tr], CX, CY, 20.0, 0.0)
    assert m is not None and m.mode is Mode.CUTOFF


def test_style_percentages():
    default = {p.name: getattr(BrainConfig(), p.name) for p in SPACE}
    aggressive = dict(default, **{k: v for k, v in AGGRESSIVE.items() if k in default})
    a, d = style(aggressive), style(default)
    assert a["Aggression"] > d["Aggression"] + 10
    assert a["Hotspot"] > d["Hotspot"]
    assert all(0 <= v <= 100 for v in a.values())
    assert a["Aggression"] >= 50 and "Typ: " in style_line(aggressive)


def test_new_bot_version_starts_fresh_baseline(tmp_path):
    cfg = TrainConfig(state_dir=str(tmp_path), episodes_per_candidate=2)
    start = {p.name: getattr(BrainConfig(), p.name) for p in SPACE}
    t1 = Tuner(cfg, start, version=1)
    t1.next_params()
    t1.state["incumbent_scores"] = [5.0, 6.0]
    t1._save()
    new_start = dict(start, w_danger=2.0)
    t2 = Tuner(cfg, new_start, version=2)
    assert t2.notice and t2.state["incumbent_scores"] == []
    assert t2.incumbent["w_danger"] == 2.0
