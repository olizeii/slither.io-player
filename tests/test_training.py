import math

import cv2

from slither_bot.brain import Action, Mode
from slither_bot.config import AGGRESSIVE, BrainConfig, HudConfig, TrainConfig, VisionConfig
from slither_bot.episode import EpisodeTracker
from slither_bot.hud import parse_final, parse_stats
from slither_bot.tuner import SPACE, Episode, Tuner
from slither_bot.vision import GameState, Vision

from . import synthetic as syn


def menu_frame():
    """Death screen layout at 960x540: purple nickname box, green button below."""
    img = syn.background()
    cv2.rectangle(img, (425, 186), (535, 198), (124, 68, 76), -1)       # BGR of rgb(76,68,124)
    cv2.rectangle(img, (451, 213), (509, 236), (110, 150, 75), -1)      # green button
    cv2.putText(img, "Play", (462, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return img


def test_menu_detected_and_not_alive():
    s = Vision(VisionConfig(minimap_enabled=False)).analyze(menu_frame(), 0.0)
    assert s.menu_button is not None
    assert abs(s.menu_button[0] - 480) < 6 and abs(s.menu_button[1] - 224) < 6
    assert not s.alive


def test_no_menu_during_play():
    s = Vision(VisionConfig(minimap_enabled=False)).analyze(syn.scene(), 0.0)
    assert s.menu_button is None and s.alive


def test_hud_parsing_german_and_english():
    cfg = HudConfig()
    assert parse_stats("Deine Länge: 1234 Dein rang: 7 von 537", cfg) == (1234, 7, 537)
    assert parse_stats("Your length: 88 Your rank: 1 of 600", cfg) == (88, 1, 600)
    assert parse_final("Deine endgültige Länge war 24", cfg) == 24


def ep(length, duration=60.0):
    return Episode(0.0, duration, length, length, 10, 500, 8.0, {})


def test_tuner_accepts_better_candidate(tmp_path):
    cfg = TrainConfig(state_dir=str(tmp_path), episodes_per_candidate=2, incumbent_every=100)
    start = {p.name: getattr(BrainConfig(), p.name) for p in SPACE}
    tuner = Tuner(cfg, start, seed=1)
    for _ in range(2):                                   # baseline
        role, _ = tuner.next_params()
        assert role == "incumbent"
        tuner.report(ep(100))
    role, cand = tuner.next_params()
    assert role == "candidate" and cand != start
    tuner.report(ep(400))
    tuner.next_params()
    msg = tuner.report(ep(500))
    assert "ACCEPTED" in msg
    assert tuner.incumbent == cand
    assert (tmp_path / "best_params.json").exists()
    # state survives a restart
    again = Tuner(cfg, start)
    assert again.incumbent == cand and again.state["episodes"] == 4


def test_tuner_early_rejects_disaster(tmp_path):
    cfg = TrainConfig(state_dir=str(tmp_path), episodes_per_candidate=3, incumbent_every=100)
    tuner = Tuner(cfg, {p.name: getattr(BrainConfig(), p.name) for p in SPACE}, seed=2)
    for _ in range(3):
        tuner.next_params()
        tuner.report(ep(1000))
    tuner.next_params()
    assert "early reject" in tuner.report(ep(20))
    assert tuner.state["candidate"] is None


def test_mutation_stays_in_bounds(tmp_path):
    tuner = Tuner(TrainConfig(state_dir=str(tmp_path), sigma_start=0.4), {}, seed=3)
    params = {p.name: p.from_unit(0.5) for p in SPACE}
    for _ in range(200):
        params = tuner._mutate(params)
    for p in SPACE:
        assert p.lo - 1e-6 <= params[p.name] <= p.hi + 1e-6


def test_aggressive_profile_is_valid():
    cfg = BrainConfig()
    for key in AGGRESSIVE:
        assert hasattr(cfg, key)


def test_episode_tracker_round_trip():
    tracker = EpisodeTracker(hud=None, final_wait_s=1.0)
    act = Action(0.0, False, Mode.FORAGE)
    alive = GameState(0.0, 960, 540, alive=True, self_radius=9.0)
    menu = GameState(0.0, 960, 540, alive=False, menu_button=(480.0, 224.0))
    t = 0.0
    while t < 30.0:
        assert tracker.update(alive, act, t) is None
        t += 1 / 30
    result = tracker.update(menu, act, t)       # menu -> ending; no OCR -> finishes next frame
    result = result or tracker.update(menu, act, t + 0.1)
    assert result is not None
    assert math.isclose(result.duration_s, 30.0, abs_tol=0.2)    # counted from first alive frame
    assert result.modes["forage"] > 25
