"""Central configuration.

Every tunable lives here as a dataclass default. A JSON file (``config.json``)
can override any field, section by section::

    {"vision": {"food_max_area": 120}, "brain": {"w_danger": 5.0}}

All pixel values in ``VisionConfig`` and ``BrainConfig`` refer to the
*analysis* resolution, i.e. the screen resolution multiplied by
``CaptureConfig.scale`` (default 0.5 -> 960x540 on a 1080p screen).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class CaptureConfig:
    backend: str = "auto"                 # auto | dxcam | bettercam | mss
    target_fps: int = 60
    scale: float = 0.5                    # analysis resolution relative to the screen
    window_title_regex: str = r"^slither\.io( |$)"
    region: list[int] | None = None       # [left, top, width, height]; skips window detection
    crop_top_px: int = 0                  # cut off browser chrome when not in fullscreen (F11)


@dataclass
class VisionConfig:
    # --- colour thresholds (OpenCV HSV: H 0..179, S 0..255, V 0..255) ---------
    # The background is a dark blue hex grid (V ~ 25..60). Everything the game
    # *lights up* (food, snakes) is bright, so V is the main separator.
    lit_lower: tuple = (0, 70, 90)        # coloured, glowing pixels
    lit_upper: tuple = (179, 255, 255)
    white_lower: tuple = (0, 0, 200)      # white/grey skins, eyes, white food
    white_upper: tuple = (179, 60, 255)
    close_kernel: int = 3                 # morphological close: merges striped skins

    # --- component classification (areas in analysis px) ----------------------
    food_min_area: int = 2
    food_max_area: int = 90               # normal food pellets
    big_food_max_area: int = 450          # remains of dead snakes are bigger ...
    big_food_min_fill: float = 0.55       # ... but still round (area / bbox area)
    big_food_max_aspect: float = 1.6
    big_food_weight: float = 1.5          # weight multiplier for big pellets
    min_self_area: int = 40
    self_search_radius: int = 6           # window around screen centre to find own head
    self_roi_radius: int = 110            # ROI for own radius / heading estimation
    head_window: int = 20

    # --- enemy heads via their eyes (white sclera + black pupil) --------------
    eye_lower: tuple = (0, 0, 215)
    eye_upper: tuple = (179, 40, 255)
    eye_min_area: int = 3
    eye_max_area: int = 120
    eye_pupil_max_v: int = 70
    eye_min_pupil_px: int = 1
    eye_pair_max_dist: float = 25.0
    self_eye_exclusion: float = 25.0

    # --- world border: dark red zone outside the map --------------------------
    border_lower1: tuple = (0, 110, 20)
    border_upper1: tuple = (12, 255, 90)
    border_lower2: tuple = (168, 110, 20)
    border_upper2: tuple = (179, 255, 90)
    border_min_area: int = 1500

    # --- minimap (bottom right) gives the position inside the circular world --
    minimap_enabled: bool = True
    minimap_rect: tuple = (0.875, 0.80, 0.985, 0.985)   # relative x0, y0, x1, y1
    minimap_radius_frac: float = 0.95
    minimap_dot_lower: tuple = (0, 0, 190)
    minimap_dot_upper: tuple = (179, 90, 255)

    # --- HUD regions that must never be interpreted as game objects ----------
    ui_rects: list = field(default_factory=lambda: [
        [0.78, 0.00, 1.00, 0.42],   # leaderboard
        [0.00, 0.88, 0.32, 1.00],   # length / rank
        [0.84, 0.76, 1.00, 1.00],   # minimap
    ])
    max_obstacle_points: int = 8000


@dataclass
class BrainConfig:
    sectors: int = 72                     # 5 degrees per sector
    view_radius: float = 600.0
    min_self_radius: float = 6.0

    # --- survival ------------------------------------------------------------
    clearance_factor: float = 1.6         # corridor half width = factor * own radius + pad
    clearance_pad: float = 4.0
    danger_radius: float = 140.0          # obstacles closer than this are penalised
    danger_radius_per_self_r: float = 6.0
    hard_min_clearance: float = 35.0      # sectors with less free space are forbidden
    escape_danger: float = 0.55           # danger straight ahead that triggers ESCAPE
    escape_boost_clearance: float = 250.0
    head_threat_radius: float = 170.0
    head_threat_closing_speed: float = 40.0   # px/s towards us
    head_predict_horizon_s: float = 0.6
    head_inflate_factor: float = 2.0
    border_warn: float = 0.86             # minimap proximity where border avoidance starts
    border_weight: float = 3.0

    # --- scoring weights -----------------------------------------------------
    w_food: float = 1.0
    w_danger: float = 4.0
    w_turn: float = 0.35
    w_switch: float = 0.15
    w_center: float = 0.25                # drift to map centre when no food is visible
    food_dist_scale: float = 180.0
    food_sigma_sectors: float = 2.0
    food_obstacle_margin: float = 15.0

    # --- motion model --------------------------------------------------------
    turn_rate: float = 5.0                # rad/s, used when vision gives no heading
    heading_vision_alpha: float = 0.6
    base_speed_px_s: float = 110.0
    boost_speed_px_s: float = 260.0
    boost_min_self_area: int = 1200       # boosting burns mass, small snakes cannot afford it

    # --- tactics -------------------------------------------------------------
    feast_radius: float = 220.0
    feast_threshold: float = 250.0
    maneuver_min_clearance: float = 60.0
    maneuver_max_danger: float = 0.35

    cutoff_enabled: bool = True
    cutoff_min_dist: float = 60.0
    cutoff_max_dist: float = 320.0
    cutoff_size_ratio: float = 1.1        # only cut off snakes not much thicker than us
    cutoff_min_enemy_speed: float = 60.0
    cutoff_leads: tuple = (60.0, 100.0, 150.0)
    cutoff_time_margin: float = 0.75
    cutoff_arrive_dist: float = 25.0
    cutoff_hold_s: float = 0.5
    cutoff_max_s: float = 3.0

    encircle_enabled: bool = True
    encircle_min_self_area: int = 6000
    encircle_size_ratio: float = 0.6
    encircle_start_dist: float = 260.0
    encircle_abort_dist: float = 380.0
    encircle_orbit_start: float = 160.0
    encircle_orbit_min: float = 60.0
    encircle_shrink_px_s: float = 25.0
    encircle_gain: float = 1.5
    encircle_max_s: float = 8.0

    # --- head tracker --------------------------------------------------------
    tracker_max_speed_px_s: float = 900.0
    tracker_max_age_s: float = 0.3
    tracker_vel_alpha: float = 0.4


@dataclass
class ControlConfig:
    cursor_radius_px: int = 220           # distance of the cursor from the screen centre
    rate_hz: int = 240
    max_angular_speed: float = 30.0       # rad/s, smooths cursor rotation


@dataclass
class RuntimeConfig:
    start_paused: bool = True             # press F8 in the game window to start
    foreground_check: bool = True         # pause while the game window is not focused
    auto_respawn: bool = False
    respawn_delay_s: float = 2.5
    respawn_retry_s: float = 4.0
    respawn_click_rel: tuple = (0.5, 0.6) # "Play" button, relative to the game region
    respawn_press_enter: bool = True
    debug: bool = False
    debug_every: int = 2
    stats_interval_s: float = 2.0


@dataclass
class BotConfig:
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    brain: BrainConfig = field(default_factory=BrainConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)


def load_config(path: str | Path | None = None) -> BotConfig:
    cfg = BotConfig()
    if path is None or not Path(path).exists():
        return cfg
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    for section, values in data.items():
        sub = getattr(cfg, section, None)
        if sub is None:
            raise KeyError(f"unknown config section: {section!r}")
        for key, value in values.items():
            if not hasattr(sub, key):
                raise KeyError(f"unknown config key: {section}.{key}")
            setattr(sub, key, value)
    return cfg


def save_config(cfg: BotConfig, path: str | Path) -> None:
    Path(path).write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
