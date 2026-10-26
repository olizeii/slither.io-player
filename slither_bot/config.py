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
    white_lower: tuple = (0, 0, 190)      # white/grey skins, eyes, white food
    white_upper: tuple = (179, 70, 255)   # same as the eye range -> one mask serves both
    close_kernel: int = 3                 # morphological close: merges striped skins

    # --- component classification (areas in analysis px) ----------------------
    food_min_area: int = 2
    food_max_area: int = 90               # normal food pellets
    big_food_max_area: int = 1500         # remains of dead snakes are bigger (and have no eyes) ...
    big_food_min_fill: float = 0.55       # ... but still round (area / bbox area)
    big_food_max_aspect: float = 1.6
    big_food_weight: float = 1.5          # weight multiplier for big pellets
    min_self_area: int = 40
    self_search_radius: int = 6           # window around screen centre to find own head
    self_roi_radius: int = 110            # ROI for own radius / heading estimation
    head_window: int = 20

    # --- enemy heads via their eyes (white sclera + black pupil) --------------
    # real frames: glowing (boosting) snakes tint their eyes, S up to ~65
    eye_lower: tuple = (0, 0, 190)
    eye_upper: tuple = (179, 70, 255)
    eye_min_area: int = 2
    eye_max_area: int = 120
    eye_pupil_max_v: int = 70
    eye_min_pupil_px: int = 1
    eye_pair_max_dist: float = 25.0
    eye_require_pair: bool = True         # real frames: glowing food mimics a single eye
    eye_min_body_area: int = 250
    self_eye_exclusion: float = 25.0

    # --- world border: dark red zone outside the map --------------------------
    border_lower1: tuple = (0, 110, 20)
    border_upper1: tuple = (12, 255, 90)
    border_lower2: tuple = (168, 110, 20)
    border_upper2: tuple = (179, 255, 90)
    border_min_area: int = 1500

    # --- minimap (bottom right) gives the position inside the circular world --
    # rect = the game's 136x136 minimap canvases, measured in the page at 1920x1080
    minimap_enabled: bool = True
    minimap_rect: tuple = (0.9146, 0.8481, 0.9854, 0.9741)  # relative x0, y0, x1, y1
    minimap_radius_frac: float = 0.95
    minimap_dot_lower: tuple = (0, 0, 190)               # low-res fallback in Vision
    minimap_dot_upper: tuple = (179, 90, 255)
    # full-resolution analysis (minimap.py): own marker is pure white ...
    minimap_self_min_v: int = 225
    minimap_self_max_s: int = 40
    # ... the population map is mid-grey
    crowd_min_v: int = 85
    crowd_max_s: int = 60
    crowd_min_pixels: int = 15
    crowd_self_exclusion: float = 0.08    # map radii around us (our own snake is on the map too)
    crowd_sigma: float = 0.08             # blur, map radii
    crowd_decay: float = 0.5              # closer crowds are preferred, map radii

    # --- start / death menu: green "Spielen"/"Play Again" button under the
    # purple nickname box (colours taken from the page's CSS) ---------------
    menu_button_lower: tuple = (65, 80, 90)
    menu_button_upper: tuple = (85, 175, 190)
    menu_nick_lower: tuple = (115, 70, 90)
    menu_nick_upper: tuple = (132, 160, 160)
    menu_button_min_area: int = 300
    menu_button_max_area: int = 4000

    # --- HUD regions that must never be interpreted as game objects ----------
    ui_rects: list = field(default_factory=lambda: [
        [0.84, 0.00, 1.00, 0.22],   # leaderboard ("Bestenliste")
        [0.00, 0.95, 0.12, 1.00],   # length / rank
        [0.90, 0.83, 1.00, 1.00],   # minimap + server label
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

    # LOOT: a snake dies in view (our kill or anyone's) -> rush to its remains
    loot_enabled: bool = True
    death_min_hits: int = 6               # head must have been tracked stably before vanishing
    death_edge_margin: float = 40.0       # vanishing near the screen edge = left the view, not died
    death_confirm_mass: float = 120.0     # loot mass that must appear where it vanished ...
    death_confirm_s: float = 1.0          # ... within this time
    loot_radius: float = 110.0            # zone around the remains; eyeless blobs inside = loot
    loot_zone_s: float = 8.0
    loot_max_dist: float = 600.0
    loot_boost_dist: float = 120.0
    loot_min_mass: float = 40.0           # zone is eaten up below this
    loot_item_min_weight: float = 60.0    # food weight that counts as a dead-snake orb
    # PIT: big accumulations of food anywhere in view (several dead snakes, big kills)
    pit_enabled: bool = True
    pit_cell: float = 70.0
    pit_min_mass: float = 700.0           # normal food clusters reach ~400, dead snakes far more
    pit_min_items: int = 6                # ... made of several separate big orbs ...
    pit_density_ratio: float = 4.5        # ... and clearly denser than the frame average
                                          # (a real frame without kills peaked at ~4x)
    # prey: big glowing things that fly across the screen. Food never moves in the
    # world, so a big item that moved 30-150 px within 0.25 s is prey.
    prey_min_shift: float = 30.0
    prey_max_shift: float = 150.0
    prey_weight: float = 0.15
    food_weight_cap: float = 200.0        # one giant blob must not outweigh a whole pile
    pit_compete_ratio: float = 0.7        # skip if an enemy head is this much closer than us
    # minimap crowd: drift towards populated areas (more kills, more loot)
    w_crowd: float = 0.35
    crowd_min_mass: float = 20.0

    # HUNT: shadow a smaller snake alongside its path until a CUTOFF is possible
    hunt_enabled: bool = False
    hunt_radius: float = 420.0
    hunt_size_ratio: float = 1.0
    hunt_offset: float = 90.0             # lateral distance to its path
    hunt_lead: float = 60.0               # how far ahead of its head we aim
    hunt_max_s: float = 6.0

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
    respawn_delay_s: float = 1.0          # menu must be visible this long before clicking
    respawn_retry_s: float = 3.0
    respawn_click_rel: tuple = (0.5, 0.415)   # fallback if the button is not detected
    respawn_press_enter: bool = False
    debug: bool = False
    debug_every: int = 2
    stats_interval_s: float = 2.0


@dataclass
class HudConfig:
    """Reads 'Deine Länge', 'Dein rang' and the final length with Windows OCR."""
    enabled: bool = True
    lang: str = "de"                      # needs the Windows OCR language pack
    interval_s: float = 0.5
    upscale: float = 3.0
    # regions in full-resolution game coordinates (relative)
    stats_rect: tuple = (0.0, 0.955, 0.12, 1.0)       # bottom left, measured on 1920x1080
    final_rect: tuple = (0.30, 0.235, 0.70, 0.295)    # "Deine endgültige Länge war N"
    length_regex: str = r"(?:L[äa]nge|length)\D{0,4}(\d+)"
    rank_regex: str = r"(?:rang|rank)\D{0,4}(\d+)\D{1,6}(\d+)"
    final_regex: str = r"(?:war|was)\D{0,3}(\d+)"


@dataclass
class TrainConfig:
    state_dir: str = "tuning"
    episodes_per_candidate: int = 3
    incumbent_every: int = 4              # every Nth episode re-plays the incumbent (fights luck)
    sigma_start: float = 0.15             # mutation size in normalised [0, 1] space
    sigma_min: float = 0.03
    sigma_max: float = 0.40
    early_reject: float = 0.35            # candidate's 1st score < this * incumbent median -> reject
    accept_margin: float = 0.02           # log-score margin a candidate must beat
    final_length_wait_s: float = 2.0      # how long to wait for the death screen OCR


@dataclass
class BotConfig:
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    brain: BrainConfig = field(default_factory=BrainConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    hud: HudConfig = field(default_factory=HudConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


# Starting point for --aggressive / --train: little fear, constant hunting, boosts a lot.
# v2 (after first user feedback "too passive"): attacks bigger snakes too, cuts off
# without boost, reacts to heads only when they really rush at us.
AGGRESSIVE = {
    "w_danger": 2.0, "danger_radius": 90.0, "clearance_factor": 1.2,
    "hard_min_clearance": 24.0, "escape_danger": 0.8, "head_threat_radius": 110.0,
    "head_threat_closing_speed": 80.0,
    "w_food": 1.4, "feast_threshold": 120.0, "boost_min_self_area": 500,
    "cutoff_size_ratio": 2.2, "cutoff_max_dist": 450.0, "cutoff_time_margin": 1.0,
    "cutoff_min_enemy_speed": 30.0, "encircle_min_self_area": 2500,
    "encircle_size_ratio": 1.0, "hunt_enabled": True, "hunt_radius": 520.0,
    "hunt_size_ratio": 1.5, "w_crowd": 0.6, "loot_boost_dist": 60.0,
}
PROFILE_VERSION = 2   # bump when behaviour changes so much that old tuning scores are meaningless


def apply_brain_params(cfg: BotConfig, params: dict) -> None:
    for key, value in params.items():
        if not hasattr(cfg.brain, key):
            raise KeyError(f"unknown brain parameter: {key}")
        setattr(cfg.brain, key, type(getattr(cfg.brain, key))(value))


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
