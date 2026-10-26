"""Self-tuning of the brain parameters between rounds.

Algorithm: a noise-aware (1+1) evolution strategy.

  * The *incumbent* is the best parameter set so far, with the scores of its
    recent rounds. A *candidate* is the incumbent mutated with Gaussian noise
    in a normalised [0, 1] space (log scale for weights and radii).
  * A candidate plays ``episodes_per_candidate`` rounds. It replaces the
    incumbent if its mean log score beats the incumbent's mean by a margin.
    A catastrophic first round rejects it right away (saves playtime).
  * Every ``incumbent_every``-th round re-plays the incumbent, so a lucky
    streak cannot keep a mediocre set on top forever.
  * Step size follows the 1/5 success rule: grow on success, shrink on failure.

The score of a round is log(final length): slither.io rounds are extremely
noisy (one bad encounter ends everything), the log keeps single huge rounds
from dominating, and the leaderboard sorts by length.

State and a CSV log live in ``tuning/`` so learning continues across sessions.
"""
from __future__ import annotations

import csv
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

from .config import TrainConfig


@dataclass(frozen=True)
class Param:
    name: str
    lo: float
    hi: float
    log: bool = False
    integer: bool = False

    def to_unit(self, v: float) -> float:
        if self.log:
            return (math.log(v) - math.log(self.lo)) / (math.log(self.hi) - math.log(self.lo))
        return (v - self.lo) / (self.hi - self.lo)

    def from_unit(self, u: float) -> float:
        u = min(max(u, 0.0), 1.0)
        v = (math.exp(math.log(self.lo) + u * (math.log(self.hi) - math.log(self.lo)))
             if self.log else self.lo + u * (self.hi - self.lo))
        return float(round(v)) if self.integer else round(v, 4)


# What the bot may change about itself. Survival, greed and aggression knobs.
SPACE = [
    Param("w_danger", 1.0, 8.0, log=True),
    Param("danger_radius", 60.0, 220.0, log=True),
    Param("clearance_factor", 1.0, 2.4),
    Param("hard_min_clearance", 18.0, 60.0),
    Param("escape_danger", 0.35, 0.9),
    Param("head_threat_radius", 80.0, 260.0, log=True),
    Param("head_predict_horizon_s", 0.2, 1.2),
    Param("w_food", 0.5, 3.0, log=True),
    Param("food_dist_scale", 80.0, 400.0, log=True),
    Param("w_turn", 0.05, 1.0, log=True),
    Param("feast_threshold", 60.0, 600.0, log=True),
    Param("boost_min_self_area", 300, 4000, log=True, integer=True),
    Param("cutoff_size_ratio", 0.6, 2.5),
    Param("cutoff_max_dist", 150.0, 500.0),
    Param("cutoff_time_margin", 0.5, 1.1),
    Param("encircle_min_self_area", 1500, 15000, log=True, integer=True),
    Param("encircle_size_ratio", 0.3, 1.2),
    Param("hunt_radius", 150.0, 600.0),
    Param("hunt_offset", 40.0, 200.0, log=True),
    Param("loot_max_dist", 250.0, 700.0),
    Param("loot_boost_dist", 40.0, 400.0, log=True),
    Param("pit_min_mass", 400.0, 2000.0, log=True),
    Param("w_crowd", 0.0, 1.2),
    Param("head_threat_closing_speed", 30.0, 200.0, log=True),
    Param("hunt_size_ratio", 0.6, 2.5),
    Param("w_travel", 0.2, 2.5, log=True),
    Param("grow_until_length", 15, 400, log=True, integer=True),
    Param("circle_min_length", 150, 3000, log=True, integer=True),
    Param("ring_closed", 0.5, 0.95),
]
PARAMS = {p.name: p for p in SPACE}

# Play style in %: where each knob sits inside its range (+1 = more of it, -1 = less).
STYLES = {
    "Aggression": [("cutoff_size_ratio", 1), ("cutoff_max_dist", 1), ("cutoff_time_margin", 1),
                   ("hunt_radius", 1), ("hunt_size_ratio", 1), ("encircle_size_ratio", 1),
                   ("encircle_min_self_area", -1), ("boost_min_self_area", -1),
                   ("circle_min_length", -1), ("grow_until_length", -1),
                   ("w_danger", -1), ("danger_radius", -1)],
    "Vorsicht": [("w_danger", 1), ("danger_radius", 1), ("clearance_factor", 1),
                 ("hard_min_clearance", 1), ("head_threat_radius", 1), ("head_predict_horizon_s", 1),
                 ("escape_danger", -1), ("head_threat_closing_speed", -1)],
    "Gier": [("w_food", 1), ("food_dist_scale", 1), ("feast_threshold", -1)],
    "Loot": [("loot_max_dist", 1), ("loot_boost_dist", -1), ("pit_min_mass", -1)],
    "Hotspot": [("w_crowd", 1), ("w_travel", 1)],
}
STYLE_NAMES = {"Aggression": "Jäger", "Vorsicht": "Überlebender", "Gier": "Sammler",
               "Loot": "Aasfresser", "Hotspot": "Hotspot-Jäger"}


def style(params: dict) -> dict[str, int]:
    out = {}
    for name, knobs in STYLES.items():
        vals = []
        for key, sign in knobs:
            if key in params and key in PARAMS:
                u = min(max(PARAMS[key].to_unit(float(params[key])), 0.0), 1.0)
                vals.append(u if sign > 0 else 1.0 - u)
        out[name] = round(100 * sum(vals) / len(vals)) if vals else 50
    return out


def style_line(params: dict) -> str:
    st = style(params)
    top = max(("Aggression", "Vorsicht", "Gier", "Loot"), key=lambda k: st[k])
    return ("Spielstil: " + " | ".join(f"{k} {v}%" for k, v in st.items())
            + f"  -> Typ: {STYLE_NAMES[top]}")


def behaviour_line(modes: dict) -> str:
    total = sum(modes.values()) or 1.0
    parts = sorted(modes.items(), key=lambda kv: -kv[1])
    return "Verhalten: " + " ".join(f"{k} {100 * v / total:.0f}%" for k, v in parts if v / total >= 0.01)


@dataclass
class Episode:
    started: float
    duration_s: float
    final_length: int | None
    peak_length: int | None
    best_rank: int | None
    players: int | None
    peak_radius: float
    modes: dict
    kills: int = 0
    deaths_seen: int = 0          # snakes that died in view (kills by anyone)
    loot_zones: int = 0

    @property
    def length(self) -> int | None:
        """The death screen's 'final length' is clean text; the in-game HUD is only a
        fallback (overnight logs: peak misreads up to 29301 for a final length of 391)."""
        return self.final_length or self.peak_length

    def score(self) -> float:
        if self.length:
            return math.log(max(self.length, 1))
        # vision proxy without OCR: thickness grows with length, plus survival
        return math.log(1.0 + 10.0 * self.peak_radius ** 2 + self.duration_s)


# old logs (before v3) have only the first 12 columns; csv.DictReader leaves the rest empty
LOG_COLUMNS = ["time", "round", "role", "score", "final_length", "peak_length", "best_rank",
               "players", "duration_s", "peak_radius", "modes", "params",
               "kills", "deaths_seen", "loot_zones", "version"]


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("-inf")


class Tuner:
    def __init__(self, cfg: TrainConfig, start_params: dict, seed: int | None = None,
                 version: int = 1) -> None:
        self.cfg = cfg
        self.dir = Path(cfg.state_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.dir / "state.json"
        self.log_path = self.dir / "episodes.csv"
        self.rng = random.Random(seed)
        names = {p.name for p in SPACE}
        start = {k: v for k, v in start_params.items() if k in names}
        self.state = {
            "incumbent": start, "incumbent_scores": [],
            "candidate": None, "candidate_scores": [],
            "sigma": cfg.sigma_start, "episodes": 0, "accepted": 0, "rejected": 0,
            "best_length": 0, "best_rank": None, "current_role": "incumbent",
            "version": version, "lengths": [],
        }
        self.notice = ""
        if self.log_path.exists():
            with self.log_path.open(encoding="utf-8") as f:
                header = f.readline().strip().split(",")
            if header != LOG_COLUMNS:            # older log format: keep it, start a new file
                archive = self.dir / f"episodes_old_{time.strftime('%Y%m%d_%H%M%S')}.csv"
                self.log_path.rename(archive)
                self.notice += f"[train] old log archived as {archive.name}\n"
        if self.state_path.exists():
            self.state.update(json.loads(self.state_path.read_text(encoding="utf-8")))
            if self.state.get("version") != version:
                # the bot itself changed: old scores say nothing about the new behaviour,
                # so start a fresh baseline from the new profile (history stays in the CSV)
                self.state.update(incumbent=start, incumbent_scores=[], candidate=None,
                                  candidate_scores=[], sigma=cfg.sigma_start, version=version,
                                  lengths=[], current_role="incumbent", best_length=0,
                                  best_rank=None, accepted=0, rejected=0)
                best = self.dir / "best_params.json"
                if best.exists():
                    best.replace(self.dir / "best_params_old.json")
                self.notice += (f"[train] new bot version {version}: tuning starts a fresh baseline "
                                "from the new profile (old rounds stay in the archived log)")
            # parameters added in a newer version: start them at the current defaults
            for role in ("incumbent", "candidate"):
                if self.state[role] is not None:
                    for k, v in start.items():
                        self.state[role].setdefault(k, v)

    # ------------------------------------------------------------ public api
    def next_params(self) -> tuple[str, dict]:
        """Which parameter set to play next round: ('incumbent'|'candidate', params)."""
        s = self.state
        if len(s["incumbent_scores"]) < self.cfg.episodes_per_candidate:
            role = "incumbent"                      # first build a baseline
        elif s["episodes"] % self.cfg.incumbent_every == self.cfg.incumbent_every - 1:
            role = "incumbent"
        else:
            if s["candidate"] is None:
                s["candidate"] = self._mutate(s["incumbent"])
                s["candidate_scores"] = []
            role = "candidate"
        s["current_role"] = role
        self._save()
        return role, dict(s[role])

    def report(self, ep: Episode) -> str:
        s = self.state
        role = s["current_role"]
        score = ep.score()
        s["episodes"] += 1
        length = ep.length or 0
        s["best_length"] = max(s["best_length"], length)
        if ep.best_rank is not None and (s["best_rank"] is None or ep.best_rank < s["best_rank"]):
            s["best_rank"] = ep.best_rank
        self._log(ep, role, score)

        verdict = ""
        if role == "incumbent":
            s["incumbent_scores"] = (s["incumbent_scores"] + [score])[-12:]
        else:
            s["candidate_scores"].append(score)
            verdict = self._judge()
        s["lengths"] = (s.get("lengths", []) + [length])[-500:]
        self._save()
        rank = f"{ep.best_rank}/{ep.players}" if ep.best_rank else "?"
        return (f"[train] round {s['episodes']} ({role}) length={length or '?'} rank={rank} "
                f"kills={ep.kills} loot={ep.loot_zones} "
                f"time={ep.duration_s / 60:.1f}min score={score:.2f} | incumbent mean="
                f"{_mean(s['incumbent_scores']):.2f} sigma={s['sigma']:.3f} "
                f"best length={s['best_length']} best rank={s['best_rank']} {verdict}\n"
                f"        {behaviour_line(ep.modes)}\n        {self.trend_line()}")

    def trend_line(self) -> str:
        """Is it getting better? Median of the last 10 rounds vs. the 10 before and the start."""
        ls = [x for x in self.state.get("lengths", []) if x]
        if len(ls) < 6:
            return f"Lernkurve: noch zu wenig Runden ({len(ls)}), ab ~20 wird es aussagekräftig"

        def med(xs):
            xs = sorted(xs)
            return xs[len(xs) // 2]
        recent, first = ls[-10:], ls[:10]
        line = f"Lernkurve: Median letzte {len(recent)} Runden = {med(recent)}, erste {len(first)} = {med(first)}"
        if len(ls) >= 20:
            change = 100 * (med(recent) / max(med(first), 1) - 1)
            line += f" ({change:+.0f}%)"
        return line + f", {self.state['accepted']} Verbesserungen übernommen"

    def describe(self, role: str) -> str:
        return f"[train] {role}: {style_line(self.state[role] or self.state['incumbent'])}"

    @property
    def incumbent(self) -> dict:
        return dict(self.state["incumbent"])

    # ------------------------------------------------------------ internals
    def _judge(self) -> str:
        s, cfg = self.state, self.cfg
        cand, inc = s["candidate_scores"], s["incumbent_scores"]
        median_inc = sorted(inc)[len(inc) // 2] if inc else 0.0
        # scores are logs: "length < early_reject * median length" <=> score < median + log(early_reject)
        if len(cand) == 1 and inc and cand[0] < median_inc + math.log(cfg.early_reject):
            return self._reject("early reject")
        if len(cand) < cfg.episodes_per_candidate:
            return f"candidate {len(cand)}/{cfg.episodes_per_candidate}"
        if _mean(cand) > _mean(inc) + cfg.accept_margin:
            s["incumbent"] = s["candidate"]
            s["incumbent_scores"] = list(cand)
            s["candidate"], s["candidate_scores"] = None, []
            s["accepted"] += 1
            s["sigma"] = min(s["sigma"] * math.exp(0.8 / 3), cfg.sigma_max)
            (self.dir / "best_params.json").write_text(json.dumps(s["incumbent"], indent=2), encoding="utf-8")
            return "-> ACCEPTED, new best parameter set"
        return self._reject("rejected")

    def _reject(self, why: str) -> str:
        s = self.state
        s["candidate"], s["candidate_scores"] = None, []
        s["rejected"] += 1
        s["sigma"] = max(s["sigma"] * math.exp(-0.2 / 3), self.cfg.sigma_min)
        return f"-> {why}"

    def _mutate(self, params: dict) -> dict:
        out = dict(params)
        sigma = self.state["sigma"]
        # mutate a random subset: big jumps in few dimensions explore better
        k = max(2, int(len(SPACE) * self.rng.uniform(0.2, 0.5)))
        for p in self.rng.sample(SPACE, k):
            u = p.to_unit(float(params.get(p.name, p.from_unit(0.5))))
            out[p.name] = p.from_unit(u + self.rng.gauss(0.0, sigma))
        return out

    def _save(self) -> None:
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=2), encoding="utf-8")
        tmp.replace(self.state_path)

    def _log(self, ep: Episode, role: str, score: float) -> None:
        new = not self.log_path.exists()
        with self.log_path.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(LOG_COLUMNS)
            w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), self.state["episodes"], role,
                        f"{score:.3f}", ep.final_length, ep.peak_length, ep.best_rank, ep.players,
                        f"{ep.duration_s:.1f}", f"{ep.peak_radius:.1f}", json.dumps(ep.modes),
                        json.dumps(self.state[role]), ep.kills, ep.deaths_seen, ep.loot_zones,
                        self.state.get("version")])
