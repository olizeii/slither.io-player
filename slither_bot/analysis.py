"""Evaluation of the training logs (tuning/episodes*.csv).

Used by ``tools/evaluate.py``. Everything works on plain dicts so old log
formats (before v3: no kills/version columns) load just as well.

Cleaning matters: the overnight run of v2 contained OCR misreads (peak length
29301 for a final length of 391, rank 0, "rank 1 of 247" at length 151).
Such rows are flagged and never used for "best run" rankings.
"""
from __future__ import annotations

import csv
import json
import math
import statistics as st
from dataclasses import dataclass, field
from pathlib import Path

from .tuner import PARAMS, style, STYLE_NAMES


@dataclass
class Run:
    file: str
    round: int
    time: str
    role: str
    length: int | None
    final_length: int | None
    peak_length: int | None
    rank: int | None
    players: int | None
    duration_s: float
    kills: int | None
    deaths_seen: int | None
    loot_zones: int | None
    version: str
    phases: dict
    params: dict
    flags: list = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not any(f.startswith("length") for f in self.flags)

    @property
    def style_type(self) -> str:
        if not self.params:
            return "-"
        s = style(self.params)
        top = max(("Aggression", "Vorsicht", "Gier", "Loot"), key=lambda k: s[k])
        return f"{STYLE_NAMES[top]} (A{s['Aggression']}/V{s['Vorsicht']})"


def _int(x) -> int | None:
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return None


def _float(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def load_runs(directory: str | Path, include_old: bool = True) -> list[Run]:
    d = Path(directory)
    files = sorted(d.glob("episodes_old_*.csv")) if include_old else []
    if (d / "episodes.csv").exists():
        files.append(d / "episodes.csv")
    runs: list[Run] = []
    for f in files:
        with f.open(encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                final, peak = _int(row.get("final_length")), _int(row.get("peak_length"))
                run = Run(
                    file=f.name, round=_int(row.get("round")) or 0, time=row.get("time", ""),
                    role=row.get("role", ""), length=final or peak, final_length=final,
                    peak_length=peak, rank=_int(row.get("best_rank")), players=_int(row.get("players")),
                    duration_s=_float(row.get("duration_s")), kills=_int(row.get("kills")),
                    deaths_seen=_int(row.get("deaths_seen")), loot_zones=_int(row.get("loot_zones")),
                    version=row.get("version") or ("v2 (old log)" if "old" in f.name else "?"),
                    phases=json.loads(row.get("modes") or "{}"),
                    params=json.loads(row.get("params") or "{}"),
                )
                flag(run)
                runs.append(run)
    return runs


def flag(run: Run, max_growth_per_s: float = 40.0) -> None:
    """Mark physically impossible values (OCR misreads)."""
    if run.length is None:
        run.flags.append("length missing")
    elif run.length < 10:                               # every round starts at length 10
        run.flags.append("length misread (below start length)")
    elif run.length > max_growth_per_s * max(run.duration_s, 1.0) + 50:
        run.flags.append("length impossible for survival time")
    if run.final_length and run.peak_length and run.peak_length > 3 * run.final_length + 50:
        run.flags.append("peak misread (ignored, final used)")
    if run.rank is not None and (run.rank < 1 or (run.players and run.rank > run.players)):
        run.flags.append("rank misread")
        run.rank = None
    elif (run.rank is not None and run.rank <= 10 and run.players and run.players > 50
          and (run.length or 0) < 300):
        run.flags.append("rank implausible")            # top 10 of a full server with length < 300?
        run.rank = None


def _median(xs):
    xs = [x for x in xs if x is not None]
    return st.median(xs) if xs else None


def _pct(xs, q):
    xs = sorted(x for x in xs if x is not None)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else None


def spearman(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 3:
        return 0.0

    def ranks(v):
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        for pos, i in enumerate(order):
            r[i] = pos
        return r
    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    vy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return cov / (vx * vy) if vx and vy else 0.0


def param_effects(runs: list[Run], min_runs: int = 30) -> list[tuple[str, float]]:
    """Which knob goes along with longer rounds? (Spearman: parameter vs. log length)"""
    good = [r for r in runs if r.clean and r.length and r.params]
    if len(good) < min_runs:
        return []
    out = []
    for name in PARAMS:
        xs = [float(r.params[name]) for r in good if name in r.params]
        ys = [math.log(r.length) for r in good if name in r.params]
        if len(xs) >= min_runs and len(set(xs)) > 3:
            out.append((name, spearman(xs, ys)))
    return sorted(out, key=lambda kv: -abs(kv[1]))


def phase_shares(runs: list[Run]) -> dict[str, float]:
    tot: dict[str, float] = {}
    for r in runs:
        for k, v in r.phases.items():
            tot[k] = tot.get(k, 0.0) + v
    s = sum(tot.values()) or 1.0
    return {k: 100 * v / s for k, v in sorted(tot.items(), key=lambda kv: -kv[1])}


def learning_curve(runs: list[Run], block: int = 50) -> list[tuple[int, float, float]]:
    """(first round of block, median length, median survival) per block of rounds."""
    good = [r for r in runs if r.clean and r.length]
    out = []
    for i in range(0, len(good), block):
        chunk = good[i:i + block]
        if len(chunk) >= max(5, block // 3):
            out.append((i + 1, _median([r.length for r in chunk]), _median([r.duration_s for r in chunk])))
    return out


def summary(runs: list[Run], top: int = 10) -> str:
    if not runs:
        return "no rounds found"
    good = [r for r in runs if r.clean and r.length]
    lines = []
    hours = sum(r.duration_s for r in runs) / 3600
    lines.append(f"Runden: {len(runs)}  (davon {len(runs) - len(good)} mit unplausiblen Werten, "
                 f"ignoriert)  Spielzeit: {hours:.1f} h")
    versions = sorted({r.version for r in runs})
    lines.append(f"Versionen im Log: {', '.join(versions)}")
    ls = [r.length for r in good]
    du = [r.duration_s for r in good]
    lines.append(f"Länge:      Median {_median(ls)}  | 75% {_pct(ls, .75)} | 90% {_pct(ls, .9)} | max {max(ls)}")
    lines.append(f"Überleben:  Median {_median(du):.0f} s | 90% {_pct(du, .9):.0f} s | max {max(du):.0f} s")
    ranks = [r.rank for r in good if r.rank]
    if ranks:
        lines.append(f"Bester Rang: {min(ranks)}  (Runden in Top 10: {sum(x <= 10 for x in ranks)})")
    kills = [r.kills for r in runs if r.kills is not None]
    if kills:
        lines.append(f"Kills: {sum(kills)} gesamt, {sum(kills) / len(kills):.2f} pro Runde, "
                     f"max {max(kills)} in einer Runde")
    else:
        lines.append("Kills: (erst ab v3 protokolliert)")

    lines.append(f"\nTOP {top} Runden (nach Länge, nur plausible Werte):")
    lines.append(f"  {'Runde':>6} {'Zeit':<19} {'Länge':>6} {'Rang':>9} {'Leben':>7} {'Kills':>5}  "
                 f"{'Rolle':<9} {'Spielstil':<28} Phasen")
    for r in sorted(good, key=lambda r: -r.length)[:top]:
        tot = sum(r.phases.values()) or 1
        ph = " ".join(f"{k}{100 * v / tot:.0f}" for k, v in sorted(r.phases.items(), key=lambda kv: -kv[1])[:3])
        rank = f"{r.rank}/{r.players}" if r.rank else "-"
        kills = "-" if r.kills is None else str(r.kills)
        lines.append(f"  {r.round:>6} {r.time:<19} {r.length:>6} {rank:>9} {r.duration_s:>6.0f}s {kills:>5}  "
                     f"{r.role:<9} {r.style_type:<28} {ph}")

    curve = learning_curve(runs)
    if curve:
        lines.append("\nLernkurve (Blöcke à 50 Runden): " + "  ".join(
            f"#{start}: {med:.0f} / {surv:.0f}s" for start, med, surv in curve))
    best10 = sorted(good, key=lambda r: -r.length)[:max(1, len(good) // 10)]
    lines.append("\nZeitanteile beste 10%: " + "  ".join(f"{k} {v:.0f}%" for k, v in list(phase_shares(best10).items())[:6]))
    lines.append("Zeitanteile alle:      " + "  ".join(f"{k} {v:.0f}%" for k, v in list(phase_shares(good).items())[:6]))
    eff = param_effects(runs)
    if eff:
        lines.append("\nWas geht mit längeren Runden einher (Spearman, |r|>0.1 ist schon auffällig):")
        for name, r in eff[:8]:
            lines.append(f"  {name:<28} {r:+.2f}  {'mehr ist besser' if r > 0 else 'weniger ist besser'}")
    flagged = [r for r in runs if r.flags]
    if flagged:
        kinds: dict[str, int] = {}
        for r in flagged:
            for f in r.flags:
                kinds[f] = kinds.get(f, 0) + 1
        lines.append("\nDatenqualität: " + ", ".join(f"{k}: {v}" for k, v in kinds.items()))
    return "\n".join(lines)


def plot(runs: list[Run], out: str | Path, show: bool = False) -> Path:
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    good = [r for r in runs if r.clean and r.length]
    bad = [r for r in runs if not r.clean and r.length]
    idx = {id(r): i for i, r in enumerate(runs)}
    fig, ax = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("slither.io bot - Trainingsauswertung", fontsize=14)

    a = ax[0][0]
    a.scatter([idx[id(r)] for r in good], [r.length for r in good], s=8, alpha=0.5, label="Runde")
    if bad:
        a.scatter([idx[id(r)] for r in bad], [min(r.length, 1e5) for r in bad], s=14, marker="x",
                  color="red", label="unplausibel (ignoriert)")
    win = 25
    if len(good) >= win:
        xs = [idx[id(r)] for r in good]
        med = [_median([g.length for g in good[max(0, i - win):i + 1]]) for i in range(len(good))]
        a.plot(xs, med, color="black", lw=2, label=f"gleitender Median ({win})")
    a.set_yscale("log")
    a.set_title("Länge pro Runde")
    a.set_xlabel("Runde")
    a.legend()

    a = ax[0][1]
    a.hist([r.duration_s for r in good], bins=30, color="tab:green")
    a.set_title("Überlebenszeit")
    a.set_xlabel("Sekunden")

    a = ax[1][0]
    best10 = sorted(good, key=lambda r: -r.length)[:max(1, len(good) // 10)]
    sb, sa = phase_shares(best10), phase_shares(good)
    keys = list(sa)[:8]
    xs = range(len(keys))
    a.bar([x - 0.2 for x in xs], [sb.get(k, 0) for k in keys], 0.4, label="beste 10%")
    a.bar([x + 0.2 for x in xs], [sa.get(k, 0) for k in keys], 0.4, label="alle")
    a.set_xticks(list(xs))
    a.set_xticklabels(keys, rotation=30)
    a.set_ylabel("% der Zeit")
    a.set_title("Was macht der Bot? (Zeitanteile)")
    a.legend()

    a = ax[1][1]
    eff = param_effects(runs)[:12]
    if eff:
        a.barh([n for n, _ in eff][::-1], [v for _, v in eff][::-1],
               color=["tab:green" if v > 0 else "tab:red" for _, v in eff][::-1])
        a.axvline(0, color="black", lw=0.8)
        a.set_title("Parameter vs. Länge (Spearman)")
    else:
        a.text(0.5, 0.5, "zu wenig Runden für Korrelationen", ha="center", va="center")
        a.set_axis_off()

    fig.tight_layout()
    out = Path(out)
    fig.savefig(out, dpi=110)
    if show:
        plt.show()
    plt.close(fig)
    return out
