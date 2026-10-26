import csv
import json

from slither_bot.analysis import load_runs, plot, spearman, summary
from slither_bot.tuner import LOG_COLUMNS

OLD_COLUMNS = LOG_COLUMNS[:12]          # v2 log format, as in the overnight run


def write(path, columns, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(columns)
        for r in rows:
            w.writerow([r.get(c, "") for c in columns])


def row(i, final, peak, dur, rank=30, players=200, kills=None):
    return {"time": f"2026-10-26 03:{i:02d}:00", "round": i, "role": "candidate", "score": 1,
            "final_length": final, "peak_length": peak, "best_rank": rank, "players": players,
            "duration_s": dur, "peak_radius": 9, "modes": json.dumps({"forage": dur * 0.6, "escape": dur * 0.4}),
            "params": json.dumps({"w_danger": 2 + i % 5}), "kills": kills, "version": 3}


def test_misreads_are_flagged_and_final_length_wins(tmp_path):
    write(tmp_path / "episodes_old_x.csv", OLD_COLUMNS, [
        row(1, 391, 29301, 66),           # the overnight misread
        row(2, 120, 125, 40, rank=0),     # rank 0
        row(3, 1, 118, 35),               # final misread
        row(4, 150, 151, 30, rank=1, players=247),
    ])
    runs = load_runs(tmp_path)
    by = {r.round: r for r in runs}
    assert by[1].length == 391 and by[1].clean
    assert by[2].rank is None
    assert not by[3].clean
    assert by[4].rank is None             # top rank on a full server with length 150: implausible
    text = summary(runs)
    assert "29301" not in text and "391" in text


def test_new_and_old_logs_load_together_and_plot(tmp_path):
    write(tmp_path / "episodes_old_x.csv", OLD_COLUMNS, [row(i, 50 + i, 50 + i, 30) for i in range(40)])
    write(tmp_path / "episodes.csv", LOG_COLUMNS, [row(i, 100 + 3 * i, 100 + 3 * i, 60, kills=i % 3) for i in range(40)])
    runs = load_runs(tmp_path)
    assert len(runs) == 80
    text = summary(runs)
    assert "Kills:" in text and "TOP 10" in text
    out = plot(runs, tmp_path / "report.png")
    assert out.exists() and out.stat().st_size > 10_000


def test_spearman():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) > 0.99
    assert spearman([1, 2, 3, 4], [40, 30, 20, 10]) < -0.99
