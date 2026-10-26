"""Interactive HSV calibration with trackbars.

    python tools/calibrate.py                 # live capture of the slither.io window
    python tools/calibrate.py --image shot.png

Keys:
  1..5  choose which range to tune: lit, white, eye, border, minimap dot
  s     save the current frame to captures/
  w     write all ranges to config.json
  q     quit
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slither_bot import debug as debug_view          # noqa: E402
from slither_bot import win32                        # noqa: E402
from slither_bot.capture import LatestFrameBuffer, ScreenCapture   # noqa: E402
from slither_bot.config import load_config, save_config            # noqa: E402
from slither_bot.main import locate_game             # noqa: E402
from slither_bot.vision import Vision                # noqa: E402

RANGES = [("lit", "lit_lower", "lit_upper"),
          ("white", "white_lower", "white_upper"),
          ("eye", "eye_lower", "eye_upper"),
          ("border", "border_lower1", "border_upper1"),
          ("minimap dot", "minimap_dot_lower", "minimap_dot_upper")]
BARS = [("H lo", 179), ("S lo", 255), ("V lo", 255), ("H hi", 179), ("S hi", 255), ("V hi", 255)]
CTRL = "calibrate"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", help="tune on a screenshot instead of live capture")
    ap.add_argument("--config", default="config.json")
    args = ap.parse_args()
    cfg = load_config(args.config)

    capture = None
    if args.image:
        static = cv2.imread(args.image)
        if static is None:
            raise SystemExit(f"cannot read {args.image}")
        if cfg.capture.scale != 1.0:
            static = cv2.resize(static, None, fx=cfg.capture.scale, fy=cfg.capture.scale,
                                interpolation=cv2.INTER_AREA)
    else:
        win32.enable_dpi_awareness()
        _, region = locate_game(cfg)
        buffer = LatestFrameBuffer()
        capture = ScreenCapture(region, cfg.capture, buffer)
        capture.start()

    cv2.namedWindow(CTRL, cv2.WINDOW_NORMAL)
    for name, maximum in BARS:
        cv2.createTrackbar(name, CTRL, 0, maximum, lambda _v: None)
    current = 0

    def load_bars(i: int) -> None:
        _, lo, hi = RANGES[i]
        values = list(getattr(cfg.vision, lo)) + list(getattr(cfg.vision, hi))
        for (name, _), v in zip(BARS, values):
            cv2.setTrackbarPos(name, CTRL, int(v))
        print(f"tuning: {RANGES[i][0]}")

    load_bars(current)
    last_id = 0
    frame = None
    while True:
        if capture is not None:
            f = buffer.wait_newer(last_id, 0.2)
            if f is not None:
                last_id, frame = f.frame_id, f.image
        else:
            frame = static
        if frame is None:
            continue

        values = [cv2.getTrackbarPos(name, CTRL) for name, _ in BARS]
        _, lo, hi = RANGES[current]
        setattr(cfg.vision, lo, tuple(values[:3]))
        setattr(cfg.vision, hi, tuple(values[3:]))

        vision = Vision(cfg.vision)
        state = vision.analyze(frame, time.perf_counter())
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, tuple(values[:3]), tuple(values[3:]))
        cv2.imshow("mask: " + "/".join(r[0] for r in RANGES), mask)
        cv2.imshow(debug_view.WINDOW, debug_view.render(frame, state, None, f"tuning {RANGES[current][0]}"))

        key = cv2.waitKey(15) & 0xFF
        if key == ord("q"):
            break
        if ord("1") <= key <= ord("5"):
            current = key - ord("1")
            load_bars(current)
        elif key == ord("s"):
            out = Path("captures")
            out.mkdir(exist_ok=True)
            path = out / f"frame_{int(time.time())}.png"
            cv2.imwrite(str(path), frame)
            print(f"saved {path}")
        elif key == ord("w"):
            save_config(cfg, args.config)
            print(f"wrote {args.config}")

    if capture is not None:
        capture.stop()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
