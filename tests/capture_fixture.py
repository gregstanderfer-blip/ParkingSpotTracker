#!/usr/bin/env python3
"""Capture a detection test fixture from the live camera.

Grabs the current frame, saves it under tests/fixtures/<LABEL>.jpg, and records
the expected per-spot occupancy in tests/fixtures/expected.json so
tests/test_detection.py can check detection against it later.

Usage:
    python tests/capture_fixture.py LABEL [SPOT_STATES ...]

    LABEL         e.g. evening, night, morning
    SPOT_STATES   optional ground-truth per spot, in spot order:
                  "occupied"/"empty" (or o/e). If omitted, the detector's current
                  reading is recorded — so eyeball it and pass explicit states if
                  the detector is wrong at capture time (e.g. at night):
                      python tests/capture_fixture.py night occupied occupied
"""
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")

import config       # noqa: E402
import monitor      # noqa: E402

FIX = Path(__file__).resolve().parent / "fixtures"


def _truthy(v):
    return str(v).lower() in ("o", "occ", "occupied", "true", "1", "yes")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    label = sys.argv[1]
    overrides = sys.argv[2:]
    FIX.mkdir(parents=True, exist_ok=True)

    # Load spot polygons (prefer the committed test copy, else the live one).
    spots_path = FIX / "spots.json"
    if not spots_path.exists():
        spots_path = Path(config.SPOTS_FILE)
    spots = json.loads(Path(spots_path).read_text())
    for s in spots:
        s["polygon_np"] = np.array(s["polygon"], np.int32)

    # Grab a settled frame.
    cap = cv2.VideoCapture(config.RTSP_URL)
    frame = None
    for _ in range(12):
        ok, f = cap.read()
        if ok and f is not None:
            frame = f
    cap.release()
    if frame is None:
        print("ERROR: could not read a frame from the camera.")
        sys.exit(1)
    frame = monitor.resize_for_proc(frame)

    from ultralytics import YOLO
    model = YOLO(config.MODEL)
    occ = monitor.spot_occupancy(monitor.detect_vehicle_boxes(model, frame), spots)

    # Expected ground truth: overrides win, else use what the detector sees.
    expected = {}
    used_override = False
    for i, s in enumerate(spots):
        if i < len(overrides):
            expected[s["name"]] = _truthy(overrides[i])
            used_override = True
        else:
            expected[s["name"]] = bool(occ[s["name"]])

    # Save image + (first time) a committed copy of the spot polygons.
    cv2.imwrite(str(FIX / f"{label}.jpg"), frame)
    if not (FIX / "spots.json").exists():
        (FIX / "spots.json").write_text(json.dumps(
            [{"name": s["name"], "polygon": s["polygon"]} for s in spots], indent=2) + "\n")

    manifest = FIX / "expected.json"
    data = json.loads(manifest.read_text()) if manifest.exists() else []
    data = [e for e in data if e.get("label") != label]  # replace same label
    data.append({"label": label, "image": f"{label}.jpg", "spots": expected})
    manifest.write_text(json.dumps(data, indent=2) + "\n")

    print(f"Saved tests/fixtures/{label}.jpg")
    print(f"  detector saw : {occ}")
    print(f"  recorded     : {expected}")
    if not used_override:
        print("  (recorded the detector's reading as ground truth — if that's "
              "wrong, re-run with explicit states, e.g. `... %s occupied occupied`)" % label)


if __name__ == "__main__":
    main()
