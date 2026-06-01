#!/usr/bin/env python3
"""Define which regions of the camera view are parking spots.

Grabs a fresh snapshot from your Blink camera, then opens a window where you draw
a box around each parking spot. The boxes are written to config.json.

Usage:
    python calibrate.py [camera_name]

If camera_name is omitted, the first camera on the account is used.
Controls: drag a box, press ENTER/SPACE to confirm it, draw the next one,
then press ESC when you're done.
"""
import asyncio
import json
import sys

import cv2

from parking_tracker.blink_client import capture_snapshot, connect
from parking_tracker.config import CONFIG_PATH, CREDS_PATH, DATA_DIR

DEFAULT_CONFIG = {
    "camera_name": "",
    "poll_interval_seconds": 60,
    "confidence_threshold": 0.35,
    "overlap_threshold": 0.15,
    "debounce_frames": 2,
    "vehicle_classes": ["car", "truck", "bus", "motorcycle"],
    "model_path": "yolov8n.pt",
    "snapshot_file": "latest.jpg",
    "spots": [],
}


async def _grab_snapshot(camera_name: str):
    blink, session = await connect(CREDS_PATH)
    try:
        if not camera_name:
            camera_name = list(blink.cameras)[0]
        path = DATA_DIR / "calibration.jpg"
        await capture_snapshot(blink, camera_name, path)
        return camera_name, path
    finally:
        await session.close()


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    requested = sys.argv[1] if len(sys.argv) > 1 else ""

    print("Grabbing a snapshot from your camera...")
    camera_name, image_path = asyncio.run(_grab_snapshot(requested))

    image = cv2.imread(str(image_path))
    if image is None:
        raise SystemExit(f"Could not read snapshot at {image_path}")

    print(
        "\nDraw a box around each parking spot.\n"
        "  • Drag to draw, then press ENTER or SPACE to confirm each box\n"
        "  • Press ESC when you've marked every spot\n"
    )
    rois = cv2.selectROIs("Mark parking spots (ESC when done)", image, showCrosshair=False)
    cv2.destroyAllWindows()

    if len(rois) == 0:
        raise SystemExit("No spots selected — nothing saved.")

    config = json.loads(CONFIG_PATH.read_text()) if CONFIG_PATH.exists() else dict(DEFAULT_CONFIG)
    config["camera_name"] = camera_name
    config["spots"] = [
        {"id": f"spot-{i + 1}", "box": [int(x), int(y), int(w), int(h)]}
        for i, (x, y, w, h) in enumerate(rois)
    ]
    CONFIG_PATH.write_text(json.dumps(config, indent=2) + "\n")

    print(f"\n✅ Saved {len(rois)} spot(s) to {CONFIG_PATH}")
    print("Next: run `python run.py` to start watching.")


if __name__ == "__main__":
    main()
