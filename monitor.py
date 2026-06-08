"""
monitor.py — watch parking spots and text you when one changes state.

    python monitor.py

What it does:
  - POLLS the Tapo C120 RTSP stream every DETECT_INTERVAL_SECONDS: it grabs a
    single frame and disconnects, rather than holding the stream open. This keeps
    WiFi bandwidth tiny (a couple of frames every ~15s instead of a 24/7 2K feed),
    which also avoids the stream corruption that long WiFi sessions can cause.
  - Runs a YOLO model to find vehicles (car / truck / bus / motorcycle).
  - Decides each spot is OCCUPIED or EMPTY via a point-in-polygon test.
  - When a spot changes state (and it holds for CONFIRM_SECONDS), it texts you
    via the Messages app and saves an annotated snapshot to snapshots/.

Stop any time with Ctrl-C.
"""

import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime

import cv2
import numpy as np

import config

# Use RTSP-over-TCP (more reliable than UDP) and silence FFmpeg's chatty
# H264 decode warnings.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")

try:
    from ultralytics import YOLO
except ImportError:
    print("Missing dependency. Run:  pip install -r requirements.txt")
    sys.exit(1)

# COCO class IDs that count as "a vehicle in the spot".
VEHICLE_CLASS_IDS = {2, 3, 5, 7}  # car, motorcycle, bus, truck


def resize_for_proc(frame):
    """Downscale a frame to config.PROC_WIDTH (keeps detection light and makes
    spot coordinates match select_spots.py, which uses the same width)."""
    h, w = frame.shape[:2]
    if w == config.PROC_WIDTH:
        return frame
    return cv2.resize(frame, (config.PROC_WIDTH, round(h * config.PROC_WIDTH / w)))


def _open_capture(url):
    """Open an RTSP capture with short open/read timeouts (passed at construction,
    where they actually take effect), so a WiFi stall fails in ~5s instead of ~30s."""
    params = []
    ot = getattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC", None)
    rt = getattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC", None)
    if ot is not None:
        params += [int(ot), 8000]
    if rt is not None:
        params += [int(rt), 5000]
    if params:
        try:
            return cv2.VideoCapture(url, cv2.CAP_FFMPEG, params)
        except Exception:
            pass
    return cv2.VideoCapture(url)


def grab_frame(url, warmup=10):
    """Connect, read a few frames (the first decoded frame after a fresh connect
    is keyframe-based and clean; we read a few to get the most recent one), keep
    the last good frame, and disconnect. Returns the frame, or None if the stream
    couldn't be read. Polling like this — instead of holding the stream open —
    is what keeps bandwidth minimal."""
    cap = _open_capture(url)
    try:
        if not cap.isOpened():
            return None
        frame = None
        for _ in range(warmup):
            ok, f = cap.read()
            if ok and f is not None:
                frame = f
        return frame
    finally:
        cap.release()


def send_imessage(to, text):
    """Send an iMessage through the macOS Messages app via AppleScript."""
    safe = text.replace("\\", "\\\\").replace('"', '\\"')
    script = (
        'tell application "Messages"\n'
        '    set targetService to 1st account whose service type = iMessage\n'
        f'    set targetBuddy to participant "{to}" of targetService\n'
        f'    send "{safe}" to targetBuddy\n'
        'end tell\n'
    )
    try:
        subprocess.run(["osascript", "-e", script], check=True,
                       capture_output=True, timeout=20)
        print(f"  -> alert sent to {to}")
    except subprocess.CalledProcessError as e:
        print(f"  !! Messages send failed: {e.stderr.decode().strip()}")
    except Exception as e:
        print(f"  !! Messages send error: {e}")


def load_spots(path):
    if not os.path.exists(path):
        print(f"No {path} found. Run:  python select_spots.py")
        sys.exit(1)
    with open(path) as f:
        spots = json.load(f)
    for s in spots:
        s["polygon_np"] = np.array(s["polygon"], np.int32)
    return spots


def box_in_spot(box, poly_np):
    """True if the vehicle box meaningfully overlaps the spot polygon. We test the
    box center plus its bottom-center (where the car meets the ground) — robust to
    a car sticking up out of the polygon."""
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    bottom = (cx, y2 - (y2 - y1) * 0.15)
    for pt in ((cx, cy), bottom):
        if cv2.pointPolygonTest(poly_np, pt, False) >= 0:
            return True
    return False


def detect_vehicle_boxes(model, frame, device=None, conf=None, imgsz=None):
    """Run YOLO on a frame and return vehicle boxes [x1,y1,x2,y2] that clear the
    confidence floor. ``conf``/``imgsz`` default to the values in config."""
    conf = config.MIN_CONFIDENCE if conf is None else conf
    imgsz = config.PROC_WIDTH if imgsz is None else imgsz
    res = model(frame, verbose=False, imgsz=imgsz, device=device)[0]
    boxes = []
    for b in res.boxes:
        if int(b.cls[0]) in VEHICLE_CLASS_IDS and float(b.conf[0]) >= conf:
            boxes.append(b.xyxy[0].tolist())
    return boxes


def spot_occupancy(boxes, spots):
    """Map each spot name -> bool occupied, given vehicle boxes. Each spot needs
    a ``polygon_np`` (np.int32 array)."""
    return {s["name"]: any(box_in_spot(b, s["polygon_np"]) for b in boxes)
            for s in spots}


def log_event(spot_name, new_state):
    new_file = not os.path.exists(config.LOG_FILE)
    with open(config.LOG_FILE, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["timestamp", "spot", "new_state"])
        w.writerow([datetime.now().isoformat(timespec="seconds"),
                    spot_name, new_state])


def save_snapshot(frame, spots, spot_name, new_state):
    img = frame.copy()
    for s in spots:
        color = (0, 0, 255) if s["occupied"] else (0, 255, 0)
        cv2.polylines(img, [s["polygon_np"]], True, color, 2)
    now = datetime.now()
    # Sort snapshots into snapshots/YYYY/MM/DD/ for easy browsing.
    day_dir = os.path.join(config.SNAPSHOT_DIR,
                           now.strftime("%Y"), now.strftime("%m"), now.strftime("%d"))
    os.makedirs(day_dir, exist_ok=True)
    safe = spot_name.replace(" ", "_")
    path = os.path.join(day_dir, f"{now:%H%M%S}_{safe}_{new_state}.jpg")
    cv2.imwrite(path, img)
    return path


def main():
    # Show logs live instead of block-buffering them when stdout isn't a terminal.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    spots = load_spots(config.SPOTS_FILE)
    print(f"Loaded {len(spots)} spot(s) from {config.SPOTS_FILE}")

    print(f"Loading model {config.MODEL} (downloads on first run)...")
    model = YOLO(config.MODEL)

    # Run detection on CPU, not MPS (Apple GPU). MPS was ~80ms faster but produced
    # intermittent WRONG results in long-running (multi-hour) sessions: a clearly
    # visible car would read as undetected for a stretch, firing false "empty"
    # alerts — while the very same frame detected correctly in a fresh process / on
    # CPU. At one poll per 15s, CPU's ~210ms is irrelevant, and it's stable.
    device = "cpu"
    print(f"Running detection on: {device}")
    print(f"Polling every {config.DETECT_INTERVAL_SECONDS:g}s "
          f"(one frame per check — no continuous streaming).")

    # Per-spot state machine.
    for s in spots:
        s["occupied"] = None          # confirmed state (None until first read)
        s["candidate"] = None         # state we're waiting to confirm
        s["candidate_since"] = 0.0

    print("Watching... (Ctrl-C to stop)\n")
    misses = 0
    try:
        while True:
            frame = grab_frame(config.RTSP_URL)
            if frame is None:
                # Couldn't get a frame (WiFi/stream issue). Hold state, no alert.
                misses += 1
                if misses == 1 or misses % 10 == 0:
                    print(f"{datetime.now():%H:%M:%S}  no frame "
                          f"(stream/WiFi issue) — holding state  [{misses}]")
                time.sleep(config.DETECT_INTERVAL_SECONDS)
                continue
            recovered = misses > 0
            if recovered:
                print(f"{datetime.now():%H:%M:%S}  stream recovered")
                misses = 0

            frame = resize_for_proc(frame)
            occ = spot_occupancy(detect_vehicle_boxes(model, frame, device=device), spots)

            now = time.time()
            # After a stream gap, re-confirm from scratch: the stall (and the noisy
            # frames around it) shouldn't let pre-gap readings push a state change
            # over the line the instant frames come back. Snap each spot's candidate
            # back to its committed state so a real change must persist a fresh
            # CONFIRM_SECONDS after recovery.
            if recovered:
                for s in spots:
                    if s["occupied"] is not None:
                        s["candidate"] = s["occupied"]
                        s["candidate_since"] = now

            for s in spots:
                raw = occ[s["name"]]

                # confirm-with-hysteresis: a change must persist CONFIRM_SECONDS.
                if s["candidate"] != raw:
                    s["candidate"] = raw
                    s["candidate_since"] = now
                confirmed = now - s["candidate_since"] >= config.CONFIRM_SECONDS

                if s["occupied"] is None:
                    # First reading: accept immediately, no alert.
                    s["occupied"] = raw
                    state = "occupied" if raw else "empty"
                    print(f"{datetime.now():%H:%M:%S}  {s['name']}: initial = {state}")
                    log_event(s["name"], state)
                    continue

                if raw != s["occupied"] and confirmed:
                    s["occupied"] = raw
                    state = "occupied" if raw else "empty"
                    stamp = f"{datetime.now():%H:%M:%S}"
                    print(f"{stamp}  {s['name']}: -> {state.upper()}")
                    log_event(s["name"], state)
                    if config.SAVE_SNAPSHOTS:
                        save_snapshot(frame, spots, s["name"], state)
                    # Text on a freed spot always; on a newly-occupied spot only
                    # if ALERT_ON_BOTH.
                    if (not raw) or config.ALERT_ON_BOTH:
                        text = (f"🅿️ {s['name']} just opened up! ({stamp})" if not raw
                                else f"🚗 {s['name']} is now occupied. ({stamp})")
                        send_imessage(config.IMESSAGE_TO, text)

            time.sleep(config.DETECT_INTERVAL_SECONDS)

    except KeyboardInterrupt:
        print("\nStopping.")


if __name__ == "__main__":
    main()
