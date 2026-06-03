"""
monitor.py — watch parking spots and text you when one frees up.

    python monitor.py

What it does:
  - Reads the live RTSP stream from your Tapo C120.
  - Runs a YOLO model to find vehicles (car / truck / bus / motorcycle).
  - Decides each spot is OCCUPIED or EMPTY based on whether a vehicle
    sits inside the polygon you drew with select_spots.py.
  - When a spot changes state (and it holds for CONFIRM_SECONDS), it texts
    you an iMessage via the Messages app — on a freed spot the alert includes
    a photo of the now-empty spot. (See ALERT_ON_BOTH / ATTACH_PHOTO in config.)
  - Every state change is logged to events.csv, saved as an annotated snapshot
    in snapshots/, and saved as a ~CLIP_SECONDS video clip in clips/.

Stop it any time with Ctrl-C.
"""

import csv
import json
import os
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np

import config

# Tapo's RTSP over UDP drops/corrupts frames during long runs; force TCP
# (matches the `-rtsp_transport tcp` that worked in ffmpeg). Must be set before
# cv2.VideoCapture opens the stream.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
# Silence the harmless but extremely chatty "SEI type ... truncated" H264 decode
# warnings the Tapo stream emits (FFmpeg log level -8 = quiet).
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")

try:
    from ultralytics import YOLO
except ImportError:
    print("Missing dependency. Run:  pip install -r requirements.txt")
    sys.exit(1)


def resize_for_proc(frame):
    """Downscale a frame to config.PROC_WIDTH (keeps detection light and makes
    spot coordinates match select_spots.py, which uses the same width)."""
    h, w = frame.shape[:2]
    if w == config.PROC_WIDTH:
        return frame
    return cv2.resize(frame, (config.PROC_WIDTH, round(h * config.PROC_WIDTH / w)))

# COCO class IDs that count as "a vehicle in the spot".
VEHICLE_CLASS_IDS = {2, 3, 5, 7}  # car, motorcycle, bus, truck


# --------------------------------------------------------------------------
# Threaded RTSP reader: always hands back the most recent frame so we never
# process stale, buffered video.
# --------------------------------------------------------------------------
class _Reader:
    """Base: a threaded RTSP reader with tolerant reconnect."""
    def __init__(self, url):
        self.url = url
        self.cap = cv2.VideoCapture(url)
        self.lock = threading.Lock()
        self.reconnects = 0
        self.running = True

    def _read_or_reconnect(self, fails):
        ok, f = self.cap.read()
        if ok:
            return True, f, 0
        # Tolerate brief glitches; only do a full (slow) reconnect after a run
        # of failures — reconnecting on every dropped frame cripples the rate.
        fails += 1
        if fails >= 10:
            self.reconnects += 1
            self.cap.release()
            time.sleep(0.5)
            self.cap = cv2.VideoCapture(self.url)
            return False, None, 0
        time.sleep(0.05)
        return False, None, fails

    def stop(self):
        # Stop the thread BEFORE releasing the capture, or release() can race
        # with an in-flight read() and segfault.
        self.running = False
        if self.t.is_alive():
            self.t.join(timeout=3)
        try:
            self.cap.release()
        except Exception:
            pass


class StreamReader(_Reader):
    """Detection reader: always hands back the most recent frame so we never
    process stale, buffered video."""
    def __init__(self, url):
        super().__init__(url)
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        self.frame = None
        self.t = threading.Thread(target=self._loop, daemon=True)
        self.t.start()

    def _loop(self):
        fails = 0
        while self.running:
            ok, f, fails = self._read_or_reconnect(fails)
            if not ok:
                continue
            with self.lock:
                self.frame = f

    def read(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()


class ClipRecorder(_Reader):
    """Keeps a rolling buffer of recent raw frames from a (low-res) substream,
    so a short clip ending at 'now' can be dumped on demand. Reading stream2
    (native 640x360) means no resize/encode in the hot loop — smooth + cheap."""
    def __init__(self, url, seconds, fps):
        super().__init__(url)
        self.fps = fps
        self.buf = deque(maxlen=max(1, int(seconds * fps)))
        self._last_t = 0.0
        self.t = threading.Thread(target=self._loop, daemon=True)
        self.t.start()

    def _loop(self):
        fails = 0
        while self.running:
            ok, f, fails = self._read_or_reconnect(fails)
            if not ok:
                continue
            now = time.time()
            if now - self._last_t >= 1.0 / self.fps:   # throttle to ~fps
                self._last_t = now
                with self.lock:
                    self.buf.append((now, f))

    def write_clip(self, path):
        """Dump the buffer to an mp4 at its true (timestamp-based) fps, so
        playback duration matches real time. Returns the path, or None."""
        with self.lock:
            items = list(self.buf)
        if len(items) < 2:
            return None
        span = items[-1][0] - items[0][0]
        fps = (len(items) - 1) / span if span > 0 else self.fps
        fps = max(1.0, min(fps, 30.0))
        h, w = items[0][1].shape[:2]
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for _, img in items:
            writer.write(img)
        writer.release()
        return path


def send_imessage(to, text, attachment=None):
    """Send an iMessage (and an optional image/video) via the Messages app.

    One AppleScript call: text first, then — if attaching — a short delay so
    Messages finishes the text before the file (otherwise it drops the file),
    with the file coerced to a POSIX file OUTSIDE the tell block.
    """
    safe = text.replace("\\", "\\\\").replace('"', '\\"')
    pre, file_send = "", ""
    if attachment:
        ap = os.path.abspath(attachment)
        pre = f'set theFile to POSIX file "{ap}"\n'
        file_send = "    delay 1\n    send theFile to targetBuddy\n"
    script = (
        f'{pre}tell application "Messages"\n'
        '    set targetService to 1st account whose service type = iMessage\n'
        f'    set targetBuddy to participant "{to}" of targetService\n'
        f'    send "{safe}" to targetBuddy\n'
        f'{file_send}'
        'end tell\n'
    )
    try:
        subprocess.run(["osascript", "-e", script], check=True,
                       capture_output=True, timeout=30)
        print(f"  -> alert sent to {to}" + (" (with photo)" if attachment else ""))
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
    """True if the vehicle box meaningfully overlaps the spot polygon.
    We test the box center plus its bottom-center (where the car meets
    the ground) — robust to a car sticking up out of the polygon."""
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
    confidence floor. Pure-ish wrapper so detection can be unit-tested on a saved
    image. ``conf``/``imgsz`` default to the values in config."""
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
    os.makedirs(config.SNAPSHOT_DIR, exist_ok=True)
    img = frame.copy()
    for s in spots:
        color = (0, 0, 255) if s["occupied"] else (0, 255, 0)
        cv2.polylines(img, [s["polygon_np"]], True, color, 2)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = spot_name.replace(" ", "_")
    path = os.path.join(config.SNAPSHOT_DIR, f"{ts}_{safe}_{new_state}.jpg")
    cv2.imwrite(path, img)
    return path


def clip_path(spot_name, new_state):
    os.makedirs(config.CLIP_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = spot_name.replace(" ", "_")
    return os.path.join(config.CLIP_DIR, f"{ts}_{safe}_{new_state}.mp4")


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

    # Use the Apple GPU (MPS) if available, else CPU.
    try:
        import torch
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    except Exception:
        device = "cpu"
    print(f"Running detection on: {device}")

    print("Connecting to camera...")
    reader = StreamReader(config.RTSP_URL)
    # Separate low-res reader that keeps a rolling buffer for clip recording.
    recorder = (ClipRecorder(config.CLIP_RTSP_URL, config.CLIP_SECONDS, config.CLIP_FPS)
                if config.SAVE_CLIPS else None)
    # wait for first frame
    t0 = time.time()
    while reader.read() is None:
        if time.time() - t0 > 20:
            print("ERROR: no frames from the stream. Check RTSP_URL in config.py.")
            reader.stop()
            if recorder:
                recorder.stop()
            sys.exit(1)
        time.sleep(0.5)
    print("Connected. Watching... (Ctrl-C to stop)\n")

    # Per-spot state machine.
    for s in spots:
        s["occupied"] = None          # confirmed state (None until first read)
        s["candidate"] = None         # state we're waiting to confirm
        s["candidate_since"] = 0.0

    try:
        while True:
            frame = reader.read()
            if frame is None:
                time.sleep(0.5)
                continue
            frame = resize_for_proc(frame)

            boxes = detect_vehicle_boxes(model, frame, device=device)
            occ = spot_occupancy(boxes, spots)

            now = time.time()
            for s in spots:
                raw = occ[s["name"]]

                # confirm-with-hysteresis
                if s["candidate"] != raw:
                    s["candidate"] = raw
                    s["candidate_since"] = now

                confirmed_long_enough = (
                    now - s["candidate_since"] >= config.CONFIRM_SECONDS
                )

                if s["occupied"] is None:
                    # first reading: accept immediately, no alert
                    if confirmed_long_enough or s["occupied"] is None:
                        s["occupied"] = raw
                        state = "occupied" if raw else "empty"
                        print(f"{datetime.now():%H:%M:%S}  {s['name']}: initial = {state}")
                        log_event(s["name"], state)
                    continue

                if raw != s["occupied"] and confirmed_long_enough:
                    s["occupied"] = raw
                    state = "occupied" if raw else "empty"
                    stamp = f"{datetime.now():%H:%M:%S}"
                    print(f"{stamp}  {s['name']}: -> {state.upper()}")
                    log_event(s["name"], state)

                    snap_path = None
                    if config.SAVE_SNAPSHOTS:
                        snap_path = save_snapshot(frame, spots, s["name"], state)
                    if recorder is not None:
                        try:
                            cp = recorder.write_clip(clip_path(s["name"], state))
                            if cp:
                                print(f"  -> saved clip {cp}")
                        except Exception as e:
                            print(f"  !! clip save failed: {e}")

                    # Text on a freed spot always; on a newly-occupied spot only
                    # if ALERT_ON_BOTH. Attach the snapshot photo when enabled.
                    if (not raw) or config.ALERT_ON_BOTH:
                        if not raw:
                            text = f"🅿️ {s['name']} just opened up! ({stamp})"
                        else:
                            text = f"🚗 {s['name']} is now occupied. ({stamp})"
                        attach = snap_path if config.ATTACH_PHOTO else None
                        send_imessage(config.IMESSAGE_TO, text, attach)

            time.sleep(config.DETECT_INTERVAL_SECONDS)

    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        reader.stop()
        if recorder is not None:
            recorder.stop()


if __name__ == "__main__":
    main()
