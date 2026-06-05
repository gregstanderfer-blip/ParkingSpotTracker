# CLAUDE.md — ParkingSpotTracker

Context for AI agents working on this project.

## Goal
Watch one or more parking spots on a fixed outdoor camera and send the user
an **iMessage** the moment a watched spot frees up (occupied → empty). Also
keep a timestamped log + snapshot of every state change so there's a record of
when vehicles arrive/leave.

Runs entirely **locally on the user's Mac mini** — no cloud service in the loop.

## Hardware / environment
- Camera: **TP-Link Tapo C120** (small, IP66, sits on a surface, magnetic base).
- Connection: WiFi, wired power. Exposes a local **RTSP/ONVIF** stream.
- RTSP uses the camera's **local Camera Account** (set in the Tapo app under
  Advanced Settings → Camera Account) — NOT the tplinkcloud login.
- Confirmed working stream (tested with ffmpeg):
  `rtsp://USER:PASS@192.168.68.60:554/stream2` → h264, 640x360, 20fps.
  The camera has a static IP (192.168.68.60), reserved in the router.
- Host: macOS (Mac mini). Alerts go through the built-in **Messages** app via
  AppleScript (`osascript`), so the user must be signed into iMessage. First
  send triggers a macOS Automation permission prompt that must be approved.

> History: an earlier version used a Blink camera via `blinkpy` (polling
> snapshots). That approach was abandoned because Blink doesn't expose a real
> local stream. The repo was fully replaced with this RTSP approach. If you see
> references to Blink anywhere, they're stale.

## Architecture (3 files + config)
- `config.py` — all settings: RTSP_URL, IMESSAGE_TO, detection tuning, file
  paths. **Git-ignored** because it contains the camera password. A sanitized
  `config.example.py` is committed as the template.
- `select_spots.py` — one-time calibration. Pulls a frame from the stream and
  lets the user click polygons around each parking spot; saves `spots.json`.
  Re-run whenever the camera physically moves.
- `monitor.py` — the daemon. **Polls** one frame per cycle via `grab_frame()`
  (open RTSP → read a few frames → disconnect) every `DETECT_INTERVAL_SECONDS`;
  it does NOT hold the stream open. Downscales to `PROC_WIDTH` (1280), runs
  **YOLOv8x** (Apple GPU/MPS when available), decides occupied vs. empty per spot
  via point-in-polygon on the box center + bottom-center. Detection lives in
  `detect_vehicle_boxes()` / `spot_occupancy()` (extracted so tests can call them).
- State machine per spot with hysteresis: a new state must hold for
  `CONFIRM_SECONDS` before it's committed. On a committed change it texts the
  iMessage (both directions if `ALERT_ON_BOTH`), appends to `events.csv`, and
  saves an annotated snapshot to `snapshots/YYYY/MM/DD/`.

## Key design decisions
- Detection chosen over motion/pixel-diff so people, shadows, and pets don't
  trigger false alerts.
- Alerts text on BOTH transitions by default (`ALERT_ON_BOTH=True`); set it False
  to fire only on spot-freed (occupied → empty). Every change is logged + snapshotted.
- **Polling, not streaming** (per user's request, to cut WiFi bandwidth and avoid
  H.264 corruption from lossy WiFi). Grabs one frame every `DETECT_INTERVAL_SECONDS`
  (~15s — cars take >15s to move) instead of holding the 2K stream open 24/7. An
  earlier version recorded ~12s video clips from a continuous stream2 buffer; that
  was removed — snapshots only now.
- Use **`stream1`** (full 2K) downscaled to `PROC_WIDTH`=1280 for detection.
  `stream2` (640x360) was tried first per the original plan but was too blurry —
  YOLO couldn't detect the cars at all. select_spots.py and monitor.py both
  downscale to the same width so spot polygon coordinates line up.
- **Model: YOLOv8x** (not n/s/m). This carport view is hard: dark, oblique
  angle, cars half-hidden under an overhang. Measured mean confidence on the
  shadowed SUV (Spot 1) at imgsz=1280 — yolov8n/s: ~0 (missed), yolov8m: 0.39
  (dips below threshold), **yolov8x: 0.73**. (Counterintuitively, imgsz=1920
  was *worse* for the SUV than 1280.) yolov8x runs ~136ms/frame on MPS — far
  faster than the polling interval needs.
- Force RTSP-over-TCP (`OPENCV_FFMPEG_CAPTURE_OPTIONS=rtsp_transport;tcp`); the
  Tapo's UDP stream drops/corrupts frames over long runs. FFmpeg's chatty SEI
  decode warnings are silenced via `OPENCV_FFMPEG_LOGLEVEL=-8`.

## Run / dev loop
```
source .venv/bin/activate
pip install -r requirements.txt   # ultralytics, opencv-python, numpy
python select_spots.py            # draw spots once -> spots.json
python monitor.py                 # start watching; Ctrl-C to stop
```
`select_spots.py` opens an OpenCV GUI window, so it must run on the Mac with a
display (not over a headless/remote shell).

## Status / what's left to test
- [x] Camera + RTSP confirmed working (ffmpeg pulled a frame → `test.jpg`).
- [x] Scripts written and syntax-checked.
- [x] Set `IMESSAGE_TO` in `config.py`.
- [x] Installed deps into `.venv` (ultralytics, opencv-python, numpy + torch).
- [x] Fixed a corrupted `RTSP_URL` in config.py (password was overwritten with a
      `[email protected]` placeholder) — restored to the working stream1 URL.
- [x] Ran `select_spots.py`; two spots saved to `spots.json` (Spot 1 = left/SUV,
      Spot 2 = center/Mercedes).
- [x] Tuned detection: stream1 + downscale to 1280 + yolov8x + MIN_CONFIDENCE
      0.30. Both spots detect reliably; the dumpster (max 0.16, outside both
      polygons) does not register.
- [x] Ran `monitor.py`; both spots correctly logged as occupied in `events.csv`.
- [x] Verified the iMessage path: a test send via `send_imessage()` delivered.
- [ ] Confirm a real occupied → empty transition fires an alert (couldn't test
      live — both cars were parked the whole session).
- [ ] Re-check detection **at night** (area is lit; tune MIN_CONFIDENCE if misses).
- [ ] Optional: launchd agent so `monitor.py` auto-starts / stays running
      (use `caffeinate -i` so the Mac doesn't sleep while monitoring).

## Tuning knobs (config.py)
- `CONFIRM_SECONDS` — higher = fewer false alerts, slower to fire (currently 90).
  The dark Spot 1 SUV in afternoon overhang shade hovers right at the threshold
  and blips below it for a poll or two; a ~6-poll window rides those out. (Was 30,
  which let occasional 2-poll blips through → false "empty" alerts.)
- `DETECT_INTERVAL_SECONDS` — poll cadence (currently 15).
- `MIN_CONFIDENCE` — raise if it sees phantom cars, lower if it misses them
  (currently 0.20: the shadowed SUV scores ~0.32-0.41 but dips toward the floor;
  0.20 keeps it detected on more frames, and empty spots still read empty — see
  the night_empty fixture). NOTE: brightening / CLAHE made the dark car WORSE.
  Root fix for the shade is camera-side WDR/HDR or more light, not software.
- `MODEL` — yolov8x.pt; smaller models miss the shadowed SUV here.
- `PROC_WIDTH` — detection/calibration width (1280). Changing it means re-running
  `select_spots.py`, since saved polygon coordinates are in this resolution.
