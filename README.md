# Parking Spot Monitor (Tapo C120 → Mac → iMessage)

Watches one or more parking spots on a TP-Link Tapo camera's video feed and
texts you the moment a spot changes between **occupied** and **empty**. It uses
on-device AI car detection (YOLOv8), so it ignores people, shadows, and pets,
and it runs entirely on your Mac — no cloud service in the loop.

Every state change is logged, snapshotted, and saved as a short video clip so
you have a record of when vehicles come and go.

## How it works

- A threaded RTSP reader pulls the live stream from the camera and always serves
  the most recent frame (auto-reconnecting through glitches).
- Each frame is downscaled to `PROC_WIDTH` and run through **YOLOv8x** to find
  vehicles (car / motorcycle / bus / truck), on the Apple GPU (MPS) when present.
- A spot is **occupied** if a detected vehicle's box sits inside the polygon you
  drew for it. A new state must hold for `CONFIRM_SECONDS` before it's committed
  (hysteresis filters out cars driving past).
- On a committed change it: logs to `events.csv`, saves an annotated image to
  `snapshots/`, saves a ~`CLIP_SECONDS` video to `clips/`, and sends an iMessage.

## Requirements

- macOS (alerts go through the built-in **Messages** app; signed into iMessage)
- Python 3.10+
- A Tapo (or any RTSP/ONVIF) camera with a clear view of the spots
- ~150 MB disk for the model (downloads automatically on first run)

## One-time setup

### 1. Enable RTSP / make a local Camera Account
In the **Tapo app**: camera → **Settings → Advanced Settings → Camera Account**.
Create a **username + password** here. This is a *local* credential, separate
from your tplinkcloud.com login — RTSP only works with this one.

### 2. Give the camera a fixed IP
Find the camera's IP (Tapo app → Settings → Device Info) and set a **DHCP
reservation** for it in your router so it never changes.

### 3. Configure
```bash
cp config.example.py config.py
```
Edit `config.py`:
- `RTSP_URL` → `rtsp://USERNAME:PASSWORD@CAMERA_IP:554/stream1`
- `IMESSAGE_TO` → your phone number or iMessage email

> Tip: test the URL in **VLC** (File → Open Network Stream) first.

### 4. Install dependencies
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # ultralytics, opencv-python, numpy (+torch)
```

### 5. Allow Messages automation
The first time it sends an alert, macOS asks to let your terminal control
**Messages** — click **OK** (or pre-approve under System Settings → Privacy &
Security → Automation).

## Usage

**Mark your spots (once, or whenever the camera moves):**
```bash
python select_spots.py
```
Click corner points around each spot, `n` for the next spot, `s` to save. This
opens a GUI window, so run it on the Mac itself (not a headless shell).

**Start watching:**
```bash
python monitor.py        # Ctrl-C to stop
```

## What you get

- **iMessage alert** on each state change (e.g. *"🅿️ Spot 1 just opened up!"* /
  *"🚗 Spot 1 is now occupied."*). Text-only by default — see the note below.
- **events.csv** — timestamped log of every occupied/empty change.
- **snapshots/** — an annotated image at each change (red = occupied, green = empty).
- **clips/** — a ~12-second video of each change so you can watch it happen.

> **About photos in texts:** sending image *attachments* via Messages/AppleScript
> is unreliable on macOS Ventura+ (the file silently drops), so alerts are
> text-only. The photo and clip are always saved locally in `snapshots/` and
> `clips/`. Flip `ATTACH_PHOTO` in `config.py` to try attachments anyway.

## Run it always-on (launchd)

To auto-start at login and keep it running, use a LaunchAgent. A template is in
[`launchd/`](launchd/). Edit the paths, then:
```bash
cp launchd/com.example.parkingspottracker.plist ~/Library/LaunchAgents/
# edit the two absolute paths inside it to match your checkout
launchctl load -w ~/Library/LaunchAgents/com.example.parkingspottracker.plist
```
It wraps the monitor in `caffeinate -i` so the Mac won't idle-sleep while
watching, restarts it if it crashes, and logs to `monitor.log`.

## Tuning (`config.py`)

| Setting | What it does |
| --- | --- |
| `CONFIRM_SECONDS` | How long a change must hold before it counts (higher = fewer false alerts). |
| `DETECT_INTERVAL_SECONDS` | How often it checks. |
| `MIN_CONFIDENCE` | Min YOLO confidence for a vehicle. Raise if it sees phantom cars; lower if it misses them (e.g. at night). |
| `MODEL` | `yolov8x.pt` here — smaller models couldn't detect the shadowed, oblique cars in this view. Try a smaller model if your angle is easier and you want it lighter. |
| `PROC_WIDTH` | Detection/calibration width. Changing it means re-running `select_spots.py`. |
| `ALERT_ON_BOTH` | Text on both transitions, or only when a spot frees up. |
| `SAVE_CLIPS` / `CLIP_SECONDS` | Save a clip of each change and how long. |

## Notes & limitations

- Detection accuracy depends on camera angle and lighting; re-check at night and
  tune `MIN_CONFIDENCE` / your spot polygons if needed.
- The Mac must stay awake/running to monitor (the launchd setup handles this).
- `config.py`, `spots.json`, `snapshots/`, `clips/`, and `events.csv` are
  git-ignored — they stay on your machine.
