# ParkingSpotTracker

Watches the parking spots outside via a **Blink camera** and sends you an
**iMessage** the moment a spot opens up.

It works by periodically grabbing a snapshot from the camera, running a YOLO
object-detection model to find vehicles, and checking whether each parking spot
you've marked is covered by a car. When a spot goes from occupied → empty, it
texts you through the macOS Messages app.

> **How Blink works:** Blink cameras are battery-powered and motion-triggered —
> they don't stream continuously. This tool asks the camera to take a fresh photo
> each polling cycle. Frequent polling drains the battery faster, so pick an
> interval that balances responsiveness against battery life (60s is a sensible
> start; consider 2–5 min for battery cameras).

## Requirements

- macOS (alerts use the built-in **Messages** app via AppleScript)
- Python 3.10+
- A Blink camera with a clear view of the spots
- You must be signed into iMessage in the Messages app

## Setup

```bash
cd ParkingSpotTracker

# 1. Install dependencies (a virtualenv is recommended)
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Configure secrets
cp .env.example .env
#    → edit .env: Blink login + the phone/Apple ID to alert

# 3. Log in to Blink once (handles the 2FA code)
python auth.py

# 4. Mark your parking spots on a live snapshot
python calibrate.py            # or: python calibrate.py "Camera Name"

# 5. Start watching
python run.py
```

The first run downloads the YOLO model weights (`yolov8n.pt`, ~6 MB) automatically.

## How alerts work

When a spot frees up you'll get an iMessage like:

> 🅿️ Parking spot 'spot-1' just opened up!

The first time you run it, macOS will ask Terminal/your shell for permission to
control Messages — allow it (System Settings → Privacy & Security → Automation).

## Tuning (`config.json`)

`calibrate.py` generates `config.json`. Adjust these to taste:

| Key | What it does |
| --- | --- |
| `poll_interval_seconds` | How often to check (lower = faster alerts, more battery drain). |
| `confidence_threshold` | Minimum YOLO confidence to count a detection as a vehicle. |
| `overlap_threshold` | Fraction of a spot a car must cover to count as occupied (0–1). |
| `debounce_frames` | Consecutive readings before a state change "sticks" (prevents false alerts from shadows/passers-by). |
| `vehicle_classes` | Which detected object types count as vehicles. |

Re-run `python calibrate.py` any time the camera moves or you want to re-mark spots.

## Run it always-on (optional)

`run.py` is a long-lived daemon. To keep it running across reboots/crashes, wrap
it in a `launchd` agent (with `KeepAlive`) or run it under `tmux`/`screen`.

## Project layout

```
auth.py          One-time Blink login (saves a token to data/)
calibrate.py     Snapshot + draw boxes around spots → config.json
run.py           Starts the watcher daemon
parking_tracker/
  config.py      Loads .env + config.json
  blink_client.py  Snapshot capture via blinkpy
  detector.py    YOLO vehicle detection + occupancy logic
  notifier.py    Sends iMessages via osascript
  state.py       Debounced per-spot occupancy state machine
  tracker.py     Polling loop + alert dispatch
```

## Notes & limitations

- Uses the **unofficial** `blinkpy` library — there is no official public Blink API.
- Detection accuracy depends on camera angle and lighting; tune `overlap_threshold`
  and spot boxes if you see false positives/negatives.
- Credentials (`.env`, `data/blink_creds.json`) and your local `config.json` are
  git-ignored — they stay on your machine.
