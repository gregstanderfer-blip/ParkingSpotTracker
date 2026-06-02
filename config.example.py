"""
Copy this file to config.py and fill in your values:

    cp config.example.py config.py

config.py is git-ignored because it holds your camera password.
"""

# ---- Camera (Tapo C120) --------------------------------------------------
# Build this from the LOCAL "Camera Account" you created in the Tapo app
# (Settings -> Advanced Settings -> Camera Account), NOT your tplinkcloud login.
#
#   rtsp://USERNAME:PASSWORD@CAMERA_IP:554/stream1
#
# stream1 = full 2K (sharp), stream2 = 640x360 (blurry). Detection downscales
# stream1 to PROC_WIDTH; clips are recorded from stream2 (see below).
RTSP_URL = "rtsp://USERNAME:PASSWORD@192.168.1.42:554/stream1"

# ---- Alerts (Apple Messages) --------------------------------------------
# Your own phone number or iMessage email, in the exact format Messages uses,
# e.g. "+15551234567" or "you@icloud.com".
IMESSAGE_TO = "+15551234567"

# ---- Detection tuning ----------------------------------------------------
CONFIRM_SECONDS = 8            # a new state must hold this long before it counts
DETECT_INTERVAL_SECONDS = 2.0  # how often to run the detector
MIN_CONFIDENCE = 0.30          # min YOLO confidence to count a vehicle
MODEL = "yolov8x.pt"           # downloads on first run; see README for why "x"
PROC_WIDTH = 1280              # detection/calibration width (must match between
                               # select_spots.py and monitor.py)

# ---- Files ---------------------------------------------------------------
SPOTS_FILE = "spots.json"      # created by select_spots.py
SNAPSHOT_DIR = "snapshots"     # annotated image saved on each state change
CLIP_DIR = "clips"             # short video saved on each state change
LOG_FILE = "events.csv"        # timestamped log of every transition
SAVE_SNAPSHOTS = True

# ---- Alerts & media on state change --------------------------------------
# Text on BOTH transitions (freed AND newly occupied), or only when freed.
ALERT_ON_BOTH = True
# Attach the snapshot to the text. Off by default — Messages attachments via
# AppleScript are unreliable on macOS Ventura+; the image is saved locally either
# way. (Text alerts are reliable.)
ATTACH_PHOTO = False
# Save ~CLIP_SECONDS of video (a rolling buffer ending at detection) on each
# change, so you can watch the transition. Recorded from the low-res stream2.
SAVE_CLIPS = True
CLIP_SECONDS = 12
CLIP_FPS = 10
CLIP_RTSP_URL = RTSP_URL.replace("/stream1", "/stream2")
