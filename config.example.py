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
# stream1 = full 2K (sharp). The monitor downscales it to PROC_WIDTH and polls a
# single frame per check (it does NOT hold the stream open), so WiFi usage is tiny.
RTSP_URL = "rtsp://USERNAME:PASSWORD@192.168.1.42:554/stream1"

# ---- Alerts (Apple Messages) --------------------------------------------
# Your own phone number or iMessage email, in the exact format Messages uses,
# e.g. "+15551234567" or "you@icloud.com".
IMESSAGE_TO = "+15551234567"

# ---- Detection tuning ----------------------------------------------------
CONFIRM_SECONDS = 90           # a new state must hold this long before it counts;
                               # long enough (~6 polls) to ride out brief dips when
                               # a dark car is in shade
DETECT_INTERVAL_SECONDS = 15   # how often to poll (grab one frame + detect). Also
                               # sets how little WiFi bandwidth is used. Cars take
                               # >15s to move, so ~15s is plenty.
MIN_CONFIDENCE = 0.20          # min YOLO confidence to count a vehicle (low, so a
                               # dark/shadowed car isn't lost; CONFIRM_SECONDS is
                               # the main guard against the remaining dips)
MODEL = "yolov8x.pt"           # downloads on first run; see README for why "x"
PROC_WIDTH = 1280              # detection/calibration width (must match between
                               # select_spots.py and monitor.py)

# ---- Files ---------------------------------------------------------------
SPOTS_FILE = "spots.json"      # created by select_spots.py
SNAPSHOT_DIR = "snapshots"     # annotated image saved on each state change
LOG_FILE = "events.csv"        # timestamped log of every transition
SAVE_SNAPSHOTS = True

# ---- Alerts on state change ----------------------------------------------
# Text on BOTH transitions (freed AND newly occupied), or only when freed.
# Alerts are text-only; the matching snapshot is saved to SNAPSHOT_DIR.
ALERT_ON_BOTH = True
