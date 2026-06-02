"""
select_spots.py — draw the parking spots you want to watch.

Run this ONCE (and again any time you move the camera):

    python select_spots.py

A window opens showing a live frame from your camera.
  - LEFT-CLICK to drop corner points around a spot (3+ points).
  - Press  n  to finish the current spot and start the next one.
  - Press  u  to undo the last point.
  - Press  s  to save all spots to spots.json and quit.
  - Press  q  to quit without saving.

Outlining each spot tightly (the actual asphalt rectangle) gives the
most reliable occupied/empty decision.
"""

import json
import os
import sys

import cv2
import numpy as np

import config

# Force RTSP-over-TCP, same as monitor.py (Tapo's UDP stream is unreliable).
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")


def grab_frame(url):
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        print(f"ERROR: could not open stream:\n  {url}\n"
              "Check the RTSP_URL in config.py (camera account + IP), "
              "and test it in VLC first.")
        sys.exit(1)
    # Read a few frames to let the stream settle.
    frame = None
    for _ in range(10):
        ok, f = cap.read()
        if ok:
            frame = f
    cap.release()
    if frame is None:
        print("ERROR: opened the stream but got no frames. Try stream1 vs stream2.")
        sys.exit(1)
    # Downscale to the same width monitor.py detects at, so the polygon
    # coordinates we save line up with what the monitor sees.
    h, w = frame.shape[:2]
    if w != config.PROC_WIDTH:
        frame = cv2.resize(frame, (config.PROC_WIDTH,
                                   round(h * config.PROC_WIDTH / w)))
    return frame


def main():
    frame = grab_frame(config.RTSP_URL)
    base = frame.copy()

    spots = []          # list of completed polygons (each = list of [x,y])
    current = []        # points of the spot being drawn

    def redraw():
        img = base.copy()
        # finished spots
        for i, poly in enumerate(spots):
            pts = np.array(poly, np.int32)
            cv2.polylines(img, [pts], True, (0, 255, 0), 2)
            cx, cy = int(np.mean(pts[:, 0])), int(np.mean(pts[:, 1]))
            cv2.putText(img, f"Spot {i+1}", (cx - 30, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        # in-progress spot
        for p in current:
            cv2.circle(img, tuple(p), 4, (0, 165, 255), -1)
        if len(current) > 1:
            cv2.polylines(img, [np.array(current, np.int32)], False,
                          (0, 165, 255), 2)
        hint = "L-click: add point | n: next spot | u: undo | s: save+quit | q: quit"
        cv2.putText(img, hint, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2)
        cv2.imshow("Select parking spots", img)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            current.append([x, y])
            redraw()

    cv2.namedWindow("Select parking spots")
    cv2.setMouseCallback("Select parking spots", on_mouse)
    redraw()

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key == ord("u"):
            if current:
                current.pop()
                redraw()
        elif key == ord("n"):
            if len(current) >= 3:
                spots.append(current.copy())
                current.clear()
                redraw()
            else:
                print("A spot needs at least 3 points.")
        elif key == ord("s"):
            if len(current) >= 3:
                spots.append(current.copy())
                current.clear()
            if not spots:
                print("No spots drawn — nothing to save.")
                continue
            data = [{"name": f"Spot {i+1}", "polygon": poly}
                    for i, poly in enumerate(spots)]
            with open(config.SPOTS_FILE, "w") as f:
                json.dump(data, f, indent=2)
            print(f"Saved {len(data)} spot(s) to {config.SPOTS_FILE}")
            break
        elif key == ord("q"):
            print("Quit without saving.")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
