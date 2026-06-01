"""Vehicle detection and parking-spot occupancy.

Runs a YOLO model over a snapshot, keeps only vehicle detections, then for each
configured spot computes how much of the spot rectangle is covered by a vehicle
box. If coverage clears ``overlap_threshold`` the spot is considered occupied.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

from ultralytics import YOLO

from .config import Spot

log = logging.getLogger(__name__)


@lru_cache(maxsize=2)
def _load_model(model_path: str) -> YOLO:
    """Load (and cache) the YOLO model. Downloads weights on first use."""
    log.info("Loading YOLO model: %s", model_path)
    return YOLO(model_path)


def _coverage(vehicle_xyxy: tuple[float, float, float, float], spot: Spot) -> float:
    """Fraction of the spot rectangle covered by a vehicle's bounding box."""
    vx1, vy1, vx2, vy2 = vehicle_xyxy
    sx, sy, sw, sh = spot.box
    sx2, sy2 = sx + sw, sy + sh

    ix1, iy1 = max(vx1, sx), max(vy1, sy)
    ix2, iy2 = min(vx2, sx2), min(vy2, sy2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)

    spot_area = max(1.0, float(sw * sh))
    return inter / spot_area


def detect_occupancy(
    image_path: Path,
    spots: list[Spot],
    *,
    model_path: str,
    vehicle_classes: frozenset[str],
    confidence_threshold: float,
    overlap_threshold: float,
) -> dict[str, bool]:
    """Return ``{spot_id: is_occupied}`` for every spot."""
    model = _load_model(model_path)
    result = model(str(image_path), conf=confidence_threshold, verbose=False)[0]
    names = result.names

    vehicles: list[tuple[float, float, float, float]] = []
    for box in result.boxes:
        label = names[int(box.cls)]
        if label in vehicle_classes:
            vehicles.append(tuple(box.xyxy[0].tolist()))

    log.debug("Detected %d vehicle(s)", len(vehicles))

    occupancy: dict[str, bool] = {}
    for spot in spots:
        best = max((_coverage(v, spot) for v in vehicles), default=0.0)
        occupancy[spot.id] = best >= overlap_threshold
    return occupancy
