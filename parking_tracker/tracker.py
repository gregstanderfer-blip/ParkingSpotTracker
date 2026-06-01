"""The polling daemon: snapshot -> detect -> debounce -> alert."""
from __future__ import annotations

import asyncio
import logging

from .blink_client import capture_snapshot, connect
from .config import CREDS_PATH, DATA_DIR, Config, load_config
from .detector import detect_occupancy
from .notifier import send_imessage
from .state import SpotState

log = logging.getLogger(__name__)


async def _poll_once(blink, cfg: Config, state: SpotState) -> None:
    await capture_snapshot(blink, cfg.camera_name, cfg.snapshot_path)
    occupancy = detect_occupancy(
        cfg.snapshot_path,
        cfg.spots,
        model_path=cfg.model_path,
        vehicle_classes=cfg.vehicle_classes,
        confidence_threshold=cfg.confidence_threshold,
        overlap_threshold=cfg.overlap_threshold,
    )

    for spot in cfg.spots:
        is_occupied = occupancy[spot.id]
        if state.update(spot.id, is_occupied) and not is_occupied:
            message = f"🅿️ Parking spot '{spot.id}' just opened up!"
            log.info(message)
            send_imessage(cfg.imessage_recipient, message)

    free = [s.id for s in cfg.spots if not state.occupied(s.id)]
    log.info("Free spots: %s", ", ".join(free) if free else "none")


async def run() -> None:
    cfg = load_config()
    DATA_DIR.mkdir(exist_ok=True)

    log.info("Connecting to Blink...")
    blink, session = await connect(CREDS_PATH)
    state = SpotState(cfg.debounce_frames)
    log.info(
        "Watching %d spot(s) on camera %r every %ds.",
        len(cfg.spots),
        cfg.camera_name,
        cfg.poll_interval_seconds,
    )

    try:
        while True:
            try:
                await _poll_once(blink, cfg, state)
            except Exception:  # one bad cycle shouldn't kill the daemon
                log.exception("Poll cycle failed; will retry next interval")
            await asyncio.sleep(cfg.poll_interval_seconds)
    finally:
        await session.close()
