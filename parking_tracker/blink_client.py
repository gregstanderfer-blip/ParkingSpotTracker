"""Thin async wrapper around blinkpy for grabbing camera snapshots.

Blink cameras are battery-powered and motion-triggered — they don't stream.
We ask the camera to take a fresh photo (`snap_picture`), wait for it to upload,
refresh the account, then save the latest image to disk.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from aiohttp import ClientSession
from blinkpy.auth import Auth
from blinkpy.blinkpy import Blink
from blinkpy.helpers.util import json_load

log = logging.getLogger(__name__)


async def connect(creds_path: Path) -> tuple[Blink, ClientSession]:
    """Open an authenticated Blink session from saved credentials.

    Returns the Blink instance and its aiohttp session — the caller is
    responsible for closing the session.
    """
    if not Path(creds_path).exists():
        raise FileNotFoundError(
            f"{creds_path} not found. Run `python auth.py` once to log in to Blink."
        )
    session = ClientSession()
    blink = Blink(session=session)
    blink.auth = Auth(await json_load(str(creds_path)), no_prompt=True, session=session)
    await blink.start()
    return blink, session


async def capture_snapshot(
    blink: Blink,
    camera_name: str,
    out_path: Path,
    settle_seconds: float = 5.0,
) -> Path:
    """Trigger a fresh photo on the named camera and save it to ``out_path``."""
    if camera_name not in blink.cameras:
        raise KeyError(
            f"Camera {camera_name!r} not found. Available cameras: "
            f"{list(blink.cameras)}"
        )
    camera = blink.cameras[camera_name]
    await camera.snap_picture()
    await asyncio.sleep(settle_seconds)  # let the new image upload
    await blink.refresh(force=True)
    await camera.image_to_file(str(out_path))
    log.debug("Saved snapshot from %s to %s", camera_name, out_path)
    return out_path
