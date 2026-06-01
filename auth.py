#!/usr/bin/env python3
"""One-time Blink login.

Run this once to authenticate (including the 2FA code Blink emails/texts you).
It saves a credentials token to data/blink_creds.json so the daemon can reconnect
without prompting. Re-run only if the token expires or you change your password.
"""
import asyncio
import logging
import os

from aiohttp import ClientSession
from blinkpy.auth import Auth
from blinkpy.blinkpy import Blink

# Importing config loads .env and exposes the credential paths.
from parking_tracker.config import CREDS_PATH, DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


async def main() -> None:
    username = os.environ.get("BLINK_USERNAME", "").strip()
    password = os.environ.get("BLINK_PASSWORD", "").strip()
    if not username or not password:
        raise SystemExit("Set BLINK_USERNAME and BLINK_PASSWORD in .env first.")

    DATA_DIR.mkdir(exist_ok=True)
    session = ClientSession()
    blink = Blink(session=session)
    blink.auth = Auth(
        {"username": username, "password": password},
        no_prompt=True,
        session=session,
    )
    await blink.start()

    if blink.auth.check_key_required():
        code = input("Enter the 2FA code Blink sent you: ").strip()
        await blink.auth.send_auth_key(blink, code)
        await blink.setup_post_verify()

    await blink.save(str(CREDS_PATH))
    print(f"\n✅ Saved credentials to {CREDS_PATH}")
    print(f"📷 Cameras found: {list(blink.cameras)}")
    print("\nNext: run `python calibrate.py` to mark your parking spots.")
    await session.close()


if __name__ == "__main__":
    asyncio.run(main())
