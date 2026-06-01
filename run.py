#!/usr/bin/env python3
"""Start the parking-spot watcher daemon."""
import asyncio
import logging

from parking_tracker.tracker import run

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

if __name__ == "__main__":
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("\nStopped.")
