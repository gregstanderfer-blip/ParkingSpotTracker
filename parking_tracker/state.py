"""Debounced per-spot occupancy state. No external dependencies."""
from __future__ import annotations


class SpotState:
    """Tracks committed occupancy per spot with debouncing.

    A spot only flips state after ``debounce_frames`` consecutive readings of
    the new value, so a single noisy detection (shadow, person walking past)
    won't trigger a false alert.
    """

    def __init__(self, debounce_frames: int) -> None:
        self._debounce = max(1, debounce_frames)
        self._committed: dict[str, bool] = {}
        self._pending: dict[str, tuple[bool, int]] = {}

    def occupied(self, spot_id: str) -> bool:
        return self._committed.get(spot_id, True)

    def update(self, spot_id: str, occupied: bool) -> bool:
        """Feed a new reading. Returns True if the committed state just changed."""
        if spot_id not in self._committed:
            self._committed[spot_id] = occupied  # establish baseline silently
            return False

        if occupied == self._committed[spot_id]:
            self._pending.pop(spot_id, None)
            return False

        pending_val, count = self._pending.get(spot_id, (occupied, 0))
        count = count + 1 if pending_val == occupied else 1
        if count >= self._debounce:
            self._committed[spot_id] = occupied
            self._pending.pop(spot_id, None)
            return True

        self._pending[spot_id] = (occupied, count)
        return False
