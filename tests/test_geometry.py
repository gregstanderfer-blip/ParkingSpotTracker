"""Pure-logic tests for the occupancy geometry — no camera or model needed."""
import numpy as np

import monitor

# A 100x100 square polygon.
SQUARE = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], np.int32)


def test_box_center_inside():
    assert monitor.box_in_spot([40, 40, 60, 60], SQUARE) is True


def test_box_fully_outside():
    assert monitor.box_in_spot([200, 200, 260, 260], SQUARE) is False


def test_box_center_above_but_bottom_inside():
    # Center is above the polygon (negative y), but the box's bottom-center —
    # where the car meets the ground — lands inside. Should count as occupied.
    assert monitor.box_in_spot([40, -80, 60, 40], SQUARE) is True


def test_box_below_polygon_is_outside():
    assert monitor.box_in_spot([40, 200, 60, 300], SQUARE) is False


def test_spot_occupancy_maps_each_spot():
    spots = [
        {"name": "A", "polygon_np": SQUARE},
        {"name": "B", "polygon_np": np.array(
            [[200, 200], [300, 200], [300, 300], [200, 300]], np.int32)},
    ]
    boxes = [[40, 40, 60, 60]]  # inside A only
    assert monitor.spot_occupancy(boxes, spots) == {"A": True, "B": False}


def test_spot_occupancy_empty_when_no_boxes():
    spots = [{"name": "A", "polygon_np": SQUARE}]
    assert monitor.spot_occupancy([], spots) == {"A": False}
