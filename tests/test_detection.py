"""Detection regression tests against saved camera snapshots.

Each fixture is a real frame captured at a particular time of day (evening,
night, morning, ...) with the ground-truth occupancy recorded in
fixtures/expected.json. The test runs the actual detection pipeline
(YOLO + polygon test, at the configured thresholds) and asserts it still reads
each spot correctly. Add fixtures with `python tests/capture_fixture.py LABEL`.

Fixture images are kept local by default (git-ignored), so on a fresh clone these
tests skip; they run wherever the images exist (i.e. on the camera's Mac).
"""
import json
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

import config
import monitor

FIX = Path(__file__).parent / "fixtures"
MANIFEST = FIX / "expected.json"


def _cases():
    if not MANIFEST.exists():
        return []
    return json.loads(MANIFEST.read_text())


CASES = _cases()


@pytest.fixture(scope="module")
def model():
    pytest.importorskip("ultralytics")
    from ultralytics import YOLO
    return YOLO(config.MODEL)


@pytest.fixture(scope="module")
def spots():
    p = FIX / "spots.json"
    if not p.exists():
        pytest.skip("fixtures/spots.json not present")
    data = json.loads(p.read_text())
    for s in data:
        s["polygon_np"] = np.array(s["polygon"], np.int32)
    return data


@pytest.mark.skipif(not CASES, reason="no detection fixtures recorded yet")
@pytest.mark.parametrize("case", CASES, ids=[c["label"] for c in CASES])
def test_fixture_occupancy(case, model, spots):
    img_path = FIX / case["image"]
    if not img_path.exists():
        pytest.skip(f"fixture image {case['image']} kept local / not present")
    img = cv2.imread(str(img_path))
    assert img is not None, f"could not read {img_path}"

    boxes = monitor.detect_vehicle_boxes(model, img)
    occ = monitor.spot_occupancy(boxes, spots)

    for name, expected in case["spots"].items():
        assert occ.get(name) == expected, (
            f"{case['label']}: {name} detected={occ.get(name)} expected={expected}")
