"""Existing synthetic scenes must stay bit-identical for a given seed.

The digests were recorded on commit 303043e, before reported-velocity support
was added to the generator.  A change here means previously published
benchmark numbers are no longer reproducible.
"""

import hashlib

import numpy as np
import pytest

from dopplertracker.stress_scenes import stress_scene

RECORDED = {
    ("clean", 0): "411be9ecfeac64d714110428da3c6c8d33b2da39dec6c1e58842d3ec457b471c",
    ("adsb_noise", 3): "a53b1bbf380a2d157a1dcadf57ef6cfe22b87d6c2e19a1d392283d22a17bd9fa",
    ("combined_errors", 7): "6674c35de810c8a82e83b3ec81bae500e80b03b56d48c5c221e4936eb3b64d88",
}


def scene_hash(scene):
    h = hashlib.sha256()
    for t in scene["tracks"]:
        h.update(np.ascontiguousarray(t.times).tobytes())
        h.update(np.ascontiguousarray(t.positions).tobytes())
    h.update(np.ascontiguousarray(scene["spectrogram"].power).tobytes())
    h.update(np.ascontiguousarray(scene["truth"]).tobytes())
    h.update(repr([(e.window, e.cpa, e.f0) for e in scene["events"]]).encode())
    h.update(repr(scene["candidate"]).encode())
    return h.hexdigest()


@pytest.mark.parametrize(("scenario", "seed"), sorted(RECORDED))
def test_existing_scenes_are_bit_identical(scenario, seed):
    assert scene_hash(stress_scene(scenario, seed)) == RECORDED[(scenario, seed)]
