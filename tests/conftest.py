"""Synthetic evidence shared by unit and capture lifecycle tests."""

import pytest

from rlreplay.core import Evidence, Profile, Sample


@pytest.fixture
def evidence():
    return Evidence(
        Profile(),
        (
            Sample("a0", "a", 1.0, (1, 0), (-0.7071062811869011, 0)),
            Sample("a1", "a", 3.0, (1, 1), (0.7071062811869011, 0.7071062811869011)),
        ),
        (("a0", "a"), ("a1", "a")),
        {
            "framework": "verl",
            "framework_revision": "a" * 40,
            "algorithm_sha256": "b" * 64,
            "entrypoint": "verl.trainer.ppo.ray_trainer.compute_advantage",
            "entrypoint_sha256": "c" * 64,
            "phase": "post_advantage",
            "receipt_origin": "synthetic-test",
            "device": "cpu",
            "dtype": "float32",
        },
    )
