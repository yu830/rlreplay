"""Scaffold-only packaging smoke test; not a training-correctness check."""

from importlib.metadata import version

import rlreplay


def test_package_metadata_matches_module() -> None:
    assert version("rlreplay") == rlreplay.__version__
