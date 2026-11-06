"""Synthetic producer example using the shared contract, capsule and replay API."""

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

from rlreplay.artifacts import read, write
from rlreplay.contracts.grpo import check
from rlreplay.core import Evidence, Profile, Sample
from rlreplay.replay import semantic_replay


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    healthy = Evidence(
        Profile(implementation="synthetic-example"),
        (
            Sample("s0", "g0", 0.0, (1, 0), (-0.7071062811869011, 0.0)),
            Sample("s1", "g0", 2.0, (1, 1), (0.7071062811869011, 0.7071062811869011)),
        ),
        (("s0", "g0"), ("s1", "g0")),
        {
            "framework": "synthetic-example",
            "framework_revision": "local-example-v1",
            "algorithm_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "phase": "hand-specified-frozen-observations",
            "receipt_origin": "synthetic-example",
            "device": "cpu",
            "dtype": "float32",
        },
    )
    faulty = replace(
        healthy,
        samples=(
            replace(healthy.samples[0], advantage=(-0.2, 0.0)),
            healthy.samples[1],
        ),
    )
    results = []
    for name, evidence, expected in (
        ("healthy", healthy, "PASS"),
        ("faulty", faulty, "SUSPECTED_MISMATCH"),
    ):
        path = write(evidence, args.directory / name)
        restored = read(path)
        assert restored == evidence
        assert check(restored).verdict == expected
        assert semantic_replay(restored)["finding"]["verdict"] == expected
        results.append({"case": name, "capsule": str(path), "verdict": expected})
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
