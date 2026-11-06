"""Independent anchors, real P0 regression values, and artifact boundaries."""

import json
import subprocess
import sys
import uuid
from dataclasses import replace

import pytest

from rlreplay.artifacts import decode, encode, read, write, write_payload
from rlreplay.contracts.grpo import check, reference
from rlreplay.core import Evidence, Profile, Sample
from rlreplay.replay import diff, export_test, run_verl


def test_independent_known_values(evidence):
    assert check(evidence).verdict == "PASS"
    assert reference(evidence)["a0"] == pytest.approx(
        (-0.7071062811869011, 0), abs=1e-12
    )
    unscaled = replace(
        evidence,
        profile=Profile(normalize_std=False),
        samples=(
            replace(evidence.samples[0], advantage=(-1.0, 0.0)),
            replace(evidence.samples[1], advantage=(1.0, 1.0)),
        ),
    )
    assert check(unscaled).verdict == "PASS"


def test_historical_variance_floor_detected(evidence):
    rows = (
        Sample("a0", "a", 1.0, (1,), (-0.005001788027584553,)),
        Sample("a1", "a", 1.0000100135803223, (1,), (0.005001788027584553,)),
    )
    broken = replace(evidence, samples=rows)
    assert check(broken).verdict == "SUSPECTED_MISMATCH"
    fixed = replace(
        broken,
        samples=(
            replace(rows[0], advantage=(-0.6196008324623108,)),
            replace(rows[1], advantage=(0.6196008324623108,)),
        ),
    )
    assert check(fixed).verdict == "PASS"


def test_float32_roundoff_has_explicit_uncertainty(evidence):
    rows = (
        Sample("a0", "a", 1.0, (1,), (0.0,)),
        Sample("a1", "a", 1.0000001192092896, (1,), (0.10651206225156784,)),
    )
    finding = check(replace(evidence, samples=rows))
    assert finding.verdict == "INCONCLUSIVE"
    assert finding.reason == "binary32_roundoff_explains_difference"


@pytest.mark.parametrize(
    "change,reason",
    [
        ({"group_id": "b"}, "group_identity"),
        ({"mask": (1, 2)}, "shape_or_mask"),
        ({"advantage": (-0.7071062811869011, 0.1)}, "padding_advantage"),
        ({"advantage": (float("nan"), 0)}, "nonfinite"),
        ({"reward": 1e300}, "outside_float32_range"),
    ],
)
def test_structural_corruption(evidence, change, reason):
    mutated = replace(
        evidence, samples=(replace(evidence.samples[0], **change), evidence.samples[1])
    )
    finding = check(mutated)
    assert (finding.verdict, finding.reason) == ("CONTRACT_VIOLATION", reason)


def test_incomplete_unknown_and_reordered(evidence):
    assert check(replace(evidence, complete_groups=False)).verdict == "INCONCLUSIVE"
    assert check(replace(evidence, provenance={})).verdict == "INCONCLUSIVE"
    assert (
        check(replace(evidence, profile=Profile(profile_id="unknown-profile"))).reason
        == "unsupported_profile"
    )
    permuted = replace(evidence, samples=tuple(reversed(evidence.samples)))
    assert check(permuted).verdict == "PASS"
    assert diff(evidence, permuted)["changed_samples"] == []


def test_capsule_round_trip_and_integrity(evidence, tmp_path):
    path = write(evidence, tmp_path)
    assert read(path) == evidence
    document = json.loads(path.read_bytes())
    document["evidence"]["profile"]["normalize_std"] = False
    with pytest.raises(ValueError, match="digest"):
        decode(json.dumps(document).encode())
    with pytest.raises(ValueError, match="duplicate"):
        decode(b'{"format":"a","format":"b"}')
    with pytest.raises(ValueError, match="byte budget"):
        decode(b" " * (4 * 1024 * 1024 + 1))


def test_untrusted_fields_and_invalid_shapes_rejected(evidence):
    document = json.loads(encode(evidence))
    # Shape faults remain inspectable. Malformed types and executable fields are rejected.
    malformed = replace(
        evidence,
        samples=(replace(evidence.samples[0], advantage=()), evidence.samples[1]),
    )
    assert check(decode(encode(malformed))).reason == "shape_or_mask"
    document["evidence"]["provenance"]["command"] = "untrusted-code"
    with pytest.raises(ValueError):
        Evidence.from_dict(document["evidence"])


def test_cli_and_regression_export(evidence, tmp_path):
    path = write(evidence, tmp_path / "capsules")
    result = subprocess.run(
        [sys.executable, "-m", "rlreplay", "check", str(path)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert json.loads(result.stdout)["finding"]["verdict"] == "PASS"
    exported = export_test(path, tmp_path / "regression")
    assert read(exported.with_name("capsule.json")) == evidence
    assert "run_verl(evidence, allow_revision_change=True)" in exported.read_text(
        encoding="utf-8"
    )
    with pytest.raises(FileExistsError):
        export_test(path, tmp_path / "regression")


@pytest.mark.parametrize("number", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_is_persisted_and_detected(evidence, number):
    mutated = replace(
        evidence,
        samples=(
            replace(evidence.samples[0], advantage=(number, 0)),
            evidence.samples[1],
        ),
    )
    capsule = encode(mutated)
    assert b'"nonfinite"' in capsule
    assert check(decode(capsule)).reason == "nonfinite"


def test_huge_integer_is_rejected_at_serialization(evidence):
    mutated = replace(
        evidence,
        samples=(replace(evidence.samples[0], reward=10**1000), evidence.samples[1]),
    )
    assert check(mutated).reason == "outside_float32_range"
    with pytest.raises(ValueError, match="float32 range"):
        encode(mutated)


def test_foreign_framework_cannot_select_verl_runner(evidence, tmp_path):
    foreign = replace(evidence, provenance={**evidence.provenance, "framework": "trl"})
    with pytest.raises(ValueError, match="verl runner requires a verl capture"):
        run_verl(foreign, allow_revision_change=True)
    path = write(foreign, tmp_path / "capsules")
    with pytest.raises(ValueError, match="verl driver capture"):
        export_test(path, tmp_path / "regression")


def test_schema_version_is_an_integer(evidence):
    value = json.loads(encode(evidence))["evidence"]
    value["schema_version"] = True
    with pytest.raises(ValueError, match="schema"):
        Evidence.from_dict(value)


def test_existing_capture_files_are_preserved(evidence, tmp_path):
    capture_id = str(uuid.uuid4())
    partial = tmp_path / f"{capture_id}.partial"
    partial.write_bytes(b"retained failure evidence")
    with pytest.raises(FileExistsError):
        write_payload(tmp_path, capture_id, encode(evidence))
    assert partial.read_bytes() == b"retained failure evidence"
    complete_id = str(uuid.uuid4())
    original = write_payload(tmp_path, complete_id, encode(evidence))
    original_bytes = original.read_bytes()
    with pytest.raises(FileExistsError, match="already published"):
        write_payload(tmp_path, complete_id, b"replacement")
    assert original.read_bytes() == original_bytes
