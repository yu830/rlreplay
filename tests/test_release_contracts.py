"""Regression cases for the 0.1.0 CLI, diff and reference boundaries."""

import hashlib
import json
import math
import subprocess
import sys
from dataclasses import replace

import pytest

from rlreplay.artifacts import decode, encode, write
from rlreplay.contracts.grpo import check
from rlreplay.core import Profile, Sample
from rlreplay.replay import diff


@pytest.mark.parametrize(
    "case",
    [
        "invalid_json",
        "deep_json",
        "missing_file",
        "invalid_schema",
        "digest_mismatch",
        "unknown_option",
        "missing_required_argument",
        "native_runtime_error",
        "output_serialization_error",
    ],
)
def test_cli_failures_are_not_numeric_verdicts(evidence, tmp_path, case):
    path = write(evidence, tmp_path / "capsules")
    command = [sys.executable, "-m", "rlreplay", "check", str(path)]
    if case in {"invalid_json", "deep_json", "invalid_schema", "digest_mismatch"}:
        malformed = tmp_path / "malformed.json"
        if case == "invalid_json":
            malformed.write_text("{", encoding="utf-8")
        elif case == "deep_json":
            malformed.write_text("[" * 200000 + "]" * 200000, encoding="utf-8")
        elif case == "invalid_schema":
            document = json.loads(path.read_bytes())
            document["evidence"]["schema_version"] = 999
            payload = json.dumps(
                document["evidence"], sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
            document["sha256"] = hashlib.sha256(payload).hexdigest()
            malformed.write_text(json.dumps(document), encoding="utf-8")
        else:
            document = json.loads(path.read_bytes())
            document["evidence"]["profile"]["normalize_std"] = False
            malformed.write_text(json.dumps(document), encoding="utf-8")
        command[-1] = str(malformed)
    elif case == "missing_file":
        command[-1] = str(tmp_path / "missing.json")
    elif case == "unknown_option":
        command.append("--not-an-option")
    elif case == "missing_required_argument":
        command.pop()
    elif case == "native_runtime_error":
        command = [
            sys.executable,
            "-c",
            (
                "import sys, rlreplay.cli as c; "
                "exec('def fail(*a, **k):\\n raise RuntimeError(\\\"native failure\\\")'); "
                "c.run_verl=fail; sys.exit(c.main(['replay', sys.argv[1], '--runner', 'verl']))"
            ),
            str(path),
        ]
    else:
        command = [
            sys.executable,
            "-c",
            (
                "import sys, rlreplay.cli as c; "
                "c.semantic_replay=lambda e: {'bad': float('nan')}; "
                "sys.exit(c.main(['replay', sys.argv[1]]))"
            ),
            str(path),
        ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 4, (case, result.stdout, result.stderr)
    assert result.stdout == ""
    error = json.loads(result.stderr)
    assert error["state"] == "ERROR"
    assert error["reason"]
    if case == "invalid_schema":
        assert error["reason"] == "unsupported evidence schema"
    if case == "digest_mismatch":
        assert error["reason"] == "capsule digest mismatch"
    if case == "native_runtime_error":
        assert error["type"] == "RuntimeError"
        assert error["reason"] == "native failure"


def test_cli_help_version_and_interrupt(evidence, tmp_path, monkeypatch):
    from rlreplay import cli

    for option in ("--help", "--version"):
        result = subprocess.run(
            [sys.executable, "-m", "rlreplay", option],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0
        if option == "--version":
            assert result.stdout.strip() == "rlreplay 0.1.0"

    def interrupt(_):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "read", interrupt)
    with pytest.raises(KeyboardInterrupt):
        cli.main(["check", "unused"])


@pytest.mark.parametrize("codec", ["json", "safetensors"])
@pytest.mark.parametrize(
    "left_value,right_value,reorder,changed",
    [
        (float("nan"), float("nan"), False, []),
        (1.0, float("nan"), False, ["a0"]),
        (float("nan"), float("inf"), False, ["a0"]),
        (float("inf"), -float("inf"), False, ["a0"]),
        (1.0, 2.0, False, ["a0"]),
        (float("nan"), float("nan"), True, []),
        (0.0, -0.0, False, []),
    ],
    ids=[
        "same-nan",
        "finite-nan",
        "nan-inf",
        "inf-sign",
        "finite-change",
        "reorder",
        "signed-zero",
    ],
)
def test_diff_compares_captured_values(
    evidence, codec, left_value, right_value, reorder, changed
):
    def make(value):
        return replace(
            evidence,
            samples=(
                replace(evidence.samples[0], advantage=(value, 0)),
                evidence.samples[1],
            ),
        )

    left = decode(encode(make(left_value), codec=codec))
    right = decode(encode(make(right_value), codec=codec))
    if reorder:
        right = replace(right, samples=tuple(reversed(right.samples)))
    result = diff(left, right)
    assert result["changed_samples"] == changed
    if math.isnan(left_value):  # Original NaN remains a diagnosed fault.
        assert result["left"]["reason"] == "nonfinite"


@pytest.mark.parametrize(
    "rewards,advantages,normalize",
    [
        ((3e38, 3e38), (0.5, 0.5), True),
        ((1e20, -1e20), (0.0, 0.0), True),
        ((3e38, 3e38), (1e30, -1e30), False),
    ],
    ids=["sum-overflow", "variance-overflow", "no-std-overflow"],
)
def test_binary32_intermediate_overflow_is_inconclusive(
    evidence, rewards, advantages, normalize
):
    rows = tuple(
        Sample(f"a{index}", "a", reward, (1,), (advantage,))
        for index, (reward, advantage) in enumerate(
            zip(rewards, advantages, strict=True)
        )
    )
    result = check(
        replace(evidence, profile=Profile(normalize_std=normalize), samples=rows)
    )
    assert (result.verdict, result.reason) == (
        "INCONCLUSIVE",
        "binary32_reference_out_of_range",
    )
