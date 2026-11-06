"""Preserve unexpected CLI diagnostics in the public known-case runner."""

import json
import runpy
import subprocess
from pathlib import Path

import pytest

WARNING = "OpenBLAS WARNING - could not determine the L2 cache size\n"
ERROR = '{"state":"ERROR","reason":"invalid capsule"}'
PASS = '{"finding":{"verdict":"PASS"}}'


@pytest.mark.parametrize(
    "expected,returncode,stdout,stderr,observed",
    [
        ("PASS", 4, "", '{"state":"ERROR","reason":"missing dependency"}', "ERROR"),
        ("ERROR", 0, '{"finding":{"verdict":"PASS"}}', "", "PASS"),
        ("PASS", 9, "unexpected output", "process failure", "UNPARSEABLE"),
    ],
)
def test_unexpected_cli_result_keeps_original_output(
    tmp_path, monkeypatch, expected, returncode, stdout, stderr, observed
):
    tool = Path(__file__).resolve().parents[1] / "tools/validate_known_cases.py"
    execute = runpy.run_path(str(tool))["execute"]
    capsule = tmp_path / "input.json"
    capsule.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], returncode, stdout, stderr
        ),
    )
    result = execute(
        "check", capsule, tmp_path, "unexpected-cli", "control", "json", expected
    )
    assert result["observed"] == observed
    assert result["matches"] is False
    stored = json.loads((tmp_path / "commands.jsonl").read_text(encoding="utf-8"))
    assert stored["exit"] == returncode
    assert stored["stdout"] == stdout
    assert stored["stderr"] == stderr


@pytest.mark.parametrize(
    "stdout,stderr,returncode,expected,observed,matches",
    [
        ("", ERROR, 4, "ERROR", "ERROR", True),
        ("", WARNING + ERROR + "\n\n", 4, "ERROR", "ERROR", True),
        ("", WARNING * 2 + ERROR, 4, "ERROR", "ERROR", True),
        (PASS, WARNING, 0, "PASS", "PASS", True),
        (PASS, '{"level":"warning","message":"diagnostic"}', 0, "PASS", "PASS", True),
        ("", WARNING, 4, "ERROR", "UNPARSEABLE", False),
        ("", ERROR[:-1], 4, "ERROR", "UNPARSEABLE", False),
        ("", '{"state":"ERROR"}', 4, "ERROR", "UNPARSEABLE", False),
        ("noise", ERROR, 4, "ERROR", "UNPARSEABLE", False),
        ("warning\n" + PASS, "", 0, "PASS", "UNPARSEABLE", False),
        ('{"finding":{"verdict":"UNKNOWN"}}', "", 0, "PASS", "UNPARSEABLE", False),
        (PASS, ERROR, 0, "PASS", "AMBIGUOUS", False),
        ("", ERROR + "\n" + ERROR, 4, "ERROR", "AMBIGUOUS", False),
        ("", PASS + "\n" + ERROR, 4, "ERROR", "AMBIGUOUS", False),
        ("", ERROR + "\n" + WARNING, 4, "ERROR", "UNPARSEABLE", False),
        ("", WARNING + ERROR, 0, "ERROR", "ERROR", False),
        (PASS, WARNING, 4, "PASS", "PASS", False),
    ],
)
def test_result_framing_keeps_diagnostics_and_requires_matching_exit(
    tmp_path, monkeypatch, stdout, stderr, returncode, expected, observed, matches
):
    tool = Path(__file__).resolve().parents[1] / "tools/validate_known_cases.py"
    execute = runpy.run_path(str(tool))["execute"]
    capsule = tmp_path / "input.json"
    capsule.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], returncode, stdout, stderr
        ),
    )
    result = execute("check", capsule, tmp_path, "framing", "control", "json", expected)
    assert result["observed"] == observed
    assert result["matches"] is matches
    stored = json.loads((tmp_path / "commands.jsonl").read_text(encoding="utf-8"))
    assert stored["stdout"] == stdout
    assert stored["stderr"] == stderr
