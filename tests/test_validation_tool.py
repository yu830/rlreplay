"""Preserve unexpected CLI diagnostics in the public known-case runner."""

import json
import runpy
import subprocess
from pathlib import Path

import pytest


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
