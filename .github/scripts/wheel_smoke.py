"""Check the installed standard-library wheel through its synthetic CLI workflow."""

import argparse
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path

import rlreplay
import rlreplay.artifacts
import rlreplay.capture
import rlreplay.contracts.grpo


def invoke(entrypoint, *arguments, expected):
    process = subprocess.run(
        [*entrypoint, *map(str, arguments)],
        capture_output=True,
        text=True,
        check=False,
    )
    if process.returncode != expected:
        raise RuntimeError(
            f"CLI {arguments} returned {process.returncode}, expected {expected}. "
            f"stdout={process.stdout} stderr={process.stderr}"
        )
    return process.stdout


def cli(*arguments, expected):
    return json.loads(
        invoke([sys.executable, "-m", "rlreplay"], *arguments, expected=expected)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--example", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert Path(rlreplay.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
    assert importlib.metadata.version("rlreplay") == rlreplay.__version__
    assert not {"numpy", "safetensors", "torch", "verl", "trl"} & sys.modules.keys()
    console = Path(sys.executable).parent / (
        "rlreplay.exe" if os.name == "nt" else "rlreplay"
    )
    module = [sys.executable, "-m", "rlreplay"]
    for option in ("--help", "--version"):
        module_output = invoke(module, option, expected=0)
        assert invoke([console], option, expected=0) == module_output
        if option == "--help":
            assert all(
                command in module_output
                for command in ("inspect", "check", "replay", "diff", "export-test")
            )
        else:
            assert module_output.strip() == f"rlreplay {rlreplay.__version__}"
    generated = subprocess.run(
        [sys.executable, str(args.example), str(args.output)],
        capture_output=True,
        text=True,
        check=True,
    )
    cases = {item["case"]: item for item in json.loads(generated.stdout)}
    assert cases["healthy"]["verdict"] == "PASS"
    assert cases["faulty"]["verdict"] == "SUSPECTED_MISMATCH"
    healthy, faulty = cases["healthy"]["capsule"], cases["faulty"]["capsule"]
    assert json.loads(invoke([console], "check", healthy, expected=0)) == cli(
        "check", healthy, expected=0
    )
    assert cli("inspect", healthy, expected=0)["complete_groups"] is True
    for command in ("check", "replay"):
        assert cli(command, healthy, expected=0)["finding"]["verdict"] == "PASS"
        assert (
            cli(command, faulty, expected=2)["finding"]["verdict"]
            == "SUSPECTED_MISMATCH"
        )
    assert cli("diff", healthy, faulty, expected=0)["changed_samples"] == ["s0"]
    evidence = rlreplay.artifacts.read(healthy)
    recorder = rlreplay.capture.Recorder(
        args.output / "bounded-recorder",
        queue_capacity=64,
        inflight_snapshot_budget=64 * 1024 * 1024,
        io_concurrency=8,
    )
    assert recorder.max_inflight_snapshot_charge == 64 * 1024 * 1024
    assert recorder.io_concurrency == 8
    capture_id = recorder.submit(evidence)
    receipts = recorder.close(timeout=10)
    complete = [
        item
        for item in receipts
        if item.get("id") == capture_id and item["state"] == "COMPLETE"
    ]
    assert len(complete) == 1
    assert rlreplay.artifacts.read(complete[0]["path"]) == evidence
    assert recorder.status()["pending_snapshot_charge"] == 0
    print(json.dumps({"wheel_version": rlreplay.__version__, "synthetic_cli": "PASS"}))


if __name__ == "__main__":
    main()
