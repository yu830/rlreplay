"""Check the installed standard-library wheel through its synthetic CLI workflow."""

import argparse
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path

import rlreplay
import rlreplay.artifacts
import rlreplay.contracts.grpo


def cli(*arguments, expected):
    process = subprocess.run(
        [sys.executable, "-m", "rlreplay", *map(str, arguments)],
        capture_output=True,
        text=True,
        check=False,
    )
    if process.returncode != expected:
        raise RuntimeError(
            f"CLI {arguments} returned {process.returncode}, expected {expected}. "
            f"stdout={process.stdout} stderr={process.stderr}"
        )
    return json.loads(process.stdout)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--example", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert Path(rlreplay.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
    assert importlib.metadata.version("rlreplay") == rlreplay.__version__
    assert not {"numpy", "safetensors", "torch", "verl", "trl"} & sys.modules.keys()
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
    assert cli("inspect", healthy, expected=0)["complete_groups"] is True
    for command in ("check", "replay"):
        assert cli(command, healthy, expected=0)["finding"]["verdict"] == "PASS"
        assert (
            cli(command, faulty, expected=2)["finding"]["verdict"]
            == "SUSPECTED_MISMATCH"
        )
    assert cli("diff", healthy, faulty, expected=0)["changed_samples"] == ["s0"]
    print(json.dumps({"wheel_version": rlreplay.__version__, "synthetic_cli": "PASS"}))


if __name__ == "__main__":
    main()
