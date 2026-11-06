"""Run the fixed synthetic controls through the installed package's normal CLI."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

from rlreplay import __version__
from rlreplay.artifacts import encode, prepare_codec, write
from rlreplay.core import Evidence

FIXTURE = Path(__file__).resolve().parents[1] / "tests/fixtures/public-cases.json"
EXITS = {
    "PASS": 0,
    "CONTRACT_VIOLATION": 1,
    "SUSPECTED_MISMATCH": 2,
    "INCONCLUSIVE": 3,
    "ERROR": 4,
}


def execute(command, capsule, destination, case, category, codec, verdict):
    arguments = [sys.executable, "-m", "rlreplay", command, str(capsule.resolve())]
    process = subprocess.run(
        arguments,
        cwd=destination,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    expected_exit = EXITS[verdict]
    result = {
        "case": case,
        "category": category,
        "codec": codec,
        "command": command,
        "capsule": capsule.relative_to(destination).as_posix(),
        "expected": verdict,
        "expected_exit": expected_exit,
        "exit": process.returncode,
        "stdout": process.stdout,
        "stderr": process.stderr,
        "capsule_sha256": hashlib.sha256(capsule.read_bytes()).hexdigest(),
    }
    parsed, parse_errors, observations = {}, {}, []
    for name, stream in (("stdout", process.stdout), ("stderr", process.stderr)):
        if not stream.strip():
            continue
        try:
            parsed[name] = json.loads(stream)
        except json.JSONDecodeError:
            parse_errors[name] = "invalid_json"
    stdout, stderr = parsed.get("stdout"), parsed.get("stderr")
    if isinstance(stdout, dict) and isinstance(stdout.get("finding"), dict):
        actual = stdout["finding"].get("verdict")
        if isinstance(actual, str):
            observations.append(actual)
    if isinstance(stderr, dict) and stderr.get("state") == "ERROR":
        observations.append("ERROR")
    observed = (
        observations[0]
        if len(set(observations)) == 1
        else ("AMBIGUOUS" if observations else "UNPARSEABLE")
    )
    result.update(
        observed=observed,
        matches=observed == verdict and process.returncode == expected_exit,
        parse_errors=parse_errors,
    )
    with (destination / "commands.jsonl").open("a", encoding="utf-8") as output:
        output.write(json.dumps(result) + "\n")
    return result


def validate(destination, *, codec="both", fixture=FIXTURE):
    # Valid binary construction must succeed before testing corrupt binary bytes.
    if codec == "both":
        prepare_codec("safetensors")
    source = json.loads(fixture.read_text(encoding="utf-8"))
    if source["format"] != "rlreplay-public-known-cases/1":
        raise ValueError("unsupported known-case fixture")
    cases = source["cases"]
    categories = Counter(item["category"] for item in cases)
    if len(cases) != 20 or categories != source["expected_categories"]:
        raise ValueError("known-case inventory differs from the fixed 20-case set")
    evidence = {
        item["id"]: Evidence.from_dict(item["evidence"])
        for item in cases
        if item["kind"] == "evidence"
    }
    records, case_results, skipped = [], [], []
    for index, item in enumerate(cases):
        codecs = (
            ("json", "safetensors") if item["kind"] == "evidence" else (item["codec"],)
        )
        variants = []
        for current in codecs:
            if current == "safetensors" and codec == "json":
                skipped.append(
                    {
                        "case": item["id"],
                        "codec": current,
                        "state": "NOT_RUN",
                        "reason": "binary_not_requested",
                    }
                )
                variants.append({"codec": current, "state": "NOT_RUN"})
                continue
            directory = destination / f"case-{index:02d}" / current
            if item["kind"] == "evidence":
                capsule = write(evidence[item["id"]], directory, codec=current)
            else:
                directory.mkdir(parents=True)
                payload = encode(evidence[item["base_case"]], codec=current)
                if item["kind"] == "digest_mismatch":
                    altered = json.loads(payload)
                    altered["evidence"]["samples"][0]["reward"] = 999
                    payload = json.dumps(altered).encode()
                elif item["kind"] == "truncated":
                    payload = payload[: -item["remove_bytes"]]
                else:
                    raise ValueError("unsupported malformed control")
                capsule = directory / (
                    "malformed.json" if current == "json" else "malformed.rlrp"
                )
                capsule.write_bytes(payload)
            runs = [
                execute(
                    command,
                    capsule,
                    destination,
                    item["id"],
                    item["category"],
                    current,
                    item["expected_verdict"],
                )
                for command in ("check", "replay")
            ]
            records.extend(runs)
            variants.append(
                {
                    "codec": current,
                    "state": "PASS" if all(r["matches"] for r in runs) else "FAIL",
                }
            )
        case_results.append(
            {
                "case": item["id"],
                "category": item["category"],
                "expected": item["expected_verdict"],
                "variants": variants,
            }
        )
    success = all(r["matches"] for r in records)
    result = {
        "status": "FAIL" if not success else ("PARTIAL" if skipped else "PASS"),
        "scope_complete": not skipped,
        "package_version": __version__,
        "fixture_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
        "unique_cases": 20,
        "case_categories": dict(categories),
        "acceptance_command_runs": len(records),
        "matched_acceptance_commands": sum(r["matches"] for r in records),
        "full_protocol_commands": 76,
        "not_run": skipped,
        "cases": case_results,
        "scope": "fixed synthetic controls through check and semantic replay; no production accuracy estimate or native training replay",
    }
    (destination / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codec", choices=("both", "json"), default="both")
    parser.add_argument(
        "--output",
        type=Path,
        help="New directory retaining capsules and command logs; otherwise use a temporary directory.",
    )
    parser.add_argument("--fixtures", type=Path, default=FIXTURE)
    args = parser.parse_args()
    try:
        if args.output is None:
            with tempfile.TemporaryDirectory(
                prefix="rlreplay-known-cases-"
            ) as temporary:
                result = validate(
                    Path(temporary), codec=args.codec, fixture=args.fixtures
                )
        else:
            args.output.mkdir(parents=True)
            result = validate(
                args.output.resolve(), codec=args.codec, fixture=args.fixtures
            )
    except (
        ImportError,
        OSError,
        ValueError,
        TypeError,
        KeyError,
        subprocess.TimeoutExpired,
    ) as error:
        print(
            json.dumps(
                {
                    "status": "ERROR",
                    "scope_complete": False,
                    "error_type": type(error).__name__,
                    "reason": str(error),
                }
            ),
            file=sys.stderr,
        )
        return 4
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "status",
                    "scope_complete",
                    "package_version",
                    "unique_cases",
                    "case_categories",
                    "acceptance_command_runs",
                    "matched_acceptance_commands",
                    "full_protocol_commands",
                    "not_run",
                )
            },
            indent=2,
        )
    )
    return 0 if result["status"] in ("PASS", "PARTIAL") else 1


if __name__ == "__main__":
    raise SystemExit(main())
