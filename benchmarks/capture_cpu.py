"""Small descriptive CPU capture run with synthetic binary32 Evidence."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.resources
import json
import math
import platform
import struct
import tempfile
import time
from collections import Counter
from pathlib import Path

from rlreplay import __version__
from rlreplay.artifacts import prepare_codec, read
from rlreplay.capture import Recorder
from rlreplay.contracts.grpo import check
from rlreplay.core import Evidence, Profile, Sample


def synthetic_evidence(groups, width):
    f32 = lambda value: struct.unpack("f", struct.pack("f", value))[0]
    rows = tuple(
        Sample(
            f"benchmark-sample-{g}-{i}",
            f"benchmark-group-{g}",
            float(reward),
            (1.0,) * width,
            (f32((reward - 2) / (math.sqrt(2) + 1e-6)),) * width,
        )
        for g in range(groups)
        for i, reward in enumerate((1, 3))
    )
    return Evidence(
        Profile(implementation="synthetic_stdlib_grpo"),
        rows,
        tuple((row.sample_id, row.group_id) for row in rows),
        {
            "framework": "synthetic",
            "framework_revision": "public-benchmark-v1",
            "algorithm_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "phase": "generated-outcome-advantage",
            "receipt_origin": "stdlib_generated_binary32_values",
            "device": "cpu",
            "dtype": "float32",
        },
    )


def percentiles(values):
    ordered = sorted(values)
    return {
        "n": len(values),
        **{
            f"p{p}_ms": ordered[math.ceil(len(ordered) * p / 100) - 1] / 1e6
            if ordered
            else None
            for p in (50, 95, 99)
        },
    }


def source_hashes():
    package = importlib.resources.files("rlreplay")
    names = (
        "__init__.py",
        "core.py",
        "capture.py",
        "artifacts.py",
        "_directory.py",
        "_snapshot.py",
        "_receipts.py",
        "_safetensors.py",
        "contracts/grpo.py",
    )
    return {
        f"rlreplay/{name}": hashlib.sha256(
            package.joinpath(name).read_bytes()
        ).hexdigest()
        for name in names
    }


def run_case(destination, codec, evidence, captures, interval_ms):
    recorder = Recorder(
        destination, codec=codec, storage_budget=64 * 1024 * 1024, check=False
    )
    warm_id = recorder.submit(evidence)
    deadline, warm = time.monotonic() + 5, []
    while recorder.status()["pending"] and time.monotonic() < deadline:
        warm.extend(recorder.poll())
        time.sleep(0.001)
    warm.extend(recorder.poll())
    if not any(e["state"] == "COMPLETE" and e.get("id") == warm_id for e in warm):
        recorder.close()
        raise RuntimeError("warmup capture did not complete")
    submitted, events, raw_caller, completion_ns = {}, [], [], []
    started = time.perf_counter_ns()

    def collect(batch):
        now = time.perf_counter_ns()
        for event in batch:
            events.append(event)
            if event["state"] == "COMPLETE":
                completion_ns.append(now - submitted[event["id"]])

    try:
        for i in range(captures):
            rest = started + int(i * interval_ms * 1e6) - time.perf_counter_ns()
            if rest > 0:
                time.sleep(rest / 1e9)
            tick = time.perf_counter_ns()
            capture_id = recorder.submit(evidence)
            raw_caller.append(time.perf_counter_ns() - tick)
            if capture_id is not None:
                submitted[capture_id] = tick
            collect(recorder.poll())
        collect(recorder.close())
    except BaseException:
        recorder.close()
        raise
    elapsed_ns = time.perf_counter_ns() - started
    terminal = Counter(e["state"] for e in events)
    complete = [e for e in events if e["state"] == "COMPLETE"]
    if len({e["id"] for e in complete}) != len(complete):
        raise RuntimeError("duplicate COMPLETE receipt")
    expected = json.dumps(evidence.to_dict(), sort_keys=True, allow_nan=False)
    findings = Counter()
    for event in complete:
        restored = read(event["path"])
        if json.dumps(restored.to_dict(), sort_keys=True, allow_nan=False) != expected:
            raise RuntimeError("completed capsule differs from the synthetic input")
        finding = check(restored)
        findings[finding.verdict] += 1
        if finding.verdict != "PASS":
            raise RuntimeError("synthetic healthy capsule did not pass its contract")
    return {
        "codec": codec,
        "status": "COMPLETE"
        if terminal["COMPLETE"] == captures
        and not terminal["ERROR"]
        and not terminal["PARTIAL"]
        else "PARTIAL",
        "offered_calls": captures,
        "submitted": len(submitted),
        "terminal": dict(terminal),
        "verified_complete": len(complete),
        "findings": dict(findings),
        "durable_call_fraction": len(complete) / captures,
        "drop_fraction": terminal["DROPPED"] / captures,
        "caller": percentiles(raw_caller),
        "observed_completion_latency": percentiles(completion_ns),
        "caller_raw_ns": raw_caller,
        "observed_completion_raw_ns": completion_ns,
        "elapsed_seconds": elapsed_ns / 1e9,
        "complete_per_second": len(complete) / (elapsed_ns / 1e9),
        "published_bytes": sum(e["bytes"] for e in complete),
        "warmup_complete": 1,
        "percentile_method": "nearest rank over observed samples; small-run p99 is not a production tail estimate",
    }


def benchmark(destination, *, captures, groups, width, interval_ms, codec):
    codecs = ("json", "safetensors") if codec == "both" else (codec,)
    for current in codecs:
        prepare_codec(current)
    evidence = synthetic_evidence(groups, width)
    before = source_hashes()
    cases = [
        run_case(destination / current, current, evidence, captures, interval_ms)
        for current in codecs
    ]
    if before != source_hashes():
        raise RuntimeError("installed SDK source changed during the run")
    result = {
        "format": "rlreplay-public-cpu-capture/1",
        "package_version": __version__,
        "runtime": {
            "python": platform.python_version(),
            "os": platform.system(),
            "machine": platform.machine(),
            "binary_dependencies": {
                name: importlib.metadata.version(name)
                for name in ("numpy", "safetensors")
            }
            if "safetensors" in codecs
            else {},
        },
        "source_sha256": before,
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "config": {
            "captures_per_codec": captures,
            "groups": groups,
            "responses_per_group": 2,
            "width": width,
            "samples_per_capsule": len(evidence.samples),
            "interval_ms": interval_ms,
            "queue_capacity": 2,
            "storage_budget_bytes": 64 * 1024 * 1024,
            "check_in_worker": False,
        },
        "scope": "one local descriptive Recorder.submit run; input construction excluded, read/check verified; no framework, GPU or external baseline",
        "cases": cases,
    }
    (destination / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--codec", choices=("json", "safetensors", "both"), default="json"
    )
    parser.add_argument("--captures", type=int, default=32)
    parser.add_argument("--groups", type=int, default=2)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--interval-ms", type=float, default=20)
    parser.add_argument(
        "--output",
        type=Path,
        help="New directory retaining results and synthetic capsules; otherwise use a temporary directory.",
    )
    args = parser.parse_args()
    if (
        not 1 <= args.captures <= 10000
        or not 1 <= args.groups <= 16
        or not 1 <= args.width <= 4096
        or not math.isfinite(args.interval_ms)
        or args.interval_ms < 0
    ):
        parser.error(
            "captures/groups/width must be within 1..10000/16/4096; interval must be finite and nonnegative"
        )
    options = {
        "captures": args.captures,
        "groups": args.groups,
        "width": args.width,
        "interval_ms": args.interval_ms,
        "codec": args.codec,
    }
    if args.output is None:
        with tempfile.TemporaryDirectory(prefix="rlreplay-cpu-capture-") as temporary:
            result = benchmark(Path(temporary), **options)
    else:
        args.output.mkdir(parents=True)
        result = benchmark(args.output.resolve(), **options)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if all(c["status"] == "COMPLETE" for c in result["cases"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
