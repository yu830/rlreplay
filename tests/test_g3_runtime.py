"""Ownership, asynchronous checks and exclusive output directory contracts."""

import os
import queue
import time
import uuid
from dataclasses import replace
from multiprocessing.reduction import ForkingPickler
from types import SimpleNamespace

import pytest

from rlreplay._receipts import pack, unpack
from rlreplay._snapshot import SnapshotBudgetExceeded, freeze
from rlreplay.artifacts import read, write
from rlreplay.capture import Recorder, _consume


def wait_for(recorder, predicate):
    events = []
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        events.extend(recorder.poll())
        if predicate(events):
            return events
        time.sleep(0.01)
    pytest.fail(f"missing receipt: {events}, status={recorder.status()}")


def test_snapshot_owns_values_before_queue_feeder(evidence, tmp_path, monkeypatch):
    mask = [1, 0]
    advantages = list(evidence.samples[0].advantage)
    source = replace(
        evidence,
        samples=(
            replace(evidence.samples[0], mask=mask, advantage=advantages),
            evidence.samples[1],
        ),
    )
    recorder = Recorder(tmp_path)
    send = recorder._incoming.put_nowait
    held = []
    with monkeypatch.context() as context:
        context.setattr(recorder._incoming, "put_nowait", held.append)
        capture_id = recorder.submit(source)
    source.provenance["phase"] = "mutated-after-submit"
    mask[0] = 0
    advantages[0] = 99
    send(held[0])
    events = recorder.close()
    complete = next(event for event in events if event.get("id") == capture_id)
    assert complete["state"] == "COMPLETE"
    captured = read(complete["path"])
    assert captured.provenance["phase"] == "post_advantage"
    assert captured.samples[0].mask == (1, 0)
    assert captured.samples[0].advantage[0] == -0.7071062811869011


def test_snapshot_budget_and_unsupported_objects(evidence, tmp_path):
    recorder = Recorder(tmp_path / "budget", snapshot_budget=1024)
    assert recorder.submit(evidence) is None
    assert any(event.get("reason") == "snapshot_budget" for event in recorder.close())
    recorder = Recorder(tmp_path / "types")
    invalid = replace(
        evidence,
        samples=(
            replace(evidence.samples[0], advantage=(object(), 0)),
            evidence.samples[1],
        ),
    )
    with pytest.raises(TypeError, match="Python numbers"):
        recorder.submit(invalid)
    assert any(event.get("reason") == "snapshot_invalid" for event in recorder.close())


def test_worker_publishes_optional_finding(evidence, tmp_path):
    broken = replace(
        evidence,
        samples=(
            replace(evidence.samples[0], advantage=(-0.2, 0)),
            evidence.samples[1],
        ),
    )
    recorder = Recorder(tmp_path, check=True)
    recorder.submit(broken)
    completed = next(
        event for event in recorder.close() if event["state"] == "COMPLETE"
    )
    assert completed["check_state"] == "COMPLETE"
    assert completed["finding_summary"]["verdict"] == "SUSPECTED_MISMATCH"
    assert completed["finding_summary"]["details"] == "recompute_from_capsule"
    assert recorder.stats["finding_SUSPECTED_MISMATCH"] == 1
    assert read(completed["path"]) == broken


def test_directory_lease_prevents_competing_writers(evidence, tmp_path):
    first = Recorder(tmp_path)
    first.submit(evidence)
    wait_for(first, lambda events: any(e["state"] == "COMPLETE" for e in events))
    with pytest.raises(BlockingIOError):
        write(evidence, tmp_path)
    second = Recorder(tmp_path)
    second.submit(evidence)
    events = second.close()
    assert any(
        e["state"] == "ERROR" and e.get("reason") == "BlockingIOError" for e in events
    )
    assert not any(e["state"] == "COMPLETE" for e in events)
    first.close()
    assert write(evidence, tmp_path).exists()


def test_dead_worker_releases_lease_and_reports_pending(
    evidence, tmp_path, monkeypatch
):
    first = Recorder(tmp_path)
    first.submit(evidence)
    wait_for(first, lambda events: any(e["state"] == "COMPLETE" for e in events))
    with monkeypatch.context() as context:
        context.setattr(first._incoming, "put_nowait", lambda _item: None)
        pending = first.submit(evidence)
    first._process.terminate()
    first._process.join()
    events = first.poll()
    assert any(e.get("id") == pending and e["state"] == "PARTIAL" for e in events)
    assert first.status()["pending"] == 0
    first.close()
    second = Recorder(tmp_path)
    second.submit(evidence)
    assert any(e["state"] == "COMPLETE" for e in second.close())


@pytest.mark.skipif(os.name == "nt", reason="Windows uses inherited ACLs")
def test_new_posix_artifacts_are_owner_only(evidence, tmp_path):
    path = write(evidence, tmp_path / "private")
    assert path.stat().st_mode & 0o077 == 0
    assert path.parent.stat().st_mode & 0o077 == 0


def test_string_subclass_cannot_undercharge_snapshot(evidence):
    class HiddenLength(str):
        def __len__(self):
            return 0

    mutated = replace(
        evidence,
        samples=(
            replace(evidence.samples[0], sample_id=HiddenLength("x" * 10000)),
            evidence.samples[1],
        ),
    )
    with pytest.raises(SnapshotBudgetExceeded):
        freeze(mutated, 8192)


def test_partial_files_consume_storage_budget(evidence, tmp_path):
    partial = tmp_path / f"{uuid.uuid4()}.partial"
    partial.write_bytes(b"x" * 4096)
    recorder = Recorder(tmp_path, storage_budget=4096)
    recorder.submit(evidence)
    events = recorder.close()
    assert any(e.get("reason") == "storage_budget" for e in events)
    assert partial.stat().st_size == 4096
    assert not list(tmp_path.glob("*.json"))


def test_oracle_failure_preserves_capture_and_redacts_error(
    evidence, tmp_path, monkeypatch, caplog
):
    incoming, outgoing = queue.Queue(), queue.Queue()
    incoming.put((str(uuid.uuid4()), evidence, time.perf_counter_ns()))
    incoming.put(None)

    def broken(_evidence):
        raise RuntimeError("SECRET-DATA")

    monkeypatch.setattr("rlreplay.capture.check_grpo", broken)
    _consume(
        tmp_path,
        incoming,
        outgoing,
        100000,
        "json",
        True,
        SimpleNamespace(is_alive=lambda: True),
    )
    event = unpack(outgoing.get_nowait())
    assert event["state"] == "COMPLETE"
    assert event["check_state"] == "ERROR"
    assert "SECRET-DATA" not in caplog.text
    assert read(tmp_path / f"{event['id']}.json") == evidence


def test_large_diagnostics_use_bounded_atomic_receipts():
    event = {
        "id": str(uuid.uuid4()),
        "state": "ERROR",
        "reason": "E" * 10000,
        "phase": "write",
        "bytes": 4194304,
        "queue_wait_ns": 2**63 - 1,
        "encode_ns": 2**63 - 1,
        "write_ns": 2**63 - 1,
        "check_ns": 2**63 - 1,
        "check_state": "COMPLETE",
        "finding": {
            "verdict": "CONTRACT_VIOLATION",
            "reason": "R" * 10000,
            "sample_ids": ["S" * 1000000],
            "max_abs_error": 3.4e38,
            "checked_samples": 32768,
            "total_samples": 32768,
        },
        "check_error": "F" * 10000,
    }
    wire = pack(event)
    assert len(ForkingPickler.dumps(wire)) + 4 <= 512
    assert unpack(wire)["finding_summary"]["details"] == "recompute_from_capsule"
    assert b"SSSSS" not in wire
