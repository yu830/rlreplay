"""Budget recovery with controlled time and a real writer integration."""

import os
import queue
import time
import uuid
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace

import pytest

from rlreplay import capture
from rlreplay._receipts import unpack
from rlreplay.artifacts import encode, read
from rlreplay.capture import Recorder


def test_hundred_full_delete_recover_cycles(evidence, tmp_path, monkeypatch):
    payload = encode(evidence)
    current = tmp_path / "seed.json"
    current.write_bytes(payload)
    clock = [0.0]
    scans = []
    stored_bytes = capture._stored_bytes

    def scan(directory):
        scans.append(clock[0])
        return stored_bytes(directory)

    monkeypatch.setattr(capture, "_stored_bytes", scan)
    monkeypatch.setattr(
        capture,
        "time",
        SimpleNamespace(
            monotonic=lambda: clock[0], perf_counter_ns=time.perf_counter_ns
        ),
    )

    class Incoming:
        index = 0
        current_path = current

        def get(self, timeout):
            if self.index == 200:
                return None
            if self.index % 2:
                assert read(self.current_path) == evidence
                self.current_path.unlink()
                clock[0] += 1.0
            capture_id = str(uuid.uuid4())
            if self.index % 2:
                self.current_path = tmp_path / f"{capture_id}.json"
            self.index += 1
            return capture_id, evidence, time.perf_counter_ns()

    incoming, outgoing = Incoming(), queue.Queue()
    capture._consume(
        tmp_path,
        incoming,
        outgoing,
        len(payload),
        "json",
        False,
        SimpleNamespace(is_alive=lambda: True),
    )
    events = [unpack(outgoing.get_nowait()) for _ in range(200)]
    assert [event["state"] for event in events] == ["DROPPED", "COMPLETE"] * 100
    assert all(event["reason"] == "storage_budget" for event in events[::2])
    assert len(scans) == 102  # startup, first full check, then one per recovery
    assert all(b - a >= 1.0 for a, b in pairwise(scans[1:]))
    assert read(incoming.current_path) == evidence


def test_under_budget_never_rescans(evidence, tmp_path, monkeypatch):
    scans = []
    stored_bytes = capture._stored_bytes

    def scan(directory):
        scans.append(directory)
        return stored_bytes(directory)

    monkeypatch.setattr(capture, "_stored_bytes", scan)
    incoming, outgoing = queue.Queue(), queue.Queue()
    for _ in range(3):
        incoming.put((str(uuid.uuid4()), evidence, time.perf_counter_ns()))
    incoming.put(None)
    capture._consume(
        tmp_path,
        incoming,
        outgoing,
        100000,
        "json",
        False,
        SimpleNamespace(is_alive=lambda: True),
    )
    assert len(scans) == 1
    assert all(unpack(outgoing.get_nowait())["state"] == "COMPLETE" for _ in range(3))


def test_budget_scan_propagates_real_filesystem_error(evidence, tmp_path):
    directory = tmp_path / "capture"
    directory.mkdir()
    (directory / "seed.json").write_bytes(encode(evidence))

    class Incoming:
        first = True

        def get(self, timeout):
            if not self.first:
                return None
            self.first = False
            directory.rename(tmp_path / "moved")
            directory.write_text("not a directory", encoding="utf-8")
            return str(uuid.uuid4()), evidence, time.perf_counter_ns()

    outgoing = queue.Queue()
    capture._consume(
        directory,
        Incoming(),
        outgoing,
        len(encode(evidence)),
        "json",
        False,
        SimpleNamespace(is_alive=lambda: True),
    )
    event = unpack(outgoing.get_nowait())
    assert (event["state"], event["phase"], event["reason"]) == (
        "ERROR",
        "budget",
        "NotADirectoryError",
    )


def test_deleted_entry_during_scan_is_ignored(tmp_path, monkeypatch):
    vanished = tmp_path / "vanished.json"
    kept = tmp_path / "kept.partial"
    vanished.write_bytes(b"x")
    kept.write_bytes(b"abc")
    scandir = capture.os.scandir

    class Entries:
        def __enter__(self):
            with scandir(tmp_path) as entries:
                self.entries = list(entries)
            vanished.unlink()
            return iter(self.entries)

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(capture.os, "scandir", lambda _: Entries())
    assert capture._stored_bytes(tmp_path) == 3


def test_storage_scan_matches_platform_suffix_case_rules(tmp_path):
    (tmp_path / "evidence.JSON").write_bytes(b"example")
    (tmp_path / "incomplete.PARTIAL").write_bytes(b"partial")
    expected = 14 if os.name == "nt" else 0
    assert capture._stored_bytes(tmp_path) == expected


def test_real_recorder_recovers_after_external_cleanup(evidence, tmp_path):
    recorder = Recorder(tmp_path, storage_budget=len(encode(evidence)))

    def await_receipt(capture_id):
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            for event in recorder.poll():
                if event.get("id") == capture_id:
                    return event
            time.sleep(0.01)
        pytest.fail(f"receipt missing: {recorder.status()}")

    try:
        first = await_receipt(recorder.submit(evidence))
        assert first["state"] == "COMPLETE"
        dropped = await_receipt(recorder.submit(evidence))
        assert (dropped["state"], dropped["reason"]) == ("DROPPED", "storage_budget")
        Path(first["path"]).unlink()
        time.sleep(1.05)
        recovered = await_receipt(recorder.submit(evidence))
        assert recovered["state"] == "COMPLETE"
        assert read(recovered["path"]) == evidence
        assert recorder.status()["pending"] == recorder.status()["reserved"] == 0
    finally:
        final = recorder.close()
    assert not any(event["state"] in {"ERROR", "PARTIAL"} for event in final)
    assert recorder._process._closed
