"""Concurrent publication preserves real receipts and bounded storage accounting."""

import os
import queue
import threading
import time
import uuid
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from rlreplay import capture
from rlreplay._receipts import unpack
from rlreplay.artifacts import encode, read
from rlreplay.capture import Recorder


def _messages(evidence, count):
    incoming = queue.Queue()
    ids = [str(uuid.uuid4()) for _ in range(count)]
    for capture_id in ids:
        incoming.put((capture_id, evidence, time.perf_counter_ns()))
    incoming.put(None)
    return incoming, ids


def _start_consumer(directory, incoming, outgoing, storage_budget, *, do_check=False):
    failures = []

    def run():
        try:
            capture._consume_concurrent(
                directory,
                incoming,
                outgoing,
                storage_budget,
                "json",
                do_check,
                SimpleNamespace(is_alive=lambda: True),
                2,
            )
        except Exception as error:  # noqa: BLE001 - re-raised on the test thread
            failures.append(error)

    thread = threading.Thread(target=run)
    thread.start()
    return thread, failures


def _finish_consumer(thread, failures):
    thread.join(timeout=10)
    assert not thread.is_alive(), "consumer failed to drain"
    if failures:
        raise failures[0]


def test_out_of_order_receipt_follows_real_publication(evidence, tmp_path, monkeypatch):
    incoming, ids = _messages(evidence, 2)
    outgoing = queue.Queue()
    entered, release = threading.Event(), threading.Event()
    original = capture.write_payload

    def hold_first(directory, capture_id, payload, *, suffix):
        if capture_id == ids[0]:
            entered.set()
            assert release.wait(10)
        return original(directory, capture_id, payload, suffix=suffix)

    monkeypatch.setattr(capture, "write_payload", hold_first)
    thread, failures = _start_consumer(tmp_path, incoming, outgoing, 1_000_000)
    try:
        assert entered.wait(10)
        first_receipt = unpack(outgoing.get(timeout=10))
        assert (first_receipt["id"], first_receipt["state"]) == (ids[1], "COMPLETE")
        assert read(tmp_path / f"{ids[1]}.json") == evidence
        assert not (tmp_path / f"{ids[0]}.json").exists()
    finally:
        release.set()
        _finish_consumer(thread, failures)
    second_receipt = unpack(outgoing.get_nowait())
    assert (second_receipt["id"], second_receipt["state"]) == (ids[0], "COMPLETE")
    assert read(tmp_path / f"{ids[0]}.json") == evidence


def test_failed_partial_is_counted_after_other_writes_drain(
    evidence, tmp_path, monkeypatch
):
    incoming, ids = _messages(evidence, 3)
    outgoing = queue.Queue()
    second_started, release_second = threading.Event(), threading.Event()
    original = capture.write_payload
    size = len(encode(evidence))

    def fail_one(directory, capture_id, payload, *, suffix):
        if capture_id == ids[0]:
            assert second_started.wait(10)
            (directory / f"{capture_id}.partial").write_bytes(b"x" * len(payload))
            raise OSError("controlled write failure")
        if capture_id == ids[1]:
            second_started.set()
            assert release_second.wait(10)
        return original(directory, capture_id, payload, suffix=suffix)

    monkeypatch.setattr(capture, "write_payload", fail_one)
    thread, failures = _start_consumer(tmp_path, incoming, outgoing, 2 * size)
    try:
        first = unpack(outgoing.get(timeout=10))
        assert (first["id"], first["state"], first["phase"]) == (
            ids[0],
            "ERROR",
            "write",
        )
    finally:
        release_second.set()
        _finish_consumer(thread, failures)
    later = [unpack(outgoing.get_nowait()), unpack(outgoing.get_nowait())]
    by_id = {event["id"]: event for event in later}
    assert by_id[ids[1]]["state"] == "COMPLETE"
    assert (by_id[ids[2]]["state"], by_id[ids[2]]["reason"]) == (
        "DROPPED",
        "storage_budget",
    )
    assert capture._stored_bytes(tmp_path) == 2 * size
    assert read(tmp_path / f"{ids[1]}.json") == evidence


def test_encoded_payloads_do_not_exceed_io_concurrency(evidence, tmp_path, monkeypatch):
    incoming, ids = _messages(evidence, 3)
    outgoing = queue.Queue()
    release, both_started = threading.Event(), threading.Event()
    original_write, original_encode = capture.write_payload, capture.encode
    lock = threading.Lock()
    encoded, started = [], []

    def count_encode(*args, **kwargs):
        encoded.append(1)
        return original_encode(*args, **kwargs)

    def hold_write(*args, **kwargs):
        with lock:
            started.append(1)
            if len(started) == 2:
                both_started.set()
        assert release.wait(10)
        return original_write(*args, **kwargs)

    monkeypatch.setattr(capture, "encode", count_encode)
    monkeypatch.setattr(capture, "write_payload", hold_write)
    thread, failures = _start_consumer(tmp_path, incoming, outgoing, 1_000_000)
    try:
        assert both_started.wait(10)
        thread.join(timeout=0.1)
        assert len(encoded) == 2
    finally:
        release.set()
        _finish_consumer(thread, failures)
    assert len(encoded) == 3
    receipts = [unpack(outgoing.get_nowait()) for _ in ids]
    assert {event["id"] for event in receipts if event["state"] == "COMPLETE"} == set(
        ids
    )


def test_concurrent_under_budget_does_not_rescan(evidence, tmp_path, monkeypatch):
    incoming, ids = _messages(evidence, 4)
    outgoing = queue.Queue()
    original = capture._stored_bytes
    scans = []

    def scan(directory):
        scans.append(directory)
        return original(directory)

    monkeypatch.setattr(capture, "_stored_bytes", scan)
    thread, failures = _start_consumer(tmp_path, incoming, outgoing, 1_000_000)
    _finish_consumer(thread, failures)
    receipts = [unpack(outgoing.get_nowait()) for _ in ids]
    assert {event["id"] for event in receipts if event["state"] == "COMPLETE"} == set(
        ids
    )
    assert len(scans) == 1


def test_concurrent_budget_scan_reports_filesystem_error(evidence, tmp_path):
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
    capture._consume_concurrent(
        directory,
        Incoming(),
        outgoing,
        len(encode(evidence)),
        "json",
        False,
        SimpleNamespace(is_alive=lambda: True),
        2,
    )
    receipt = unpack(outgoing.get_nowait())
    assert (receipt["state"], receipt["phase"], receipt["reason"]) == (
        "ERROR",
        "budget",
        "NotADirectoryError",
    )


def test_check_failure_is_reported_without_losing_published_file(
    evidence, tmp_path, monkeypatch
):
    incoming, ids = _messages(evidence, 1)
    outgoing = queue.Queue()

    def fail_check(_evidence):
        raise ValueError("controlled oracle failure")

    monkeypatch.setattr(capture, "check_grpo", fail_check)
    thread, failures = _start_consumer(
        tmp_path, incoming, outgoing, 1_000_000, do_check=True
    )
    _finish_consumer(thread, failures)
    receipt = unpack(outgoing.get_nowait())
    assert (receipt["id"], receipt["state"], receipt["check_state"]) == (
        ids[0],
        "COMPLETE",
        "ERROR",
    )
    assert receipt["check_error"] == "ValueError"
    assert read(tmp_path / f"{ids[0]}.json") == evidence


@pytest.mark.parametrize("codec", ["json", "safetensors"])
def test_real_concurrent_recorder_completes_and_drains(evidence, tmp_path, codec):
    if codec == "safetensors":
        pytest.importorskip("safetensors")
    recorder = Recorder(
        tmp_path,
        queue_capacity=8,
        inflight_snapshot_budget=64 * 1024 * 1024,
        io_concurrency=8,
        codec=codec,
        check=True,
    )
    try:
        ids = [recorder.submit(evidence) for _ in range(6)]
        assert all(ids)
        events = recorder.close(timeout=15)
    finally:
        recorder.close()
    complete = [event for event in events if event["state"] == "COMPLETE"]
    assert {event["id"] for event in complete} == set(ids)
    assert all(event["check_state"] == "COMPLETE" for event in complete)
    assert all(read(event["path"]) == evidence for event in complete)
    assert not any(event["state"] in {"ERROR", "PARTIAL"} for event in events)
    assert recorder.status()["pending"] == recorder.status()["reserved"] == 0
    assert recorder.status()["pending_snapshot_charge"] == 0
    assert recorder._process._closed


def test_concurrent_encode_error_does_not_prevent_next_capture(evidence, tmp_path):
    invalid = replace(
        evidence, provenance={**evidence.provenance, "invalid_field": "synthetic"}
    )
    recorder = Recorder(tmp_path, queue_capacity=4, io_concurrency=2)
    try:
        bad_id = recorder.submit(invalid)
        good_id = recorder.submit(evidence)
        assert bad_id and good_id
        events = recorder.close(timeout=15)
    finally:
        recorder.close()
    by_id = {event["id"]: event for event in events if "id" in event}
    assert (by_id[bad_id]["state"], by_id[bad_id]["phase"]) == ("ERROR", "encode")
    assert by_id[good_id]["state"] == "COMPLETE"
    assert read(by_id[good_id]["path"]) == evidence


def test_concurrent_recorder_recovers_after_external_cleanup(evidence, tmp_path):
    recorder = Recorder(
        tmp_path, storage_budget=len(encode(evidence)), io_concurrency=2
    )

    def receipt(capture_id):
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            for event in recorder.poll():
                if event.get("id") == capture_id:
                    return event
            time.sleep(0.01)
        pytest.fail(f"missing receipt for {capture_id}: {recorder.status()}")

    try:
        first = receipt(recorder.submit(evidence))
        assert first["state"] == "COMPLETE"
        dropped = receipt(recorder.submit(evidence))
        assert (dropped["state"], dropped["reason"]) == ("DROPPED", "storage_budget")
        Path(first["path"]).unlink()
        time.sleep(1.05)
        restored = receipt(recorder.submit(evidence))
        assert restored["state"] == "COMPLETE"
        assert read(restored["path"]) == evidence
    finally:
        final = recorder.close()
    assert not any(event["state"] in {"ERROR", "PARTIAL"} for event in final)
    assert recorder.status()["pending"] == recorder.status()["reserved"] == 0


def _paused_concurrent_writer(
    directory, incoming, outgoing, storage_budget, codec, do_check, io_concurrency
):
    from rlreplay import capture as worker

    original = worker.write_payload

    def pause(*args, **kwargs):
        Path(os.environ["RLREPLAY_CONCURRENT_PAUSE_MARKER"]).write_text(
            "paused", encoding="utf-8"
        )
        time.sleep(3)
        return original(*args, **kwargs)

    worker.write_payload = pause
    worker._writer(
        directory, incoming, outgoing, storage_budget, codec, do_check, io_concurrency
    )


def test_timeout_keeps_unpublished_captures_partial(evidence, tmp_path, monkeypatch):
    marker = tmp_path / "paused"
    monkeypatch.setenv("RLREPLAY_CONCURRENT_PAUSE_MARKER", str(marker))
    with monkeypatch.context() as context:
        context.setattr(capture, "_writer", _paused_concurrent_writer)
        recorder = Recorder(tmp_path / "capsules", queue_capacity=8, io_concurrency=2)
    try:
        ids = [recorder.submit(evidence)]
        deadline = time.monotonic() + 10
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert marker.exists()
        ids.extend(recorder.submit(evidence) for _ in range(2))
        assert all(ids)
        events = recorder.close(timeout=0.05)
        assert {event["id"] for event in events if event["state"] == "PARTIAL"} == set(
            ids
        )
        assert not any(event["state"] == "COMPLETE" for event in events)
        assert recorder.status()["pending_snapshot_charge"] == 0
        assert recorder._process._closed
    finally:
        recorder.close()


@pytest.mark.parametrize("value", [0, -1, 9, True, 1.2, None])
def test_invalid_io_concurrency_rejected(tmp_path, value):
    with pytest.raises(ValueError, match="io_concurrency"):
        Recorder(tmp_path, io_concurrency=value)
