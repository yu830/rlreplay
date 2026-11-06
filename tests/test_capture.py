"""Capture queue, writer lifecycle and real filesystem failure behavior."""

import queue
import time
from dataclasses import replace

import pytest

from rlreplay.artifacts import read
from rlreplay.capture import Recorder


def test_actual_writer_completion(evidence, tmp_path):
    recorder = Recorder(tmp_path)
    capture_id = recorder.submit(evidence)
    events = recorder.close()
    written = [event for event in events if event["state"] == "COMPLETE"]
    assert len(written) == 1
    assert written[0]["id"] == capture_id
    assert read(written[0]["path"]) == evidence
    assert list(tmp_path.glob("*.partial")) == []
    assert recorder.close() == []


def test_queue_full_is_reported(evidence, tmp_path, monkeypatch):
    recorder = Recorder(tmp_path)

    def full(_message):
        raise queue.Full

    with monkeypatch.context() as context:
        context.setattr(recorder._incoming, "put_nowait", full)
        assert recorder.submit(evidence) is None
    events = recorder.close()
    assert any(event.get("reason") == "queue_full" for event in events)


def test_storage_budget_and_disk_error(evidence, tmp_path):
    recorder = Recorder(tmp_path / "budget", storage_budget=1)
    recorder.submit(evidence)
    events = recorder.close()
    assert any(event.get("reason") == "storage_budget" for event in events)
    occupied = tmp_path / "occupied"
    occupied.write_text("a regular file", encoding="utf-8")
    recorder = Recorder(occupied)
    recorder.submit(evidence)
    events = recorder.close()
    assert any(event["state"] == "ERROR" for event in events)
    assert not any(event["state"] == "COMPLETE" for event in events)


def test_worker_exit_has_no_false_completion(evidence, tmp_path):
    recorder = Recorder(tmp_path)
    recorder._process.terminate()
    recorder._process.join()
    assert recorder.submit(evidence) is None
    assert any(
        event.get("reason") == "writer_unavailable" for event in recorder.close()
    )


def test_admission_precedes_encoding(evidence, tmp_path, monkeypatch):
    recorder = Recorder(tmp_path, queue_capacity=1)
    first = recorder.reserve()
    second = recorder.reserve()
    assert first is not None and second is not None

    def must_not_freeze(*_args):
        pytest.fail("snapshot ran without available admission capacity")

    with monkeypatch.context() as context:
        context.setattr("rlreplay.capture.freeze", must_not_freeze)
        assert recorder.submit(evidence) is None
    recorder.cancel(first, reason="test_cancel")
    assert recorder.submit(evidence) is not None
    events = recorder.close()
    assert any(event.get("reason") == "inflight_budget" for event in events)
    assert any(event["state"] == "COMPLETE" for event in events)


def test_async_encoding_failure_releases_credit(evidence, tmp_path):
    recorder = Recorder(tmp_path, queue_capacity=1)
    invalid = replace(evidence, provenance={**evidence.provenance, "unknown": "field"})
    capture_id = recorder.submit(invalid)
    events = []
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and not any(
        e.get("id") == capture_id for e in events
    ):
        events.extend(recorder.poll())
        time.sleep(0.01)
    assert any(
        event.get("id") == capture_id
        and event["state"] == "ERROR"
        and event["phase"] == "encode"
        for event in events
    )
    assert recorder.status()["pending"] == 0
    assert recorder.submit(evidence) is not None
    assert any(event["state"] == "COMPLETE" for event in recorder.close())


def test_retained_closed_recorder_releases_ipc_objects(evidence, tmp_path):
    recorder = Recorder(tmp_path)
    recorder.submit(evidence)
    assert any(event["state"] == "COMPLETE" for event in recorder.close())
    assert recorder._incoming is None
    assert recorder._outgoing is None
    assert recorder._process._closed
    assert recorder.poll() == []
    assert recorder.submit(evidence) is None
    assert recorder.poll() == [{"state": "ERROR", "reason": "writer_unavailable"}]
