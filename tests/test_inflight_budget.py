"""Aggregate snapshot admission, ownership and terminal credit release."""

import os
import queue
import time
from dataclasses import replace
from pathlib import Path

import pytest

from rlreplay import capture
from rlreplay._snapshot import freeze
from rlreplay.artifacts import read
from rlreplay.capture import Recorder


def wait_for(recorder, capture_id):
    deadline = time.monotonic() + 10
    events = []
    while time.monotonic() < deadline:
        events.extend(recorder.poll())
        if any(event.get("id") == capture_id for event in events):
            return events
        time.sleep(0.01)
    pytest.fail(f"missing receipt: {events}, status={recorder.status()}")


def paused_writer(directory, incoming, outgoing, storage_budget, codec, do_check):
    """Pause this test's actual writer before its file publication."""
    from rlreplay import capture as worker

    original_write = worker.write_payload
    original_writer = worker._writer

    def pause(*args, **kwargs):
        Path(os.environ["RLREPLAY_TEST_PAUSE_MARKER"]).write_text(
            "paused", encoding="utf-8"
        )
        time.sleep(3)
        return original_write(*args, **kwargs)

    worker.write_payload = pause
    original_writer(directory, incoming, outgoing, storage_budget, codec, do_check)


def test_total_budget_bounds_pending_and_releases_on_real_receipts(
    evidence, tmp_path, monkeypatch
):
    charge = freeze(evidence, 16 * 1024 * 1024)[1]
    budget = charge * 2 + charge // 2
    recorder = Recorder(tmp_path, queue_capacity=64, inflight_snapshot_budget=budget)
    send = recorder._incoming.put_nowait
    held = []
    try:
        with monkeypatch.context() as context:
            context.setattr(recorder._incoming, "put_nowait", held.append)
            first, second = recorder.submit(evidence), recorder.submit(evidence)
            assert first and second
            assert recorder.status()["pending_snapshot_charge"] == 2 * charge <= budget
            assert recorder.submit(evidence) is None
            assert len(held) == 2
            assert any(
                event.get("reason") == "inflight_snapshot_budget"
                for event in recorder.poll()
            )
        for item in held:
            send(item)
        events = wait_for(recorder, first)
        if not any(event.get("id") == second for event in events):
            events.extend(wait_for(recorder, second))
        completed = [event for event in events if event["state"] == "COMPLETE"]
        assert {event["id"] for event in completed} == {first, second}
        assert all(read(event["path"]) == evidence for event in completed)
        assert recorder.status()["pending_snapshot_charge"] == 0
        third = recorder.submit(evidence)
        assert third is not None
        assert any(
            event.get("id") == third and event["state"] == "COMPLETE"
            for event in recorder.close()
        )
    finally:
        recorder.close()
    assert recorder.status()["pending"] == recorder.status()["reserved"] == 0


def test_per_snapshot_limit_keeps_its_reason(evidence, tmp_path):
    charge = freeze(evidence, 16 * 1024 * 1024)[1]
    recorder = Recorder(
        tmp_path, snapshot_budget=charge - 1, inflight_snapshot_budget=64 * 1024 * 1024
    )
    assert recorder.submit(evidence) is None
    assert any(event.get("reason") == "snapshot_budget" for event in recorder.close())


def test_tighter_total_limit_controls_drop_reason(evidence, tmp_path):
    charge = freeze(evidence, 16 * 1024 * 1024)[1]
    recorder = Recorder(
        tmp_path, snapshot_budget=charge - 1, inflight_snapshot_budget=charge // 2
    )
    assert recorder.submit(evidence) is None
    assert any(
        event.get("reason") == "inflight_snapshot_budget" for event in recorder.close()
    )


def test_preexisting_reservations_share_total_budget(evidence, tmp_path, monkeypatch):
    charge = freeze(evidence, 16 * 1024 * 1024)[1]
    recorder = Recorder(
        tmp_path, queue_capacity=64, inflight_snapshot_budget=charge + charge // 2
    )
    send = recorder._incoming.put_nowait
    held = []
    try:
        first, second = recorder.reserve(), recorder.reserve()
        assert first and second
        with monkeypatch.context() as context:
            context.setattr(recorder._incoming, "put_nowait", held.append)
            assert recorder.submit(evidence, reservation=first) == first
            assert recorder.submit(evidence, reservation=second) is None
        state = recorder.status()
        assert state["reserved"] == 0
        assert state["pending_snapshot_charge"] == charge
        assert any(
            event.get("reason") == "inflight_snapshot_budget"
            for event in recorder.poll()
        )
        assert len(held) == 1
        send(held[0])
        assert any(
            event.get("id") == first and event["state"] == "COMPLETE"
            for event in recorder.close()
        )
    finally:
        recorder.close()


def test_queue_full_and_cancel_release_reservations(evidence, tmp_path, monkeypatch):
    recorder = Recorder(
        tmp_path, queue_capacity=64, inflight_snapshot_budget=64 * 1024 * 1024
    )

    def full(_message):
        raise queue.Full

    with monkeypatch.context() as context:
        context.setattr(recorder._incoming, "put_nowait", full)
        assert recorder.submit(evidence) is None
    reservation = recorder.reserve()
    assert reservation is not None
    recorder.cancel(reservation, reason="test_cancel")
    assert (
        recorder.status()["reserved"]
        == recorder.status()["pending_snapshot_charge"]
        == 0
    )
    assert any(event.get("reason") == "queue_full" for event in recorder.close())


def test_async_error_releases_charge_and_next_capture_succeeds(evidence, tmp_path):
    recorder = Recorder(
        tmp_path, queue_capacity=64, inflight_snapshot_budget=64 * 1024 * 1024
    )
    invalid = replace(
        evidence, provenance={**evidence.provenance, "invalid_field": "synthetic"}
    )
    bad_id = recorder.submit(invalid)
    events = wait_for(recorder, bad_id)
    assert any(
        event.get("id") == bad_id
        and event["state"] == "ERROR"
        and event["phase"] == "encode"
        for event in events
    )
    assert recorder.status()["pending_snapshot_charge"] == 0
    good_id = recorder.submit(evidence)
    assert any(
        event.get("id") == good_id and event["state"] == "COMPLETE"
        for event in recorder.close()
    )


@pytest.mark.parametrize("exception", [MemoryError, KeyboardInterrupt])
def test_unexpected_freeze_failure_releases_reservation(
    evidence, tmp_path, monkeypatch, exception
):
    recorder = Recorder(
        tmp_path, queue_capacity=64, inflight_snapshot_budget=64 * 1024 * 1024
    )

    def exhausted(*_args):
        raise exception("simulated allocation failure")

    try:
        with monkeypatch.context() as context:
            context.setattr(capture, "freeze", exhausted)
            with pytest.raises(exception, match="simulated allocation failure"):
                recorder.submit(evidence)
        assert recorder.status()["reserved"] == 0
        assert recorder.status()["pending_snapshot_charge"] == 0
        capture_id = recorder.submit(evidence)
        assert any(
            event.get("id") == capture_id and event["state"] == "COMPLETE"
            for event in recorder.close()
        )
    finally:
        recorder.close()


def test_close_timeout_during_real_writer_pause_is_partial(
    evidence, tmp_path, monkeypatch
):
    marker = tmp_path / "worker-paused"
    monkeypatch.setenv("RLREPLAY_TEST_PAUSE_MARKER", str(marker))
    with monkeypatch.context() as context:
        context.setattr(capture, "_writer", paused_writer)
        recorder = Recorder(
            tmp_path / "capsules",
            queue_capacity=64,
            inflight_snapshot_budget=64 * 1024 * 1024,
        )
    try:
        ids = [recorder.submit(evidence)]
        deadline = time.monotonic() + 10
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert marker.exists(), "worker did not enter the controlled pause"
        ids.extend(recorder.submit(evidence) for _ in range(3))
        assert all(ids) and recorder.status()["pending"] == 4
        assert not any(event["state"] == "COMPLETE" for event in recorder.poll())
        events = recorder.close(timeout=0.05)
        assert {event["id"] for event in events if event["state"] == "PARTIAL"} == set(
            ids
        )
        assert not any(event["state"] == "COMPLETE" for event in events)
        assert recorder.status()["pending_snapshot_charge"] == 0
        assert recorder._process._closed
    finally:
        recorder.close()


def test_default_total_bound_is_unchanged(tmp_path):
    recorder = Recorder(tmp_path, queue_capacity=2)
    assert recorder.max_inflight_snapshot_charge == 3 * 16 * 1024 * 1024
    recorder.close()
    larger = Recorder(tmp_path / "larger", queue_capacity=64)
    assert larger.max_inflight_snapshot_charge == 65 * 16 * 1024 * 1024
    larger.close()


@pytest.mark.parametrize("value", [0, -1, True, 1.2])
def test_invalid_total_budget_rejected(tmp_path, value):
    with pytest.raises(ValueError, match="inflight snapshot budget"):
        Recorder(tmp_path, inflight_snapshot_budget=value)
