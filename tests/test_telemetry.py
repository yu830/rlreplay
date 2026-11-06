"""Fixed telemetry schema, missing data and receipt/coverage ownership."""

import copy
import json
import time
from collections import Counter

import pytest

from rlreplay.adapters.verl import VerlObserver
from rlreplay.capture import Recorder
from rlreplay.core import Profile
from rlreplay.telemetry import COUNTER_NAMES, GAUGE_NAMES, metrics_snapshot


def test_fixed_numeric_names_exclude_dynamic_and_sensitive_fields():
    secret = "sample-private-id /private/capsule/path token=fixture-only"
    stats = {"attempted": 7, "COMPLETE": 3, "finding_SUSPECTED_MISMATCH": 1, secret: 1}
    status = {
        "reserved": 1,
        "pending": 2,
        "worker_alive": True,
        "receipt_channel_open": False,
        "oldest_pending_seconds": 0.25,
        "pending_snapshot_charge": 4096,
        "path": secret,
    }
    observer = {
        "calls": 9,
        "captured_samples": 8,
        "captured_groups": 2,
        "ERROR": 1,
        "error": secret,
    }
    snapshot = metrics_snapshot(stats, recorder_status=status, observer_stats=observer)
    assert snapshot["counters"] == {
        "rlreplay.capture.attempts": 7,
        "rlreplay.capture.completions": 3,
        "rlreplay.check.suspected_mismatches": 1,
        "rlreplay.observer.calls": 9,
        "rlreplay.observer.captured_samples": 8,
        "rlreplay.observer.captured_groups": 2,
        "rlreplay.observer.errors": 1,
    }
    assert set(snapshot["counters"]) <= set(COUNTER_NAMES)
    assert set(snapshot["gauges"]) <= set(GAUGE_NAMES)
    assert not set(COUNTER_NAMES) & set(GAUGE_NAMES)
    assert secret not in json.dumps(snapshot)
    assert snapshot["gauges"]["rlreplay.capture.worker_alive"] == 1
    assert snapshot["gauges"]["rlreplay.capture.receipt_channel_open"] == 0
    assert snapshot["gauges"]["rlreplay.capture.inflight.snapshot_charge"] == 4096


def test_missing_is_absent_and_explicit_zero_is_preserved():
    stats = Counter()
    assert metrics_snapshot(stats) == {"counters": {}, "gauges": {}}
    assert not stats  # Counter.__missing__ must not invent zero observations.
    assert metrics_snapshot(
        {"COMPLETE": None}, recorder_status={"worker_alive": None}
    ) == {"counters": {}, "gauges": {}}
    assert metrics_snapshot({"COMPLETE": 0})["counters"] == {
        "rlreplay.capture.completions": 0
    }


def test_cumulative_counters_current_gauges_and_detached_snapshots():
    stats, current = Counter(attempted=5, reserved=4), {"reserved": 2, "pending": 2}
    first = metrics_snapshot(stats, recorder_status=current)
    stats.update(attempted=3, reserved=1)
    current.update(reserved=0, pending=1)
    second = metrics_snapshot(stats, recorder_status=current)
    assert first["counters"]["rlreplay.capture.attempts"] == 5
    assert second["counters"]["rlreplay.capture.attempts"] == 8
    assert second["counters"]["rlreplay.capture.reservations"] == 5
    assert second["gauges"]["rlreplay.capture.inflight.reserved"] == 0
    assert first["gauges"]["rlreplay.capture.inflight.pending"] == 2
    assert second["gauges"]["rlreplay.capture.inflight.pending"] == 1
    assert metrics_snapshot(Counter()) == {"counters": {}, "gauges": {}}
    large = 2**54 + 1
    assert (
        metrics_snapshot({"attempted": large})["counters"]["rlreplay.capture.attempts"]
        == large
    )


@pytest.mark.parametrize(
    "bad", [-1, float("nan"), float("inf"), True, "sensitive-value"]
)
def test_invalid_present_counter_is_not_hidden_as_zero(bad):
    with pytest.raises(ValueError) as caught:
        metrics_snapshot({"ERROR": bad})
    assert "sensitive-value" not in str(caught.value)


def test_invalid_status_values_are_explicit():
    with pytest.raises(ValueError, match="boolean status flag"):
        metrics_snapshot({}, recorder_status={"worker_alive": 1})
    with pytest.raises(ValueError, match="finite nonnegative"):
        metrics_snapshot({}, recorder_status={"oldest_pending_seconds": -0.1})


def test_conversion_does_not_advance_receipts_or_change_coverage(
    evidence, tmp_path, monkeypatch
):
    recorder = Recorder(tmp_path, check=True)
    try:
        capture_id = recorder.submit(evidence)
        deadline = time.monotonic() + 5
        while recorder._pending and time.monotonic() < deadline:
            recorder._collect()  # The application's explicit receipt-processing step.
            time.sleep(0.001)
        assert not recorder._pending
        cached_status = recorder.status()
        observer = VerlObserver(recorder, Profile(), {}, mode="disabled")
        observer.before(None, None)
        old_coverage = copy.deepcopy(observer.coverage())
        old_events = copy.deepcopy(list(recorder._events))

        def forbidden(*_args, **_kwargs):
            pytest.fail("pure metrics conversion called a live runtime method")

        with monkeypatch.context() as context:
            for name in ("poll", "close", "status", "_collect"):
                context.setattr(recorder, name, forbidden)
            context.setattr(observer, "coverage", forbidden)
            snapshot = metrics_snapshot(
                recorder.stats,
                recorder_status=cached_status,
                observer_stats=observer.stats,
            )
            assert snapshot["counters"]["rlreplay.capture.completions"] == 1
            assert snapshot["counters"]["rlreplay.observer.disabled_calls"] == 1
            assert list(recorder._events) == old_events
        assert observer.coverage() == old_coverage
        events = recorder.poll()
        assert (
            sum(e["state"] == "COMPLETE" and e["id"] == capture_id for e in events) == 1
        )
    finally:
        recorder.close()
