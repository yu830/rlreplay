"""Pure numeric snapshots for application-owned loggers and metric collectors."""

import math
from collections.abc import Mapping

_RECORDER_COUNTERS = {
    "attempted": "rlreplay.capture.attempts",
    "reserved": "rlreplay.capture.reservations",
    "submitted": "rlreplay.capture.submissions",
    "COMPLETE": "rlreplay.capture.completions",
    "DROPPED": "rlreplay.capture.drops",
    "ERROR": "rlreplay.capture.errors",
    "PARTIAL": "rlreplay.capture.partials",
    "CANCELLED": "rlreplay.capture.cancellations",
    "event_details_evicted": "rlreplay.capture.event_details_evicted",
    "check_errors": "rlreplay.check.errors",
    "finding_PASS": "rlreplay.check.pass",
    "finding_CONTRACT_VIOLATION": "rlreplay.check.contract_violations",
    "finding_SUSPECTED_MISMATCH": "rlreplay.check.suspected_mismatches",
    "finding_INCONCLUSIVE": "rlreplay.check.inconclusive",
}
_OBSERVER_COUNTERS = {
    "calls": "rlreplay.observer.calls",
    "disabled_calls": "rlreplay.observer.disabled_calls",
    "metadata_samples": "rlreplay.observer.metadata_samples",
    "cadence_skipped_calls": "rlreplay.observer.cadence_skips",
    "admission_rejected_calls": "rlreplay.observer.admission_rejections",
    "available_samples_at_admitted_calls": "rlreplay.observer.available_samples",
    "snapshot_calls": "rlreplay.observer.snapshots",
    "captured_samples": "rlreplay.observer.captured_samples",
    "captured_groups": "rlreplay.observer.captured_groups",
    "submitted_calls": "rlreplay.observer.submissions",
    "NOT_RUN": "rlreplay.observer.not_run",
    "UNSUPPORTED": "rlreplay.observer.unsupported",
    "DROPPED": "rlreplay.observer.drops",
    "ERROR": "rlreplay.observer.errors",
}
_STATUS_GAUGES = {
    "closed": "rlreplay.capture.closed",
    "worker_alive": "rlreplay.capture.worker_alive",
    "receipt_channel_open": "rlreplay.capture.receipt_channel_open",
    "reserved": "rlreplay.capture.inflight.reserved",
    "pending": "rlreplay.capture.inflight.pending",
    "pending_snapshot_charge": "rlreplay.capture.inflight.snapshot_charge",
    "oldest_pending_seconds": "rlreplay.capture.oldest_pending.age",
}
_FLAGS = frozenset({"closed", "worker_alive", "receipt_channel_open"})

COUNTER_NAMES = (*_RECORDER_COUNTERS.values(), *_OBSERVER_COUNTERS.values())
GAUGE_NAMES = tuple(_STATUS_GAUGES.values())
ADDITIVE_GAUGE_NAMES = (
    "rlreplay.capture.inflight.reserved",
    "rlreplay.capture.inflight.pending",
    "rlreplay.capture.inflight.snapshot_charge",
)
METRIC_UNITS = {
    "rlreplay.capture.inflight.snapshot_charge": "By",
    "rlreplay.capture.oldest_pending.age": "s",
}


def _values(source, fields, *, flags=frozenset()):
    result = {}
    if source is None:
        return result
    for key, name in fields.items():
        if key not in source or source[key] is None:
            continue
        value = source[key]
        if key in flags:
            if type(value) is not bool:
                raise ValueError(f"{name} requires a boolean status flag")
            value = int(value)
        elif (
            type(value) not in (int, float)
            or value < 0
            or (type(value) is float and not math.isfinite(value))
        ):
            raise ValueError(f"{name} requires a finite nonnegative number")
        result[name] = value
    return result


def metrics_snapshot(
    recorder_stats: Mapping[str, object],
    *,
    recorder_status: Mapping[str, object] | None = None,
    observer_stats: Mapping[str, object] | None = None,
) -> dict[str, dict[str, int | float]]:
    """Read fixed numeric fields from caller-owned snapshots without advancing IPC.

    Counters are cumulative for one source instance; gauges are current values.
    Missing/None fields are omitted. Invalid present values raise ValueError
    without including their content. No history, identifiers, or labels are kept.
    """
    counters = _values(recorder_stats, _RECORDER_COUNTERS)
    counters.update(_values(observer_stats, _OBSERVER_COUNTERS))
    return {
        "counters": counters,
        "gauges": _values(recorder_status, _STATUS_GAUGES, flags=_FLAGS),
    }
