# Numeric telemetry snapshots

`rlreplay.telemetry.metrics_snapshot` converts existing runtime statistics into a fixed set of numeric metrics. It has no optional dependencies, keeps no history, creates no instruments or threads, and performs no network I/O.

```python
from rlreplay.telemetry import metrics_snapshot

# Run on the application's normal receipt-processing path.
cached_status = recorder.status()
cached_metrics = metrics_snapshot(
    dict(recorder.stats),
    recorder_status=cached_status,
    observer_stats=dict(observer.stats),  # Optional; omit for other integrations.
)
existing_logger.info("RLReplay metrics: %s", cached_metrics)
```

`Recorder.status()` itself processes incoming receipts. The conversion function only reads the supplied mappings; it never calls `status`, `poll`, `close`, or `coverage`, and never drains events. The application remains responsible for receipt processing, cache freshness and any synchronization with its existing collector.

## Counter and gauge semantics

The return value has two mappings: `counters` for cumulative, monotonic source counts during one Recorder/Observer lifetime, and `gauges` for current values. Missing or `None` inputs produce no observation. An explicit zero remains zero. Invalid present values raise `ValueError` without including their content; count values must be finite nonnegative Python numbers, while status flags must be booleans.

The function does not track previous snapshots or calculate deltas. A newly created Recorder starts a new source lifetime. A collector spanning successive Recorders must either aggregate their lifetime deltas in application-owned state or begin a new collection lifetime; continuing an old monotonic stream with a smaller source total is not a valid conversion. Source identity stays in application state and need not become a metric label. Repeatedly adding a cumulative snapshot to a synchronous counter would double count.

`COUNTER_NAMES` and `GAUGE_NAMES` expose the entire fixed vocabulary for one-time instrument creation. Unknown fields are ignored. No metric name or label is derived from a capture/sample/run ID, path, exception text, group, or dynamic reason.

Recorder counters use the following source fields:

| Source field | Metric |
|---|---|
| `attempted`, `reserved`, `submitted` | `rlreplay.capture.attempts`, `.reservations`, `.submissions` |
| `COMPLETE`, `DROPPED`, `ERROR`, `PARTIAL`, `CANCELLED` | `rlreplay.capture.completions`, `.drops`, `.errors`, `.partials`, `.cancellations` |
| `event_details_evicted` | `rlreplay.capture.event_details_evicted` |
| `check_errors` | `rlreplay.check.errors` |
| `finding_PASS`, `finding_CONTRACT_VIOLATION`, `finding_SUSPECTED_MISMATCH`, `finding_INCONCLUSIVE` | `rlreplay.check.pass`, `.contract_violations`, `.suspected_mismatches`, `.inconclusive` |

`attempts` counts reservation attempts, including admission rejection; it is distinct from Observer calls. `completions` counts processed publication receipts. Source event categories can overlap: an initialization error can coexist with partial pending captures, and a persisted capture can have a check error. They are not an exhaustive partition suitable for blindly deriving a success rate.

Optional Observer counters are:

| Source field | `rlreplay.observer.` suffix |
|---|---|
| `calls`, `disabled_calls`, `metadata_samples` | `calls`, `disabled_calls`, `metadata_samples` |
| `cadence_skipped_calls`, `admission_rejected_calls` | `cadence_skips`, `admission_rejections` |
| `available_samples_at_admitted_calls` | `available_samples` |
| `snapshot_calls`, `captured_samples`, `captured_groups`, `submitted_calls` | `snapshots`, `captured_samples`, `captured_groups`, `submissions` |
| `NOT_RUN`, `UNSUPPORTED`, `DROPPED`, `ERROR` | `not_run`, `unsupported`, `drops`, `errors` |

Sample/group counters count instances across calls, including repeated identities. Snapshot/submission coverage does not establish publication or correctness-check coverage. This converter does not infer missing denominators or emit a fabricated coverage ratio.

Current gauges come from a caller-supplied status snapshot:

| Source field | Metric | Unit |
|---|---|---|
| `reserved` | `rlreplay.capture.inflight.reserved` | `1` |
| `pending` | `rlreplay.capture.inflight.pending` | `1` |
| `pending_snapshot_charge` | `rlreplay.capture.inflight.snapshot_charge` | `By` |
| `oldest_pending_seconds` | `rlreplay.capture.oldest_pending.age` | `s` |
| `closed`, `worker_alive`, `receipt_channel_open` | `rlreplay.capture.closed`, `.worker_alive`, `.receipt_channel_open` | `1` |

Flags are emitted as 0/1. Worker liveness alone is not a health assertion. Snapshot charge is conservative input accounting, not encoded file size or process RSS. `METRIC_UNITS` contains the non-default units; all other names use `1`.

## Local OpenTelemetry example

The [OTel Metrics API](https://opentelemetry.io/docs/specs/otel/metrics/api/#asynchronous-counter) defines asynchronous counters as absolute cumulative observations. Current additive quantities such as pending work map to ObservableUpDownCounter; liveness and oldest age map to ObservableGauge. The [Python InMemoryMetricReader](https://opentelemetry-python.readthedocs.io/en/stable/sdk/metrics.export.html#opentelemetry.sdk.metrics.export.InMemoryMetricReader) supports local collection without an external exporter.

The application owns the optional OTel installation and provider. This example uses no global provider registration or periodic reader:

```python
from opentelemetry.metrics import Observation
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.resources import Resource
from rlreplay.telemetry import (
    ADDITIVE_GAUGE_NAMES,
    COUNTER_NAMES,
    GAUGE_NAMES,
    METRIC_UNITS,
)


# The normal application loop replaces cached_metrics with a complete snapshot.
def callback(group, name):
    def observe(_options):
        value = cached_metrics[group].get(name)
        return [] if value is None else [Observation(value)]

    return observe


reader = InMemoryMetricReader()
provider = MeterProvider(
    metric_readers=[reader],
    resource=Resource({"service.name": "rlreplay-local-example"}),
    shutdown_on_exit=False,
)
meter = provider.get_meter("rlreplay")
for name in COUNTER_NAMES:
    meter.create_observable_counter(
        name,
        callbacks=[callback("counters", name)],
        unit=METRIC_UNITS.get(name, "1"),
    )
for name in GAUGE_NAMES:
    create = (
        meter.create_observable_up_down_counter
        if name in ADDITIVE_GAUGE_NAMES
        else meter.create_observable_gauge
    )
    create(name, callbacks=[callback("gauges", name)], unit=METRIC_UNITS.get(name, "1"))

collected = reader.get_metrics_data()
provider.shutdown()
```

Callbacks read the cached mapping and do not advance the runtime. Missing values return no observations, and the example adds no attributes. An application can connect its existing collector/exporter explicitly; RLReplay does not enable one automatically.
