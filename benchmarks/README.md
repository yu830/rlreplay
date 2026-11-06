# CPU capture benchmark

Run a small current-SDK measurement from the repository root:

```bash
uv run --locked --extra binary python benchmarks/capture_cpu.py --codec both
```

The default is one sequential run per codec: 32 offered captures, two groups of two responses, 128 positions, queue capacity 2 and a 20ms offered interval. One completed warmup precedes measurement. Evidence is generated using only the standard library and explicitly rounded binary32 advantages; no model, training framework or GPU is used. JSON alone needs no binary extra:

```bash
uv run --locked python benchmarks/capture_cpu.py --codec json
```

For a functional CI smoke run:

```bash
uv run --locked --extra binary python benchmarks/capture_cpu.py --codec both --captures 4 --groups 1 --width 4
```

`--output NEW_DIRECTORY` retains the summary and synthetic capsules. Otherwise the script uses a temporary directory. It does not download anything or install a baseline.

## What is measured

- Caller p50/p95/p99 time covers `Recorder.submit` on already constructed Evidence. Input generation and warmup are excluded.
- Every actual COMPLETE is read, compared with the complete synthetic input, and checked by the correctness oracle. Submission alone is not counted as completion.
- Observed completion latency runs from submit start until the application observes the receipt. It includes polling cadence and is not an exact storage-device latency.
- Completion rate uses actual COMPLETE divided by elapsed measurement/drain time, including the configured pacing. At a 20ms offered interval, approximately 50 captures/s is expected from the schedule; this is not maximum sustainable throughput.
- Submitted/COMPLETE/DROPPED/ERROR/PARTIAL counts, durable fraction, codec, shape, SDK version and relevant source hashes accompany the timings. Drops remain visible and cause a PARTIAL result rather than a successful full-capture run.

Percentiles use nearest rank over the observed samples. With 4 or 32 captures, p99 is a descriptive observed tail, not a qualified production percentile. Results depend on local scheduling, filesystem and dependencies. No external or earlier-release baseline is included, and the script makes no speedup claim.

## Recorded local example

[Alpha1 Windows result](results/cpu-capture-0.1.0a1-windows.json) is one local run of the command above on Python 3.12.13, NumPy 2.5.4 and safetensors 0.8.0. Both codecs completed and independently verified 32/32 captures with no drops. The file retains relative source names, versions, configuration, counts and percentiles, without account names or local absolute paths. It is an example for reproducing the method, not a cross-machine ranking or a performance guarantee.
