# Documentation

RLReplay records declared training contracts and numeric observations, checks
them independently, and carries reproducible failures into replay and regression.

| What you want to do | Start here |
|---|---|
| Understand the long-term project | [Goal](../GOAL.md) |
| Run a healthy/faulty example | [Quick start](quickstart.md) · [中文上手](zh/quickstart.md) |
| Check a framework version or configuration | [Compatibility](compatibility.md) |
| Understand component responsibilities | [Architecture](architecture.md) |
| Inspect the mathematical assumptions | [Outcome-GRPO contract](contracts/grpo.md) |
| Capture a computation in training | [Adapters](adapters.md) |
| Set budgets and manage storage | [Operating capture](production.md) |
| Feed an existing logger or collector | [Telemetry](telemetry.md) |
| Reproduce a framework defect | [verl normalization case](case-studies/verl-normalization.md) |
| Reproduce the validation controls | [Validation](validation.md) |
| Measure capture cost and coverage | [Benchmarks](../benchmarks/README.md) |
| Contribute a fix or adapter | [Contributing](../CONTRIBUTING.md) |

The current release is experimental. The compatibility and validation pages
identify the scope supported by actual evidence.
