# Contributing to RLReplay

RLReplay grows through reproducible training problems, explicit contracts and
reviewable evidence. Small bug fixes, synthetic failure cases, documentation and
framework compatibility work are welcome.

## Development checks

Use Python 3.11 or later and uv. The lockfile defines the development environment.
The optional numeric codec is included in the complete CPU test run.

```bash
uv sync --python 3.12 --locked --dev --extra binary
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync pytest -q
uv build --wheel
```

Run the synthetic healthy/faulty example from the checkout.

```bash
uv run --no-sync python examples/frozen_contract.py ./demo-output
```

The example validates its two expected results. The CI workflow also installs the
built wheel into a separate environment and checks its JSON workflow through the
CLI. The configured matrix is Windows with Python 3.11, 3.12 and 3.14, plus Ubuntu
24.04 with Python 3.12. Framework training integration requires the separately
qualified environment described in the compatibility guide. [Compatibility](docs/compatibility.md)

## Report a problem

Include versions, the declared profile, a minimal command and the expected versus
observed result. Prefer synthetic inputs. A finding needs enough evidence to
distinguish a contract violation, a suspected numerical mismatch and an
inconclusive check. State missing inputs or unsupported modes explicitly.

Review material before attaching it to a public issue or pull request. Sample
IDs, source metadata, absolute paths and numeric rewards can reveal private
information. Share only data you have permission to publish. Suspected
vulnerabilities use the private reporting process in the security policy. [Security policy](SECURITY.md)

## Change a contract or adapter

Describe the native boundary, version and resolved configuration. Preserve sample
identity through framework transformations and declare the roles of masks and
probability tensors. Keep the independent reference separate from the code being
checked. Include a healthy case, a fault case and relevant numerical or missing-data
controls. Document the supported subset and its source/runtime assumptions.

Changes to capture should exercise ownership, resource budgets, failure receipts
and the training function's original result or exception. Measure performance
against equal useful work and report coverage, drops and inconclusive outcomes
alongside latency. A test failure should lead to a correction or an explicit
limitation supported by evidence.

Keep changes focused, preserve compatible behavior and update the affected public
documentation. PR validation should identify the actual commands and results.
Local validation, native framework execution, remote CI and upstream acceptance
are separate evidence. The architecture and adapter guide describe current
extension points. [Architecture](docs/architecture.md) · [Adapters](docs/adapters.md)
