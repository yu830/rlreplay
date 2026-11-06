# Quick start

This walkthrough creates two synthetic JSON capsules, then checks and replays
their captured arithmetic. Run it from the source checkout containing `pyproject.toml`.
The two outputs should be PASS for the healthy case and SUSPECTED_MISMATCH for the
faulty case. Framework training integration has additional requirements described
below.

## Choose the Python environment

For local development, install the locked development and optional codec dependencies.

```powershell
uv sync --locked --dev --extra binary
$python = (Resolve-Path .venv/Scripts/python.exe).Path
$example = (Resolve-Path examples/frozen_contract.py).Path
```

For an already supplied local wheel, use a new environment in a separate directory.
Set `$wheel` to its full path and `$example` to the supplied `frozen_contract.py`
file before running these commands. The default JSON workflow needs only the
RLReplay distribution.

```powershell
uv venv --python 3.12 .venv
$python = (Resolve-Path .venv/Scripts/python.exe).Path
uv pip install --python $python --no-deps $wheel
```

Use an empty working directory for the wheel example so its `.venv` is dedicated
to this walkthrough. Keep the `$python` value for all following commands. On Linux,
the corresponding environment interpreter is `.venv/bin/python`.

## Generate and inspect the examples

These PowerShell commands select a fresh output directory and resolve the actual
generated capsule paths.

```powershell
$demo = Join-Path ([IO.Path]::GetTempPath()) ("rlreplay-demo-" + [guid]::NewGuid())
& $python $example $demo
$healthy = (Get-ChildItem -LiteralPath (Join-Path $demo 'healthy') -Filter '*.json').FullName
$faulty = (Get-ChildItem -LiteralPath (Join-Path $demo 'faulty') -Filter '*.json').FullName
& $python -m rlreplay inspect $healthy
& $python -m rlreplay check $healthy
& $python -m rlreplay check $faulty
```

The generator prints both paths and verifies their expected findings. `inspect`
prints the declared profile, provenance and sample count. The healthy check exits
0; the faulty check exits 2. PowerShell exposes the most recent command's exit
code as `$LASTEXITCODE`.

## Replay and compare

```powershell
& $python -m rlreplay replay $healthy
& $python -m rlreplay replay $faulty
& $python -m rlreplay diff $healthy $faulty
```

Semantic replay recomputes the independent reference and reports whether the
captured observations satisfy it. The two replay commands exit 0 and 2. `diff`
reports `s0` as changed and exits 0 for a completed comparison.

| Exit code | Meaning |
|---|---|
| 0 | PASS, or successful inspect/diff/export operation |
| 1 | CONTRACT_VIOLATION |
| 2 | SUSPECTED_MISMATCH |
| 3 | INCONCLUSIVE, including missing or unsupported evidence |
| 4 | Input, source or environment error |

The generated examples are synthetic and use explicit synthetic provenance.
Native verl replay requires a real verl driver capsule, the qualified training
environment and matching recorded source/runtime fingerprints. With those inputs,
use `replay <capsule> --runner verl`. A candidate fix can be selected explicitly
with `--allow-revision-change` and its actual source is recorded in the result.

`export-test <verl-capsule> <new-directory>` creates a native candidate regression
and fixture. Review the generated test, then run it with pytest in the compatible
verl environment. Standard TRL currently supports runtime capture and semantic
replay; its native-method regression input contract remains pending.

## Connect a training job

Use the pinned versions and supported configurations in the compatibility guide.
The verl observer wraps its driver advantage call. The TRL mixin observes the
native reward and advantage hooks. Keep capture output in an appropriately
restricted directory, close the Recorder, and inspect persisted/check/drop
receipts. Sampling coverage and checked coverage describe different stages.

Runnable CPU examples are supplied for the qualified environments.

```bash
python examples/verl_observer.py demo-verl-healthy
python examples/trl_capture.py demo-trl-healthy
```

The verl example uses a small preconstructed DataProto. The TRL example builds a
tiny model and tokenizer locally and executes one training step. Their compatible
framework dependencies must be installed in the chosen training environment.

## Read an optional binary capsule

The numeric codec is available through the `binary` extra. Keep that extra in
the uv command when reading a `.rlrp` capsule from a source checkout.

```bash
uv sync --locked --extra binary
uv run --locked --extra binary python -m rlreplay inspect /path/to/capsule.rlrp
```

Select `codec="safetensors"` on the Recorder or artifact writer to produce this
format. JSON remains available through the standard-library runtime.

[Adapter guide](adapters.md) · [Compatibility](compatibility.md) ·
[Runtime operation](production.md) · [Telemetry](telemetry.md)
