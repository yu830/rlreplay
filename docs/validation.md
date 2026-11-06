# Validation and known controls

The local prototype has unit checks, a fixed synthetic control set and a small reproducible CPU capture benchmark. These provide scoped evidence; fixed control ratios do not estimate production accuracy, false-positive rate or defect prevalence.

## Run the test suite

```bash
uv run --locked --extra binary pytest -q
uv run --locked --extra binary ruff check .
uv run --locked --extra binary ruff format --check .
```

The `0.1.0a1` candidate's 91-item suite includes three regression cases for unexpected CLI responses in the public validator. It was run from isolated installed packages on 2026-11-06:

| Environment | Recorded result |
|---|---|
| Windows, Python 3.11.15, binary extra | 90 passed, 1 platform skip |
| Windows, Python 3.12.13, binary extra | 90 passed, 1 skipped; the POSIX-permission case uses a Windows inherited-ACL boundary |
| Windows, Python 3.14.7, binary extra | 90 passed, 1 platform skip |
| WSL/Linux, Python 3.12.13, installed wheel | 91 passed |
| GitHub Actions, initial public commit | Four hosted jobs passed, including tests and installed-wheel checks |

The Windows 3.11 and 3.14 environments also executed the JSON example in isolated no-site mode. The full public control set below was rerun from the candidate with 76/76 expected verdicts and exit codes. The first public commit passed all four hosted CI jobs. Run the commands above to validate another checkout/environment. [Initial CI evidence](https://github.com/yu830/rlreplay/actions/runs/42161533969) · [Current CI status](https://github.com/yu830/rlreplay/actions/workflows/ci.yml)

## Fixed public 20-case control set

```bash
uv run --locked --extra binary python tools/validate_known_cases.py
```

The tool uses the installed package's normal `python -m rlreplay check` and default semantic `replay` entry points. Its inputs are entirely in [public-cases.json](../tests/fixtures/public-cases.json); no original capture directory, framework installation, model or private notes are required.

The fixed inventory is:

| Category | Cases | Expected behavior |
|---|---:|---|
| Healthy | 6 | PASS |
| Fault | 8 | CONTRACT_VIOLATION or SUSPECTED_MISMATCH, as declared per case |
| Inconclusive | 4 | INCONCLUSIVE for rounding, incomplete evidence, missing source or unsupported profile |
| Bad payload | 2 | ERROR for a JSON digest mismatch and a truncated binary capsule |

Eighteen evidence cases run in JSON and safetensors formats through both commands: 72 invocations. The JSON-only bad payload and binary-only bad payload each run through both commands: four more. The complete acceptance set therefore contains **20 unique controls and 76 check/replay commands**. Extra inspect commands are not included in this count.

Actual alpha1 local execution matched all **76/76** expected verdicts and exit codes, with both codecs available. One additional normal `inspect` invocation also succeeded. An environment error or malformed/unexpected CLI response remains a mismatch with the original exit/stdout/stderr retained; parsing is based on the actual response, not the expected verdict.

`--output NEW_DIRECTORY` preserves capsules, `commands.jsonl` and `summary.json` for review. Without it, files are temporary and the summary is printed. Paths inside command records are relative to that output directory.

### Fixture provenance and transformations

The fixture contains synthetic reward/mask/advantage values, formula anchors and controlled mutations. Framework-derived numbers retain their original framework revisions, owner/source digests, numerical profiles and runtime metadata. TRL's epsilon `1e-4` and verl/formula epsilon `1e-6` remain explicit. The known variance-floor values correspond to the mechanism fixed in [verl #6348](https://github.com/verl-project/verl/pull/6348); a control does not assert that current upstream still contains a defect.

Run, sample and group identities were replaced with deterministic public aliases. Numerical values/types, sample order, membership order/relationships, completeness flags and profiles were preserved, including duplicate/group faults and nonfinite tags. Collection origin identifies the published synthetic projection. Original source digests are retained provenance identifiers, not proof that the reader's runtime matches the producer or that an upstream accepts a defect claim.

Semantic replay re-evaluates declared numerical contracts. It does not reproduce a complete training job or validate model weights/reward meaning. Native framework replay has its own explicit environment and provenance requirements.

### Core-only subset

```bash
uv run --locked python tools/validate_known_cases.py --codec json
```

This selects 38/76 check/replay commands, reports `PARTIAL` and explicitly marks the 19 binary variants, including the truncated binary control, `NOT_RUN`. It cannot establish the complete 20-case/two-format acceptance result. In the default `both` mode, binary dependencies and valid binary construction are checked before malformed binary inputs; a missing optional dependency is an execution error, not evidence that a corrupt binary payload was correctly rejected.

## Reproduce a small CPU capture run

```bash
uv run --locked --extra binary python benchmarks/capture_cpu.py --codec both
```

See [benchmark method](../benchmarks/README.md) and the [recorded alpha1 local result](../benchmarks/results/cpu-capture-0.1.0a1-windows.json). The 32-capture run completed/read/checked 32/32 per codec; a separate 4-capture/tiny-shape functional smoke also completed both formats. These are descriptive current-SDK runs, with no external baseline or release-to-release percentage claim. Small-sample p99 and paced completion rate are not production guarantees.

The local publication-asset checks also cover exact fixture transformation, missing binary dependencies stopping before commands, JSON-subset reporting and binary32 synthetic inputs. They are separate from the recorded 88-test SDK suite. Remote CI, independent-user workloads and broad production numerical qualification remain NOT_RUN or require further evidence.
