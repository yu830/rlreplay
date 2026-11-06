# Compatibility and evidence scope

Version strings alone can cover different development commits. Use the pinned
revision, recorded owner hashes and the supported configuration together.

## Framework boundaries

| Framework | Qualified revision | Implemented boundary | Replay and regression |
|---|---|---|---|
| verl | `31cfccb4dbad8f1f4aa41a34eb589afbe283e8fe` | CPU DataProto and driver outcome-GRPO, full-group sampling | Semantic replay, native `compute_advantage`, native pytest export |
| TRL | `a16af6502c4a1922045be533c94f66dd7c981374` | CPU single-process standard GRPOTrainer, runtime mixin and frozen importer | Semantic replay; native method replay/export require additional input contracts |
| NeMo RL | Source research only | Future receipt/ledger import | Runtime adapter qualification pending |

The verl vectorized driver at the qualified revision drops the no-std flag. This
known failing configuration remains a regression case. The scalar path and
independent candidate verification provide separate evidence. The qualified
framework examples exercise CPU boundaries. GPU transport and larger training
systems require their own environment qualification.

TRL uses sample standard deviation and denominator epsilon `1e-4`; the qualified
verl profile uses `1e-6`. Each is checked against its declared profile. Standard
TRL runtime observation requires the narrow settings in the adapter guide.
Its experimental async GRPO helper uses different variance/epsilon semantics and
requires a separate contract.

## Upstream source watch

Official main snapshots checked on 2026-11-05 were TRL
`bcfb115a5fcc6f367a9a77c65617e747fff8c28d` and verl
`7cea5392165ac26d5eadf879701337c85e4a2b2f`. The TRL Trainer and verl legacy
driver/core files matched the qualified sources. TRL's statistics-owner file
changed even though the `nanstd` function body matched. These snapshots have
source comparison evidence only. The source gate requires separate qualification
before a changed owner can be accepted.

The comparisons use immutable official sources. [TRL Trainer](https://github.com/huggingface/trl/blob/bcfb115a5fcc6f367a9a77c65617e747fff8c28d/trl/trainer/grpo_trainer.py)
[TRL statistics owner](https://github.com/huggingface/trl/blob/bcfb115a5fcc6f367a9a77c65617e747fff8c28d/trl/trainer/utils.py)
[verl legacy driver](https://github.com/verl-project/verl/blob/7cea5392165ac26d5eadf879701337c85e4a2b2f/verl/trainer/ppo/ray_trainer.py)

## Python, storage and operation

The package declares Python 3.11 or later. The primary development and training
environments use Python 3.12.13. Platform results must identify their interpreter
and optional dependencies, with package-install checks separate from framework
execution. Local qualification records are maintained with each development run.

The `0.1.0a1` local candidate matrix completed 90 tests with one POSIX-permission skip on Windows
CPython 3.11.15, 3.12.13 and 3.14.7, and 91 tests on WSL CPython 3.12.13. The
3.11/3.14 installed-wheel runs used NumPy 2.4.6/2.5.4 respectively and safetensors
0.8.0. Both endpoints also ran the JSON example in isolated no-site mode using
only the standard library and RLReplay. Framework training was exercised on the
separately qualified 3.12 environment. Hosted CI covers Windows 3.11/3.12/3.14 and Linux 3.12. Python 3.13 awaits validation.

Core/JSON uses the standard library. The optional binary codec requires NumPy
and safetensors. JSON schema v1 remains readable. Windows directories use NTFS
ACLs, including paths accessed through WSL. POSIX-mode checks run on a native
Linux filesystem. Independent producer directories are supported within the
qualified runtime; cross-host filesystems, shared Windows/WSL locks and global
quota coordination need deployment-specific qualification.

Existing integration measurements cover local bounded workloads. Longer production runs,
multi-GPU/multi-node operation require separate qualification. Hosted CI evidence is linked from the validation page.
See [adapter integration](adapters.md) and [runtime contracts](production.md).
