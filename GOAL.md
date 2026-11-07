# RLReplay Goal

## Mission

Build a practical debugging and regression-testing tool for LLM training.
Enable engineers to detect silent violations of
declared training contracts, diagnose them through independent validation and
linked evidence, replay failing computations under explicit reproducibility
conditions, and turn verified defects into automated regression tests.

## Long-term objectives

### Correctness and diagnosis

Make training assumptions explicit and check them against observed execution.
Develop independent oracles for meaningful contracts across data identities,
masks, probabilities, policy provenance, rewards, advantages, loss computation
and optimizer state. Preserve uncertainty when evidence or numerical
qualification is insufficient. Connect findings to the computation and source
versions needed to investigate them.

### Evidence, replay and regression

Capture sufficient, bounded evidence to reproduce a failure at a well-defined
computation boundary. Record the configuration, implementation and runtime
assumptions required by each replay. Support comparison of original and candidate
implementations and convert verified defects into portable regression cases.
Expand replay boundaries when real incidents establish the required inputs and
acceptable reproducibility guarantees.

### Production integration

Integrate with existing training systems through low-intrusion observation points.
Measure and control capture overhead, memory, storage and coverage. Isolate
diagnostic failures from training, minimize captured data, and document explicit
failure and retention policies. Qualify longer-running and distributed workloads
against their actual deployment requirements.

### An extensible ecosystem

Share evidence models, versioned semantic contracts, oracles and replay mechanisms
across training frameworks and algorithms. Keep framework-specific translation
in thin adapters and connect results to established telemetry and testing tools.
Reuse upstream capabilities and mature open-source components where they satisfy
the required contracts.

## How the project expands

The starting implementation covers outcome-GRPO in pinned verl and TRL
configurations. The goal spans LLM training, and the project will expand along
three evidence-driven directions.

- **Contract coverage** from current group/reward/mask/advantage checks to further
  loss, probability, policy and state-consistency contracts.
- **Execution coverage** from qualified local computation boundaries to larger,
  longer-running and multi-worker deployments with documented replay conditions.
- **Ecosystem coverage** through additional framework and algorithm integrations
  driven by concrete failure cases and reusable upstream evidence.

Each extension should bring a real use case, an explicit semantic contract,
an independent verification path, a minimal reproducible example, and measured
operating costs. Record its supported configurations and limitations before
promoting it into the compatibility matrix.

## What progress should demonstrate

- Relevant defects become detectable and reproducible through normal user-facing
  entry points.
- Healthy, faulty and inconclusive cases receive appropriately qualified results.
- A captured failure can support diagnosis, candidate verification and regression
  where the declared replay boundary permits it.
- Integration cost, capture coverage and resource use are measured together.
- Real users can identify and fix training failures with less investigation and
  reproduction work than their existing workflow requires.

This document defines the project's long-term direction. The
[compatibility matrix](docs/compatibility.md) identifies verified configurations,
and the [validation record](docs/validation.md) describes their evidence. Future
scope changes should state the new contracts and acceptance criteria while
preserving the correctness, diagnosis, replay and regression mission.
