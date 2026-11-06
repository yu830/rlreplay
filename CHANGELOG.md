# Changelog

## 0.1.0a1 — Unreleased

Initial experimental candidate for LLM training correctness and failure replay.

- Versioned Evidence, Profile and Finding models for declared outcome-GRPO
  membership, mask and advantage contracts, with qualified numeric verdicts.
- Independent checks and semantic replay through the local CLI, evidence diffs,
  and native verl replay with source/runtime checks and regression export.
- Opt-in verl driver observation and TRL single-process CPU Trainer hooks for
  pinned supported configurations.
- Bounded asynchronous capture with owned snapshots, isolated encoding and
  optional checks, durable completion receipts, and per-directory writer leases.
- Default JSON capsules and an optional safetensors representation, including
  integrity checks and preservation of nonfinite values and numeric fault evidence.
- Fixed numeric telemetry snapshots for application-owned loggers and collectors.
- Synthetic examples, compatibility and contract documentation, and a CPU CI
  workflow covering tests, wheel installation and the synthetic CLI workflow.

Native replay and generated native regressions currently target verl. TRL uses
runtime capture, independent checks and semantic replay. Production-scale GPU
overhead, multi-GPU execution and wider framework or algorithm support require
their own validation. Local validation evidence and configured CI coverage are
documented separately from future release and remote CI results.
