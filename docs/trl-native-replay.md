# TRL native-method replay

RLReplay can capture and replay the standard TRL Trainer's
`_generate_and_score_completions` method at a frozen generation/reward boundary.
The installed native method recomputes padding, masks, reward aggregation and
advantages. Generation and reward functions are replaced with their captured
outputs; model generation, gradients and optimizer steps are outside this replay.

This uses a separate `rlreplay-trl-native/1` JSON bundle. It contains **prompt and
completion text, token IDs, rewards and scalar observations**, and is created only
by an explicit `capture_trl` call. It is not enabled by the default numeric
Recorder. The encoded bundle limit is 4 MiB. Keep it in a directory appropriate
for the sensitivity of the captured content.

## Qualified boundary

The qualified source is TRL `a16af6502c4a1922045be533c94f66dd7c981374`.
Use a real initialized CPU `GRPOTrainer` in training mode, with the same fast
tokenizer, configuration and runtime as the capture. The current subset requires:

- One process, non-conversational text prompts and distinct contiguous complete
  groups, each with at least two generations.
- One finite float32 reward function of weight 1, group scaling and
  `sum_then_normalize` aggregation.
- `beta=0`, one iteration, one step per generation, gradient accumulation 1,
  and gradient checkpointing disabled.
- No vLLM, custom rollout, continuous batching, tools, environments, images,
  truncation masking, sampling logprobs or additional generation fields.
- Native dispatch for `_generate`, `_calculate_rewards` and
  `_generate_and_score_completions`; these methods must not be overridden.

The caller owns the Trainer. Prefer a fresh Trainer for replay: temporary provider
replacements are restored even on failure, but the native method's logging and
metric updates are not rolled back. The qualified replay does not use model
weights: attempts to enter model/logprob computation raise an error.

## Run the synthetic example

From a source checkout with the qualified TRL dependencies installed:

```bash
python -m examples.trl_native_replay demo-trl-native
```

The example creates a tiny random model and tokenizer locally, captures a real
generation/reward call, replays on a new real Trainer, and exports a pytest
regression. It downloads no model. `result.json` records the bundle path, `PASS`,
the exact native-observation comparison and the exported test. The shared model
setup is also used by the ordinary [TRL training example](../examples/trl_capture.py).

For another CLI replay, supply the printed bundle path and a trusted local Trainer
factory explicitly:

```bash
python -m rlreplay replay PATH_TO_BUNDLE --runner trl --trainer-factory examples.trl_native_replay:make_trainer
python -m rlreplay export-test PATH_TO_BUNDLE trl-native-regression --runner trl --trainer-factory examples.trl_native_replay:make_trainer
python -m pytest -q trl-native-regression
```

The factory is a zero-argument `module:function` returning a matching real
Trainer. Its module must be importable in the selected interpreter. The reference
comes from your command, never from the bundle; importing a factory executes
ordinary local Python code. The example factory matches the synthetic example
only. Supply your own factory for other qualified inputs.

Ordinary replay requires matching source fingerprints. To test a candidate source
change, pass `--allow-revision-change`; runtime, tokenizer and configuration checks
still apply. Exported regressions explicitly enable candidate source changes and
assert the independent advantage contract. They retain the original bundle and
report the loaded source fingerprints. A deliberate semantic fault should fail
the assertion; import/setup failures are environment errors, not evidence of the
expected regression.

## Python API

```python
from rlreplay.trl_replay import capture_trl, read_trl, run_trl, write_trl

# trainer and fresh_trainer are caller-owned, matching real GRPOTrainer instances.
inputs = [
    {"prompt": "question", "rlreplay_group_id": "group-1"},
    {"prompt": "question", "rlreplay_group_id": "group-1"},
]
bundle = capture_trl(trainer, inputs)  # trainer must use num_generations=2 here
path = write_trl(bundle, "new-trl-bundles")
result = run_trl(read_trl(path), fresh_trainer)
```

Capture calls the native method once and may generate text. It is an explicit
debugging operation, not an asynchronous training hook. The inputs accept exactly
`prompt` and `rlreplay_group_id`. Errors propagate to the caller; a failed capture
does not produce a successful bundle.

`finding` checks scalar advantages, the token view, and the native layout derived
independently from frozen token lengths and padding configuration. Incorrect IDs,
masks or valid-token counts produce `CONTRACT_VIOLATION`; all-zero masks cannot
hide scalar discrepancies or pass as a valid native layout.
`native_observation_equal` compares the captured and replayed native outputs;
it is distinct from the Oracle verdict. A candidate may correctly fix a captured
fault and therefore differ from the original observation.

Existing numeric TRL Evidence v1 capsules continue to support semantic replay.
They do not contain the frozen providers needed here. A native bundle is not
accepted as an ordinary numeric capsule by `check`, `inspect` or default semantic
replay; use the explicit TRL runner and the Python reader above.

Qualification includes the real tiny CPU method, exact original-output replay,
and a separate full-source epsilon mutation that fails the same exported
regression before returning to the pristine source. That mutation is a test
fault, not a claim of an upstream TRL defect. Other configurations and production
overhead require their own evidence.
