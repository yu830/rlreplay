# Adapters and contract extensions

An adapter translates observed framework inputs and outputs into `Evidence`.
The independent GRPO check consumes declared semantics, identities and numeric
observations. The current evidence schema covers outcome reward, response mask
and advantage contracts. Additional algorithm domains require their own evidence
and acceptance cases.

Runnable CPU examples are provided for [verl](../examples/verl_observer.py) and
[TRL](../examples/trl_capture.py). Run them in an environment with the qualified
framework revision and dependencies installed. The examples generate their small
synthetic inputs locally and write results to the supplied new directory.

## TRL runtime integration

Use the qualified standard GRPOTrainer source listed in the compatibility guide.
Keep the dataset's opaque `rlreplay_group_id` column through RepeatSampler with
`remove_unused_columns=False`. Give each dataset example a distinct group ID.
Each invocation adds its run/call namespace and response position to sample IDs.

```python
from trl import GRPOTrainer
from rlreplay.adapters.trl_runtime import GRPOCaptureMixin
from rlreplay.capture import Recorder


class ObservedTrainer(GRPOCaptureMixin, GRPOTrainer):
    pass


def train(model, args, dataset, tokenizer, reward, output_directory, run_id):
    recorder = Recorder(output_directory, check=True)
    try:
        trainer = ObservedTrainer(
            model=model,
            args=args,
            train_dataset=dataset,
            processing_class=tokenizer,
            reward_funcs=reward,
            capture_recorder=recorder,
            capture_run_id=run_id,
            capture_sample_every=10,
        )
        trainer.train()
        print(dict(trainer.rlreplay_stats))
        print(list(trainer.rlreplay_events))
    finally:
        receipts = recorder.close()
        print(receipts)
```

Call the entry script under `if __name__ == "__main__"`. The caller owns the
Recorder lifecycle. Constructor configuration/source failures raise before
training. Runtime observation failures produce bounded events containing phase
and exception type. Native return objects and exceptions propagate to the caller.
Reentrant generation cancels capture for that invocation and reports UNSUPPORTED
while the native calls continue.

The supported settings are CPU float32, one process, one finite reward function
with weight 1, `scale_rewards="group"`, `multi_objective_aggregation="sum_then_normalize"`,
at least two generations, one iteration, one step per generation and gradient
accumulation 1. The text-only single-turn path uses the standard backend, with
tools, environments, vLLM, custom rollout, continuous batching and truncation
filtering disabled. Evaluation calls are excluded from capture. Configuration is
read from the actual Trainer again at each sampled call.

The adapter reserves capacity before reading inputs. It captures the complete
admitted batch, bounded by `capture_max_elements=65536`, and skips tensor
conversion when the output exceeds this bound. `capture_sample_every` samples
generation calls. `rlreplay_stats` separates calls, skips, admission rejections,
snapshot counts and submissions. Recorder counters track subsequent persistence
and Oracle results. This integration operates in a single synchronous Trainer
call stream.

Rewards are copied from the real `_calculate_rewards` result. Scalar advantages
and masks come from the real `_generate_and_score_completions` result. Evidence
represents scalar advantage multiplied by completion mask. All-zero masks carry
no observation of the original scalar. Preserve any additional native values
required by a future replay boundary in an explicitly reviewed input format.

Source gates compare the installed Trainer and statistics owner files with the
qualified revision's hashes, and check that the mixin's two super calls dispatch
directly to the qualified native methods. Intermediate overrides require separate
qualification. A source-only comparison of a newer version is a separate step.
Callbacks and surrounding instrumentation remain part of the producer's trusted
implementation and should be included in local provenance manifests alongside
the recorded upstream owners.

## Frozen observations

`adapters.trl.from_frozen_batch()` remains available when the integration already
owns frozen observations. Supply explicit sample/group IDs, the producer
membership, actual configuration, source metadata, rewards, scalar advantages
and masks. It shares the existing artifact format and GRPO Oracle with verl.
Unsupported configurations raise. Incomplete groups preserve INCONCLUSIVE.

The standard TRL method combines generation, rewards and advantage computation.
The numeric capsule supports independent semantic replay. The separate explicit
[TRL native-method bundle](trl-native-replay.md) records the additional frozen
generation/reward providers and Trainer configuration for native replay/export.
It contains text and token IDs and is not created by the default numeric capture
path. The `--runner verl` command retains its framework guard.

## Adding a framework boundary

1. Locate the actual producer identity and consumer output boundaries in pinned
   upstream source. Copy values only after admission, and preserve complete groups.
2. Map configuration to explicit `Profile` fields, including epsilon, standard
   deviation correction, singleton policy and normalization. Record the applied
   implementation and runtime owners.
3. Build `Sample` objects from actual rewards, masks and output advantages, then
   `Evidence` with an independently observed membership receipt. Maintain identity
   through reordering. A copied consumer ID list alone cannot establish producer
   lineage.
4. Call the existing `check`, artifact reader/writer and semantic replay on both
   healthy and deliberately faulty observations. Include legal numerical
   differences, unsupported modes and incomplete inputs.
5. Exercise the real framework entry point, verify training outputs/exceptions,
   and qualify budgets/cleanup. Record the exact supported version and subset.

The runnable synthetic example exercises the Core/format/replay contract without
framework dependencies. Its source labels identify the synthetic producer.

```powershell
uv run --locked python examples/frozen_contract.py ./new-example-output
```

## Adding an algorithm contract

Start with one concrete computation and its independent oracle. Specify the
needed inputs, valid variants, numeric uncertainty and missing-evidence behavior.
Where the current outcome-GRPO fields suffice, explicit profile mapping can reuse
the existing checker. Other domains need a reviewed schema/contract change and
an explicit callable checker with `Finding` results. The worker and CLI currently
select the GRPO checker directly, so adding another checker includes changing
those real entry points and verifying old capsules. Entry-point discovery and a
generic plugin registry can be evaluated when an actual second algorithm needs them.

See [compatibility](compatibility.md), [runtime operation](production.md) and
[telemetry integration](telemetry.md) for the associated contracts.
