"""Opt-in hooks for the qualified single-process CPU GRPOTrainer boundary."""

import hashlib
import inspect
import logging
import platform
from collections import Counter, deque
from importlib.metadata import version
from pathlib import Path

from rlreplay.adapters.trl import (
    SUPPORTED_REVISION,
    from_frozen_batch,
    profile_from_config,
)

LOGGER = logging.getLogger(__name__)


def source_provenance(trainer):
    """Bind the installed native owners to the qualified source, before training."""
    from trl import GRPOTrainer
    from trl.trainer.utils import nanstd

    native = super(GRPOCaptureMixin, trainer)
    for name in ("_calculate_rewards", "_generate_and_score_completions"):
        if getattr(native, name).__func__ is not getattr(GRPOTrainer, name):
            raise ValueError("TRL capture requires the qualified native hook dispatch")
    owners = (
        (
            GRPOTrainer,
            "28f815d86f6416842debd444dcbd815b4373fe86ad2917e180b4d1780a220605",
        ),
        (nanstd, "ad364fb106b4a31331267ebef0d58fb3ad0bfbfd04accc09227212d29b6f8ba5"),
    )
    digests = [
        hashlib.sha256(
            Path(inspect.getfile(inspect.unwrap(owner))).read_bytes()
        ).hexdigest()
        for owner, _expected in owners
    ]
    if digests != [expected for _owner, expected in owners]:
        raise ValueError("installed TRL owners differ from the qualified source")
    return {
        "framework": "trl",
        "framework_revision": SUPPORTED_REVISION,
        "algorithm_sha256": digests[0],
        "entrypoint_sha256": digests[0],
        "groupwise_sha256": digests[1],
        "entrypoint": "trl.trainer.grpo_trainer.GRPOTrainer._generate_and_score_completions",
        "phase": "post_generate_score.scalar_broadcast_mask",
        "receipt_origin": "trainer_input_group_ids_and_native_reward_output",
        "python_version": platform.python_version(),
        "torch_version": version("torch"),
        "numpy_version": version("numpy"),
        "device": "cpu",
        "dtype": "float32",
        "capture_mode": "sampled",
        "world_size": "1",
        "rank": "0",
    }


def resolved_configuration(trainer):
    """Read the actual Trainer settings used by the supported native path."""
    if trainer.rollout_func is not None or trainer.use_transformers_continuous_batching:
        raise ValueError("custom rollout and continuous batching require qualification")
    return {
        "world_size": trainer.accelerator.num_processes,
        "reward_count": len(trainer.reward_funcs),
        "reward_weights": trainer.reward_weights.tolist(),
        "scale_rewards": trainer.scale_rewards,
        "multi_objective_aggregation": trainer.multi_objective_aggregation,
        "mask_truncated_completions": trainer.mask_truncated_completions,
        "use_vllm": trainer.use_vllm,
        "tools": bool(trainer.tools),
        "environment": trainer.environment_factories is not None,
        "multimodal": trainer._is_vlm,
        "num_iterations": trainer.args.num_iterations,
        "steps_per_generation": trainer.args.steps_per_generation,
        "gradient_accumulation_steps": trainer.args.gradient_accumulation_steps,
        "num_generations": trainer.num_generations,
    }


class GRPOCaptureMixin:
    """Place before GRPOTrainer in the MRO and close the caller-owned Recorder.

    Inputs carry an opaque group ID from the dataset through RepeatSampler. The
    two hooks call their native super methods exactly once. Runtime capture errors
    are reported separately, and the native result/exception stays with training.
    """

    def __init__(
        self,
        *args,
        capture_recorder,
        capture_run_id,
        capture_group_key="rlreplay_group_id",
        capture_sample_every=1,
        capture_max_elements=65536,
        **kwargs,
    ):
        if any(
            type(value) is not int or value < 1
            for value in (capture_sample_every, capture_max_elements)
        ) or any(
            not isinstance(value, str) or not value
            for value in (capture_run_id, capture_group_key)
        ):
            raise ValueError("invalid TRL capture configuration")
        provenance = source_provenance(self)
        super().__init__(*args, **kwargs)
        profile_from_config(resolved_configuration(self))
        self.rlreplay_recorder = capture_recorder
        self.rlreplay_stats = Counter()
        self.rlreplay_events = deque(maxlen=128)
        self._rlreplay_provenance = {**provenance, "run_id": capture_run_id}
        self._rlreplay_group_key = capture_group_key
        self._rlreplay_cadence = capture_sample_every
        self._rlreplay_limit = capture_max_elements
        self._rlreplay_pending = None
        self._rlreplay_active = False

    def _rlreplay_error(self, error, phase):
        self.rlreplay_stats["ERROR"] += 1
        self.rlreplay_events.append(
            {"state": "ERROR", "reason": type(error).__name__, "phase": phase}
        )
        self._rlreplay_cancel("capture_failed")

    def _rlreplay_cancel(self, reason):
        if self._rlreplay_pending is not None:
            self.rlreplay_recorder.cancel(
                self._rlreplay_pending["reservation"], reason=reason
            )
            self._rlreplay_pending = None

    def _rlreplay_begin(self, inputs):
        self.rlreplay_stats["calls"] += 1
        if (self.rlreplay_stats["calls"] - 1) % self._rlreplay_cadence:
            self.rlreplay_stats["cadence_skipped_calls"] += 1
            return
        reservation = self.rlreplay_recorder.reserve()
        if reservation is None:
            self.rlreplay_stats["admission_rejected_calls"] += 1
            return
        self._rlreplay_pending = {"reservation": reservation}
        if not self.model.training:
            raise ValueError("TRL evaluation capture requires separate qualification")
        configuration = resolved_configuration(self)
        profile_from_config(configuration)
        if not inputs or len(inputs) > self._rlreplay_limit:
            raise ValueError("TRL input exceeds the element budget")
        groups = tuple(row[self._rlreplay_group_key] for row in inputs)
        if any(not isinstance(group, str) or not group for group in groups):
            raise ValueError("opaque nonempty group IDs required")
        namespace = (
            f"{self._rlreplay_provenance['run_id']}/{self.rlreplay_stats['calls']}"
        )
        self._rlreplay_pending.update(
            configuration=configuration,
            input_groups=groups,
            sample_ids=tuple(f"{namespace}/{index}" for index in range(len(inputs))),
            group_ids=tuple(f"{namespace}/{group}" for group in groups),
        )

    def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
        output = super()._calculate_rewards(
            inputs, prompts, completions, completion_ids_list
        )
        pending = self._rlreplay_pending
        if pending is not None:
            try:
                if (
                    "rewards" in pending
                    or tuple(row[self._rlreplay_group_key] for row in inputs)
                    != pending["input_groups"]
                    or tuple(output.shape) != (len(pending["sample_ids"]), 1)
                    or str(output.device) != "cpu"
                    or str(output.dtype) != "torch.float32"
                ):
                    raise ValueError("unsupported native TRL reward boundary")
                pending["rewards"] = output.detach()[:, 0].tolist()
            except Exception as error:
                LOGGER.exception(
                    "TRL reward capture failed (%s)",
                    type(error).__name__,
                    exc_info=False,
                )
                self._rlreplay_error(error, "rewards")
        return output

    def _rlreplay_finish(self, output):
        pending = self._rlreplay_pending
        if pending is None:
            return
        advantages, masks = output["advantages"], output["completion_mask"]
        count = len(pending["sample_ids"])
        if (
            output.get("tool_mask") is not None
            or tuple(advantages.shape) != (count,)
            or str(advantages.device) != "cpu"
            or str(advantages.dtype) != "torch.float32"
            or len(masks.shape) != 2
            or masks.shape[0] != count
            or str(masks.device) != "cpu"
            or count * masks.shape[1] > self._rlreplay_limit
        ):
            raise ValueError("unsupported TRL output layout or element budget")
        evidence = from_frozen_batch(
            sample_ids=pending["sample_ids"],
            group_ids=pending["group_ids"],
            rewards=pending["rewards"],
            scalar_advantages=advantages.detach().tolist(),
            completion_masks=masks.detach().tolist(),
            membership=tuple(
                zip(pending["sample_ids"], pending["group_ids"], strict=True)
            ),
            configuration=pending["configuration"],
            provenance={
                **self._rlreplay_provenance,
                "step": str(self.state.global_step),
            },
        )
        self.rlreplay_stats["snapshot_calls"] += 1
        self.rlreplay_stats["captured_samples"] += count
        self.rlreplay_stats["captured_groups"] += len(set(pending["group_ids"]))
        capture_id = self.rlreplay_recorder.submit(
            evidence, reservation=pending["reservation"]
        )
        if capture_id is not None:
            self.rlreplay_stats["submitted_calls"] += 1

    def _generate_and_score_completions(self, inputs):
        if self._rlreplay_active:
            # A reward callback may reenter generation. Neither call then owns a
            # supported capture boundary, but both native calls still execute.
            self.rlreplay_stats["calls"] += 1
            self.rlreplay_stats["UNSUPPORTED"] += 1
            self.rlreplay_events.append(
                {"state": "UNSUPPORTED", "reason": "reentrant_generation"}
            )
            self._rlreplay_cancel("reentrant_generation")
            return super()._generate_and_score_completions(inputs)
        self._rlreplay_active = True
        try:
            try:
                self._rlreplay_begin(inputs)
            except Exception as error:
                LOGGER.exception(
                    "TRL capture failed before training (%s)",
                    type(error).__name__,
                    exc_info=False,
                )
                self._rlreplay_error(error, "before")
            try:
                result = super()._generate_and_score_completions(inputs)
            except BaseException:
                self._rlreplay_cancel("training_exception")
                raise
            try:
                self._rlreplay_finish(result)
            except Exception as error:
                LOGGER.exception(
                    "TRL capture failed after training (%s)",
                    type(error).__name__,
                    exc_info=False,
                )
                self._rlreplay_error(error, "after")
            finally:
                self._rlreplay_cancel("capture_finished")
            return result
        finally:
            self._rlreplay_active = False
