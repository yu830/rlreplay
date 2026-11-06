"""CPU outcome-GRPO observation at the verl driver advantage boundary."""

import hashlib
import inspect
import logging
import platform
from collections import Counter, deque
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

from rlreplay.core import Evidence, Profile, Sample

SUPPORTED_REVISION = "31cfccb4dbad8f1f4aa41a34eb589afbe283e8fe"
SAMPLE_ID_FIELD = "rlreplay_sample_id"
LOGGER = logging.getLogger(__name__)


def runtime_versions() -> dict[str, str]:
    return {
        "python_version": platform.python_version(),
        "torch_version": version("torch"),
        "numpy_version": version("numpy"),
    }


def algorithm_digest() -> str:
    from verl.trainer.ppo import core_algos

    return hashlib.sha256(Path(inspect.getfile(core_algos)).read_bytes()).hexdigest()


def driver_digest() -> str:
    from verl.trainer.ppo import ray_trainer

    return hashlib.sha256(Path(inspect.getfile(ray_trainer)).read_bytes()).hexdigest()


def groupwise_digest() -> str:
    from verl.utils import group_mean_std

    owner = inspect.getfile(inspect.unwrap(group_mean_std))
    return hashlib.sha256(Path(owner).read_bytes()).hexdigest()


def membership(data) -> tuple[tuple[str, str], ...]:
    """Call at the producer boundary after sample IDs are assigned, before reorder."""
    ids = data.non_tensor_batch[SAMPLE_ID_FIELD]
    groups = data.non_tensor_batch["uid"]
    pairs = tuple(zip(ids, groups, strict=True))
    if any(not isinstance(value, str) or not value for pair in pairs for value in pair):
        raise ValueError("this adapter requires nonempty string sample and group IDs")
    return pairs


@dataclass(frozen=True)
class Snapshot:
    rows: tuple[Sample, ...]
    membership: tuple[tuple[str, str], ...]
    reservation: str


class VerlObserver:
    """Optional observer that leaves training exceptions and outputs with the caller.

    The caller declares its resolved configuration. This adapter supports CPU
    outcome-only rewards, synchronous single-turn GRPO, and no reward shaping or
    rollout rejection. Sample IDs must survive the framework's own reorder.
    """

    def __init__(
        self,
        recorder,
        profile: Profile,
        provenance: dict[str, str],
        *,
        max_groups=2,
        max_elements=65536,
        sample_every=1,
        mode="sampled",
    ):
        if mode not in {"disabled", "metadata", "sampled"} or any(
            type(value) is not int or value < 1
            for value in (max_groups, max_elements, sample_every)
        ):
            raise ValueError("invalid observation configuration")
        self.recorder = recorder
        self.profile = profile
        self.provenance = dict(provenance)
        self.max_groups = max_groups
        self.max_elements = max_elements
        self.mode = mode
        self.sample_every = sample_every
        self.events = deque(maxlen=128)
        self.stats = Counter()
        self._runtime_recorded = False
        self.provenance.update(
            capture_mode=mode,
            sample_every=str(sample_every),
            max_groups=str(max_groups),
        )

    def _event(self, event):
        self.stats[event["state"]] += 1
        self.events.append(event)

    def coverage(self) -> dict:
        """Keep attempted calls, skipped calls, sampled rows, and durable receipts separate."""
        return {
            "mode": self.mode,
            "sample_every": self.sample_every,
            "max_groups": self.max_groups,
            "observer": dict(self.stats),
            "recorder": dict(self.recorder.stats) if self.recorder is not None else {},
        }

    def before(self, data, receipt) -> Snapshot | None:
        self.stats["calls"] += 1
        if self.mode == "disabled":
            self.stats["disabled_calls"] += 1
            return None
        if self.mode == "metadata":
            pairs = membership(data)
            self.stats["metadata_samples"] += len(pairs)
            self._event(
                {"state": "NOT_RUN", "reason": "metadata_only", "samples": len(pairs)}
            )
            return None
        if (self.stats["calls"] - 1) % self.sample_every:
            self.stats["cadence_skipped_calls"] += 1
            return None
        reservation = self.recorder.reserve()
        if reservation is None:
            self.stats["admission_rejected_calls"] += 1
            return None
        try:
            snapshot = self._snapshot(data, receipt, reservation)
        except BaseException:
            self.recorder.cancel(reservation, reason="snapshot_failed")
            raise
        if snapshot is None:
            self.recorder.cancel(
                reservation, reason="snapshot_unsupported_or_over_budget"
            )
        else:
            self.stats["snapshot_calls"] += 1
            self.stats["captured_samples"] += len(snapshot.rows)
            self.stats["captured_groups"] += len(
                {row.group_id for row in snapshot.rows}
            )
        return snapshot

    def _snapshot(self, data, receipt, reservation) -> Snapshot | None:
        if self.provenance.get("framework_revision") != SUPPORTED_REVISION:
            self._event({"state": "UNSUPPORTED", "reason": "framework_revision"})
            return None
        pairs = membership(data)
        self.stats["available_samples_at_admitted_calls"] += len(pairs)
        if not self._runtime_recorded:
            self.provenance.update(runtime_versions())
            self._runtime_recorded = True
        rewards = data.batch["token_level_rewards"]
        masks = data.batch["response_mask"]
        if (
            str(rewards.device) != "cpu"
            or str(rewards.dtype) != "torch.float32"
            or str(masks.device) != "cpu"
        ):
            self._event({"state": "UNSUPPORTED", "reason": "cpu_float32_required"})
            return None
        if len(rewards.shape) != 2 or len(masks.shape) != 2:
            self._event({"state": "UNSUPPORTED", "reason": "input_layout"})
            return None
        # Select stable group IDs, then capture every row belonging to those groups.
        groups = set(sorted({group for _, group in pairs})[: self.max_groups])
        selected = [index for index, (_, group) in enumerate(pairs) if group in groups]
        if len(selected) * max(rewards.shape[1], masks.shape[1]) > self.max_elements:
            self._event({"state": "DROPPED", "reason": "element_budget"})
            return None
        selected_rewards = rewards[selected]
        if bool(((selected_rewards != 0).sum(dim=-1) > 1).any()):
            self._event(
                {"state": "UNSUPPORTED", "reason": "outcome_only_rewards_required"}
            )
            return None
        reward_values = selected_rewards.sum(dim=-1).tolist()
        mask_values = masks[selected].tolist()
        rows = tuple(
            Sample(
                pairs[index][0],
                pairs[index][1],
                reward_values[position],
                tuple(mask_values[position]),
                (),
            )
            for position, index in enumerate(selected)
        )
        selected_members = tuple(
            (sample, group) for sample, group in receipt if group in groups
        )
        return Snapshot(rows, selected_members, reservation)

    def after(self, data, snapshot: Snapshot | None) -> str | None:
        if snapshot is None:
            return None
        advantages = data.batch["advantages"]
        if (
            str(advantages.device) != "cpu"
            or str(advantages.dtype) != "torch.float32"
            or len(advantages.shape) != 2
        ):
            self.recorder.cancel(snapshot.reservation, reason="output_layout")
            self._event({"state": "UNSUPPORTED", "reason": "output_layout"})
            return None
        if len(snapshot.rows) * advantages.shape[1] > self.max_elements:
            self.recorder.cancel(snapshot.reservation, reason="output_element_budget")
            self._event({"state": "DROPPED", "reason": "output_element_budget"})
            return None
        pairs = membership(data)
        if len({sample for sample, _ in pairs}) != len(pairs):
            raise ValueError("duplicate consumer sample IDs")
        positions = {
            sample: (index, group) for index, (sample, group) in enumerate(pairs)
        }
        rows = []
        for before in snapshot.rows:
            index, group = positions[before.sample_id]
            observed = tuple(advantages[index].tolist())
            rows.append(
                Sample(before.sample_id, group, before.reward, before.mask, observed)
            )
        evidence = Evidence(
            self.profile, tuple(rows), snapshot.membership, dict(self.provenance)
        )
        capture_id = self.recorder.submit(evidence, reservation=snapshot.reservation)
        if capture_id is not None:
            self.stats["submitted_calls"] += 1
        return capture_id

    def around(self, function, data, receipt, **kwargs):
        try:
            estimator = kwargs.get("adv_estimator")
            estimator = getattr(estimator, "value", estimator)
            normalize = kwargs.get("norm_adv_by_std_in_grpo", True)
            expected_profile = Profile(
                implementation=estimator, normalize_std=normalize
            )
            if (
                estimator not in {"grpo", "grpo_vectorized"}
                or type(normalize) is not bool
                or self.profile != expected_profile
            ):
                self.stats["calls"] += 1
                self._event({"state": "UNSUPPORTED", "reason": "profile_call_mismatch"})
                snapshot = None
            else:
                snapshot = self.before(data, receipt)
        except Exception as error:
            # This boundary owns observer errors. Training itself runs outside it.
            # Exception messages can contain captured data. Log only the error type.
            LOGGER.exception(
                "RLReplay observer failed before training (%s)",
                type(error).__name__,
                exc_info=False,
            )
            self._event(
                {"state": "ERROR", "reason": type(error).__name__, "phase": "before"}
            )
            snapshot = None
        try:
            result = function(data, **kwargs)
        except BaseException:
            if snapshot is not None:
                self.recorder.cancel(snapshot.reservation, reason="training_exception")
            raise
        try:
            self.after(result, snapshot)
        except Exception as error:
            LOGGER.exception(
                "RLReplay observer failed after training (%s)",
                type(error).__name__,
                exc_info=False,
            )
            if snapshot is not None:
                self.recorder.cancel(snapshot.reservation, reason="capture_failed")
            self._event(
                {"state": "ERROR", "reason": type(error).__name__, "phase": "after"}
            )
        return result
