"""Frozen-batch import for a pinned, single-process TRL GRPO boundary.

The native scalar advantage is represented as a derived token view by broadcasting
it over the completion mask. The caller retains the original scalar observation.
"""

import math
from collections import Counter

from rlreplay.core import Evidence, Profile, Sample

SUPPORTED_REVISION = "a16af6502c4a1922045be533c94f66dd7c981374"


def profile_from_config(configuration: dict) -> Profile:
    """Map actual resolved Trainer settings to the supported semantic subset."""
    supported = {
        "world_size": 1,
        "reward_count": 1,
        "scale_rewards": "group",
        "multi_objective_aggregation": "sum_then_normalize",
        "mask_truncated_completions": False,
        "use_vllm": False,
        "tools": False,
        "environment": False,
        "multimodal": False,
        "num_iterations": 1,
        "steps_per_generation": 1,
        "gradient_accumulation_steps": 1,
    }
    for key, expected in supported.items():
        if configuration.get(key) != expected:
            raise ValueError(f"unsupported TRL configuration: {key}")
    if configuration.get("reward_weights") != [1.0]:
        raise ValueError("unsupported TRL reward weights")
    generations = configuration.get("num_generations")
    if type(generations) is not int or generations < 2:
        raise ValueError("TRL proof requires at least two generations per group")
    return Profile(implementation="trl_grpo", epsilon=1e-4)


def from_frozen_batch(
    *,
    sample_ids,
    group_ids,
    rewards,
    scalar_advantages,
    completion_masks,
    membership,
    configuration: dict,
    provenance: dict[str, str],
) -> Evidence:
    """Import exact finite observations, preserving identity and completeness gaps."""
    profile = profile_from_config(configuration)
    if (
        provenance.get("framework") != "trl"
        or provenance.get("framework_revision") != SUPPORTED_REVISION
    ):
        raise ValueError("unsupported TRL source revision")
    if provenance.get("device") != "cpu" or provenance.get("dtype") != "float32":
        raise ValueError("TRL proof requires captured CPU float32 data")
    samples = []
    for sample_id, group_id, reward, advantage, mask in zip(
        sample_ids, group_ids, rewards, scalar_advantages, completion_masks, strict=True
    ):
        if any(
            not isinstance(value, str) or not value for value in (sample_id, group_id)
        ):
            raise ValueError("sample and group IDs must be nonempty strings")
        for value in (reward, advantage):
            if (
                type(value) not in (int, float)
                or abs(value) > 3.4028234663852886e38
                or not math.isfinite(value)
            ):
                raise ValueError(
                    "TRL proof requires finite reward and advantage observations"
                )
        if not mask or any(
            type(value) not in (int, float, bool) or value not in (0, 1)
            for value in mask
        ):
            raise ValueError("TRL proof requires a binary completion mask")
        samples.append(
            Sample(
                sample_id,
                group_id,
                reward,
                tuple(mask),
                tuple(advantage * value for value in mask),
            )
        )
    producer = tuple(tuple(pair) for pair in membership)
    if any(
        len(pair) != 2 or any(not isinstance(value, str) or not value for value in pair)
        for pair in producer
    ):
        raise ValueError("invalid producer membership")
    counts = Counter(row.group_id for row in samples)
    expected_count = configuration["num_generations"]
    complete = (
        bool(samples)
        and all(count == expected_count for count in counts.values())
        and {row.sample_id for row in samples}
        == {sample_id for sample_id, _ in producer}
    )
    return Evidence(
        profile, tuple(samples), producer, dict(provenance), complete_groups=complete
    )
