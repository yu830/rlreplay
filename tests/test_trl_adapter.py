"""Semantic mapping and frozen-input boundaries without optional TRL imports."""

from dataclasses import replace

import pytest

from rlreplay.adapters.trl import (
    SUPPORTED_REVISION,
    from_frozen_batch,
    profile_from_config,
)
from rlreplay.artifacts import decode, encode
from rlreplay.contracts.grpo import check
from rlreplay.core import Profile
from rlreplay.replay import diff, semantic_replay


@pytest.fixture
def configuration():
    return {
        "world_size": 1,
        "reward_count": 1,
        "reward_weights": [1.0],
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
        "num_generations": 3,
    }


@pytest.fixture
def frozen(configuration):
    return {
        "sample_ids": ["s0", "s1", "s2"],
        "group_ids": ["g", "g", "g"],
        "rewards": [0.0, 1.0, 2.0],
        "scalar_advantages": [-0.9999000099990001, 0.0, 0.9999000099990001],
        "completion_masks": [[1, 0], [1, 1], [1, 0]],
        "membership": (("s0", "g"), ("s1", "g"), ("s2", "g")),
        "configuration": configuration,
        "provenance": {
            "framework": "trl",
            "framework_revision": SUPPORTED_REVISION,
            "algorithm_sha256": "a" * 64,
            "phase": "post_generate_score.scalar_broadcast_mask",
            "receipt_origin": "synthetic-unit-test",
            "device": "cpu",
            "dtype": "float32",
        },
    }


def test_trl_uses_shared_contract_and_artifact(frozen):
    evidence = from_frozen_batch(**frozen)
    assert check(evidence).verdict == "PASS"
    assert evidence.profile.implementation == "trl_grpo"
    assert evidence.profile.epsilon == 1e-4
    assert decode(encode(evidence)) == evidence
    assert semantic_replay(evidence)["finding"]["verdict"] == "PASS"


def test_legal_epsilon_difference_is_explicit(frozen):
    trl = from_frozen_batch(**frozen)
    verl = replace(
        trl,
        profile=Profile(),
        samples=(
            replace(trl.samples[0], advantage=(-0.9999990000010001, 0.0)),
            replace(trl.samples[1], advantage=(0.0, 0.0)),
            replace(trl.samples[2], advantage=(0.9999990000010001, 0.0)),
        ),
        provenance={
            "framework": "verl",
            "framework_revision": "synthetic-unit-test-no-upstream-revision",
            "algorithm_sha256": "synthetic-no-source-digest",
            "phase": "synthetic_epsilon_comparison",
            "receipt_origin": "synthetic-unit-test-hand-derived-values",
            "device": "cpu",
            "dtype": "float32",
        },
    )
    assert check(trl).verdict == check(verl).verdict == "PASS"
    assert diff(trl, verl)["same_profile"] is False
    assert check(replace(trl, profile=Profile())).verdict == "SUSPECTED_MISMATCH"


@pytest.mark.parametrize(
    "key,value",
    [
        ("scale_rewards", "batch"),
        ("scale_rewards", "none"),
        ("world_size", 2),
        ("num_generations", 1),
        ("multi_objective_aggregation", "normalize_then_sum"),
        ("mask_truncated_completions", True),
        ("tools", True),
        ("use_vllm", True),
    ],
)
def test_unsupported_configuration(configuration, key, value):
    configuration[key] = value
    with pytest.raises(ValueError):
        profile_from_config(configuration)


def test_missing_reward_and_incomplete_membership(frozen):
    with pytest.raises(ValueError, match="finite reward"):
        from_frozen_batch(**{**frozen, "rewards": [0.0, None, 2.0]})
    missing = {**frozen}
    for field in (
        "sample_ids",
        "group_ids",
        "rewards",
        "scalar_advantages",
        "completion_masks",
    ):
        missing[field] = missing[field][:-1]
    evidence = from_frozen_batch(**missing)
    assert evidence.complete_groups is False
    assert check(evidence).verdict == "INCONCLUSIVE"
