"""Explicit local replay runners and regression export."""

import shutil
from dataclasses import replace
from pathlib import Path

from rlreplay.artifacts import read
from rlreplay.contracts.grpo import check, reference
from rlreplay.core import PROFILE_ID, Evidence


def semantic_replay(evidence: Evidence) -> dict:
    finding = check(evidence)
    result = {"runner": "semantic", "finding": finding.to_dict()}
    if finding.checked_samples:
        result["reference_advantages"] = reference(evidence)
    return result


def run_verl(evidence: Evidence, *, allow_revision_change=False) -> Evidence:
    """Replay the real driver entry point using the explicitly installed verl.

    Revision changes are opt-in for testing fixes. The returned provenance records
    the loaded implementation digests, keeping the original capsule untouched.
    """
    if evidence.provenance.get("framework") != "verl":
        raise ValueError("verl runner requires a verl capture")
    profile = evidence.profile
    if (
        profile.profile_id != PROFILE_ID
        or profile.epsilon != 1e-6
        or profile.ddof != 1
        or profile.singleton != "mean0_std1"
        or profile.implementation not in {"grpo", "grpo_vectorized"}
    ):
        raise ValueError("unsupported replay profile")
    if (
        evidence.provenance.get("device") != "cpu"
        or evidence.provenance.get("dtype") != "float32"
    ):
        raise ValueError("runner requires captured CPU float32 evidence")
    if (
        evidence.provenance.get("entrypoint")
        != "verl.trainer.ppo.ray_trainer.compute_advantage"
    ):
        raise ValueError("driver entrypoint provenance required")
    import numpy as np
    import torch
    from verl import DataProto
    from verl.trainer.ppo.ray_trainer import compute_advantage

    from rlreplay.adapters.verl import (
        SAMPLE_ID_FIELD,
        algorithm_digest,
        driver_digest,
        groupwise_digest,
        runtime_versions,
    )

    actual = {
        **runtime_versions(),
        "algorithm_sha256": algorithm_digest(),
        "entrypoint_sha256": driver_digest(),
        "groupwise_sha256": groupwise_digest(),
    }
    if not allow_revision_change and any(
        evidence.provenance.get(key) != value for key, value in actual.items()
    ):
        raise ValueError("installed implementation differs from capture")
    if not evidence.samples:
        raise ValueError("empty replay input")
    width = len(evidence.samples[0].mask)
    if not width or any(len(row.mask) != width for row in evidence.samples):
        raise ValueError("driver requires a rectangular response mask")
    rewards = torch.zeros((len(evidence.samples), width), dtype=torch.float32)
    rewards[:, 0] = torch.tensor(
        [row.reward for row in evidence.samples], dtype=torch.float32
    )
    data = DataProto.from_dict(
        tensors={
            "token_level_rewards": rewards,
            "response_mask": torch.tensor(
                [row.mask for row in evidence.samples], dtype=torch.float32
            ),
        },
        non_tensors={
            "uid": np.array([row.group_id for row in evidence.samples], dtype=object),
            SAMPLE_ID_FIELD: np.array(
                [row.sample_id for row in evidence.samples], dtype=object
            ),
        },
    )
    result = compute_advantage(
        data, profile.implementation, norm_adv_by_std_in_grpo=profile.normalize_std
    )
    samples = tuple(
        replace(row, advantage=tuple(values))
        for row, values in zip(
            evidence.samples, result.batch["advantages"].tolist(), strict=True
        )
    )
    provenance = {**evidence.provenance, **actual}
    if any(evidence.provenance.get(key) != value for key, value in actual.items()):
        provenance["framework_revision"] = "unverified-local-revision"
    return replace(evidence, samples=samples, provenance=provenance)


def diff(left: Evidence, right: Evidence) -> dict:
    """Compare by stable sample identity instead of current batch row order."""
    left_rows = {row.sample_id: row for row in left.samples}
    right_rows = {row.sample_id: row for row in right.samples}
    changed = [
        sample
        for sample in sorted(left_rows.keys() & right_rows.keys())
        if left_rows[sample] != right_rows[sample]
    ]
    return {
        "same_profile": left.profile == right.profile,
        "same_membership": set(left.membership) == set(right.membership),
        "changed_samples": changed,
        "only_left": sorted(left_rows.keys() - right_rows.keys()),
        "only_right": sorted(right_rows.keys() - left_rows.keys()),
        "left": check(left).to_dict(),
        "right": check(right).to_dict(),
    }


def export_test(capsule: str | Path, directory: str | Path) -> Path:
    """Export a fixed local verl driver test; the input never supplies executable code."""
    evidence = read(capsule)
    if (
        evidence.provenance.get("framework") != "verl"
        or evidence.provenance.get("entrypoint")
        != "verl.trainer.ppo.ray_trainer.compute_advantage"
    ):
        raise ValueError("export requires verl driver capture provenance")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    with Path(capsule).open("rb") as stream:
        binary = stream.read(2) == b"PK"
    fixture_name = "capsule.rlrp" if binary else "capsule.json"
    shutil.copyfile(capsule, directory / fixture_name)
    test_path = directory / "test_replay.py"
    test_path.write_text(
        '"""Generated regression for a locally installed verl candidate. Review before use."""\n'
        "from pathlib import Path\n"
        "from rlreplay.artifacts import read\n"
        "from rlreplay.contracts.grpo import check\n"
        "from rlreplay.replay import run_verl\n\n"
        "def test_captured_advantage_contract():\n"
        f'    evidence = read(Path(__file__).with_name("{fixture_name}"))\n'
        "    replayed = run_verl(evidence, allow_revision_change=True)\n"
        "    finding = check(replayed)\n"
        '    assert finding.verdict == "PASS", finding.to_dict()\n',
        encoding="utf-8",
    )
    return test_path
