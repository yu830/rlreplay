"""Capture a real pinned verl CPU driver call, without a model or Ray cluster.

Install the qualified verl revision and its dependencies in your environment.
Run with a new output directory. The default is healthy; --case known-no-std
records the pinned vectorized driver's known mismatch and returns exit code 2.
"""

import argparse
import json
import uuid
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="new output directory")
    parser.add_argument(
        "--case", choices=("healthy", "known-no-std"), default="healthy"
    )
    args = parser.parse_args()

    import numpy as np
    import torch
    from verl import DataProto
    from verl.trainer.ppo.core_algos import AdvantageEstimator
    from verl.trainer.ppo.ray_trainer import compute_advantage

    from rlreplay import __version__
    from rlreplay.adapters.verl import (
        SAMPLE_ID_FIELD,
        SUPPORTED_REVISION,
        VerlObserver,
        algorithm_digest,
        driver_digest,
        groupwise_digest,
        membership,
    )
    from rlreplay.artifacts import read
    from rlreplay.capture import Recorder
    from rlreplay.contracts.grpo import check
    from rlreplay.core import Profile
    from rlreplay.replay import run_verl, semantic_replay

    actual_sources = {
        "algorithm_sha256": algorithm_digest(),
        "entrypoint_sha256": driver_digest(),
        "groupwise_sha256": groupwise_digest(),
    }
    qualified_sources = {
        "algorithm_sha256": "369b614f239f237a04a38a40b1877cd6fd14aa7bd5f9ff4fe4d7758b682a6948",
        "entrypoint_sha256": "eb41f99ccc66976eab14ebe35aa1aa5b7318e02469240ef1569ac4c884c94312",
        "groupwise_sha256": "409403d0002ff0d1fd1c41812163b1cfa52a52248677aab00e6ffcf3b937b442",
    }
    if actual_sources != qualified_sources:
        raise ValueError(
            f"this example requires qualified verl source {SUPPORTED_REVISION}"
        )
    torch.set_num_threads(1)
    args.directory.mkdir(parents=True, exist_ok=False)

    prompts = DataProto.from_dict(
        tensors={
            "token_level_rewards": torch.zeros((2, 3), dtype=torch.float32),
            "response_mask": torch.ones((2, 3), dtype=torch.float32),
        },
        non_tensors={"uid": np.array(["group-a", "group-b"], dtype=object)},
    )
    data = prompts.repeat(repeat_times=2, interleave=True)
    data.non_tensor_batch[SAMPLE_ID_FIELD] = np.array(
        ["a-0", "a-1", "b-0", "b-1"], dtype=object
    )
    data.batch["token_level_rewards"] = torch.tensor(
        [[1.0, 0, 0], [3.0, 0, 0], [2.0, 0, 0], [6.0, 0, 0]], dtype=torch.float32
    )
    data.batch["response_mask"] = torch.tensor(
        [[1, 1, 0], [1, 0, 0], [1, 1, 1], [1, 0, 0]], dtype=torch.float32
    )
    # Bind producer identities after repetition, before the framework reorders rows.
    producer_membership = membership(data)
    data.reorder(torch.tensor([2, 0, 3, 1]))

    known_mismatch = args.case == "known-no-std"
    estimator = (
        AdvantageEstimator.GRPO_VECTORIZED
        if known_mismatch
        else AdvantageEstimator.GRPO
    )
    normalize = not known_mismatch
    expected = "SUSPECTED_MISMATCH" if known_mismatch else "PASS"
    provenance = {
        "framework": "verl",
        "framework_revision": SUPPORTED_REVISION,
        **actual_sources,
        "adapter_version": __version__,
        "entrypoint": "verl.trainer.ppo.ray_trainer.compute_advantage",
        "phase": "post_advantage",
        "receipt_origin": "synthetic_after_repeat_before_reorder",
        "device": "cpu",
        "dtype": "float32",
        "run_id": str(uuid.uuid4()),
        "step": "0",
    }
    recorder = Recorder(args.directory / "capsules", check=True)
    observer = VerlObserver(
        recorder,
        Profile(implementation=estimator.value, normalize_std=normalize),
        provenance,
        max_groups=2,
    )
    try:
        result = observer.around(
            compute_advantage,
            data,
            producer_membership,
            adv_estimator=estimator,
            norm_adv_by_std_in_grpo=normalize,
        )
        assert result is data
    finally:
        receipts = recorder.close()
        (args.directory / "receipts.json").write_text(
            json.dumps(receipts, indent=2), encoding="utf-8"
        )
    completed = [receipt for receipt in receipts if receipt["state"] == "COMPLETE"]
    assert len(completed) == 1, receipts
    assert completed[0]["check_state"] == "COMPLETE"
    assert completed[0]["finding_summary"]["verdict"] == expected
    evidence = read(completed[0]["path"])
    finding = check(evidence)
    assert finding.verdict == expected, finding
    replayed = run_verl(evidence)
    assert replayed.samples == evidence.samples
    assert check(replayed).verdict == expected
    report = {
        "rlreplay_version": __version__,
        "framework_revision": SUPPORTED_REVISION,
        "case": args.case,
        "estimator": estimator.value,
        "normalize_std_requested": normalize,
        "capsule": completed[0]["path"],
        "finding": finding.to_dict(),
        "semantic_replay": semantic_replay(evidence),
        "native_replay_finding": check(replayed).to_dict(),
        "observed_advantages": {
            sample.sample_id: sample.advantage for sample in evidence.samples
        },
        "coverage": observer.coverage(),
        "observer_events": list(observer.events),
        "recorder_status": recorder.status(),
        "scope": "Preconstructed CPU DataProto and real driver; no rollout or training step",
    }
    (args.directory / "result.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 2 if known_mismatch else 0


if __name__ == "__main__":
    raise SystemExit(main())
