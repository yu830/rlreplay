"""Capture and replay a genuine CPU TRL method with frozen providers.

The example uses local synthetic prompts, a tiny random model and no downloads.
Generation and reward providers are frozen for replay; the complete native
Trainer method computes masks, reward aggregation and advantages again.
"""

import argparse
import json
import tempfile
from pathlib import Path

if __package__:
    from .trl_capture import build_components, synthetic_reward
else:
    from trl_capture import build_components, synthetic_reward


def make_trainer():
    """Return a real initialized Trainer for the example's qualified CPU route."""
    from trl import GRPOTrainer

    directory = tempfile.TemporaryDirectory(prefix="rlreplay-trl-native-")
    model, configuration, dataset, tokenizer = build_components(Path(directory.name))
    trainer = GRPOTrainer(
        model=model,
        args=configuration,
        reward_funcs=synthetic_reward,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer._rlreplay_example_directory = directory
    return trainer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="new output directory")
    args = parser.parse_args()

    from rlreplay.trl_replay import (
        capture_trl,
        export_trl_test,
        read_trl,
        run_trl,
        write_trl,
    )

    args.directory.mkdir(parents=True, exist_ok=False)
    trainer = make_trainer()
    inputs = [
        dict(row)
        for row in trainer.train_dataset
        for _ in range(trainer.num_generations)
    ]
    bundle = capture_trl(trainer, inputs)
    bundle_path = write_trl(bundle, args.directory / "capture")
    replayed = run_trl(read_trl(bundle_path), make_trainer())
    assert replayed["finding"]["verdict"] == "PASS", replayed
    assert replayed["native_observation_equal"], replayed
    exported = export_trl_test(
        bundle_path,
        args.directory / "regression",
        trainer_factory="examples.trl_native_replay:make_trainer",
    )
    report = {
        "bundle": str(bundle_path),
        "replay": replayed,
        "exported_test": str(exported),
        "trainer_factory": "examples.trl_native_replay:make_trainer",
        "scope": (
            "CPU native-method replay with frozen generation/reward providers; "
            "synthetic inputs; no optimizer step or generation replay"
        ),
    }
    (args.directory / "result.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
