"""Capture one genuine CPU GRPOTrainer step with a local tiny model/tokenizer.

Requires qualified TRL a16af6502c4a1922045be533c94f66dd7c981374 and its
dependencies. No pretrained model, dataset download, GPU, or external tracker is
used. Rewards are synthetic rank-within-group values, not model-quality scores.
"""

import argparse
import json
import uuid
from collections import Counter
from importlib.metadata import version
from pathlib import Path


def synthetic_reward(completions, rlreplay_group_id, **kwargs):
    counts = Counter()
    values = []
    for group in rlreplay_group_id:
        values.append(float(counts[group]))
        counts[group] += 1
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="new output directory")
    args = parser.parse_args()

    import torch
    from datasets import Dataset
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import (
        GPT2Config,
        GPT2LMHeadModel,
        PreTrainedTokenizerFast,
        set_seed,
    )
    from trl import GRPOConfig, GRPOTrainer

    from rlreplay import __version__
    from rlreplay.adapters.trl import SUPPORTED_REVISION
    from rlreplay.adapters.trl_runtime import GRPOCaptureMixin
    from rlreplay.artifacts import read
    from rlreplay.capture import Recorder
    from rlreplay.contracts.grpo import check
    from rlreplay.replay import semantic_replay

    class CapturingTrainer(GRPOCaptureMixin, GRPOTrainer):
        pass

    args.directory.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    set_seed(20261105)
    backend = Tokenizer(
        WordLevel(
            {
                "[PAD]": 0,
                "[BOS]": 1,
                "[EOS]": 2,
                "[UNK]": 3,
                "alpha": 4,
                "beta": 5,
                "question": 6,
                "answer": 7,
                **{f"token{i}": i for i in range(8, 32)},
            },
            unk_token="[UNK]",
        )
    )
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="[PAD]",
        bos_token="[BOS]",
        eos_token="[EOS]",
        unk_token="[UNK]",
        padding_side="left",
    )
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=32,
            n_layer=1,
            n_head=2,
            n_embd=16,
            n_positions=64,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
        )
    )
    before = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters()
    }
    configuration = GRPOConfig(
        output_dir=str(args.directory / "trainer-output"),
        use_cpu=True,
        per_device_train_batch_size=6,
        num_generations=3,
        max_completion_length=4,
        max_steps=1,
        learning_rate=1e-3,
        gradient_accumulation_steps=1,
        steps_per_generation=1,
        num_iterations=1,
        beta=0.0,
        scale_rewards="group",
        multi_objective_aggregation="sum_then_normalize",
        use_vllm=False,
        fp16=False,
        bf16=False,
        gradient_checkpointing=False,
        mask_truncated_completions=False,
        report_to="none",
        save_strategy="no",
        logging_strategy="no",
        disable_tqdm=True,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        shuffle_dataset=False,
        optim="adamw_torch",
        seed=20261105,
    )
    dataset = Dataset.from_list(
        [
            {"prompt": "alpha question", "rlreplay_group_id": "group-alpha"},
            {"prompt": "beta question", "rlreplay_group_id": "group-beta"},
        ]
    )
    recorder = Recorder(args.directory / "capsules", check=True)
    try:
        trainer = CapturingTrainer(
            model=model,
            args=configuration,
            reward_funcs=synthetic_reward,
            train_dataset=dataset,
            processing_class=tokenizer,
            capture_recorder=recorder,
            capture_run_id=str(uuid.uuid4()),
        )
        assert str(trainer.accelerator.device) == "cpu"
        trainer.train()
    finally:
        receipts = recorder.close()
        (args.directory / "receipts.json").write_text(
            json.dumps(receipts, indent=2), encoding="utf-8"
        )
    completed = [receipt for receipt in receipts if receipt["state"] == "COMPLETE"]
    assert len(completed) == 1, receipts
    assert completed[0]["check_state"] == "COMPLETE"
    assert completed[0]["finding_summary"]["verdict"] == "PASS"
    evidence = read(completed[0]["path"])
    finding = check(evidence)
    assert finding.verdict == "PASS", finding
    assert len(evidence.samples) == 6
    assert len({sample.group_id for sample in evidence.samples}) == 2
    assert trainer.state.global_step == 1
    changed = [
        name
        for name, parameter in model.named_parameters()
        if not torch.equal(parameter.detach(), before[name])
    ]
    assert changed, "the native optimizer step did not change model parameters"
    replayed = semantic_replay(evidence)
    assert replayed["finding"]["verdict"] == "PASS"
    report = {
        "rlreplay_version": __version__,
        "trl_version": version("trl"),
        "framework_revision": SUPPORTED_REVISION,
        "global_step": trainer.state.global_step,
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "changed_parameter_tensors": len(changed),
        "capsule": completed[0]["path"],
        "finding": finding.to_dict(),
        "semantic_replay": replayed,
        "capture_stats": dict(trainer.rlreplay_stats),
        "capture_events": list(trainer.rlreplay_events),
        "recorder_stats": dict(recorder.stats),
        "recorder_status": recorder.status(),
        "scope": "One CPU tiny-model training step; synthetic rewards; semantic replay",
        "native_trl_replay": "UNSUPPORTED",
    }
    (args.directory / "result.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
