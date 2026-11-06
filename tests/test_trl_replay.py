"""Frozen-provider bundle boundaries; real tiny-model input is synthetic and public."""

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from rlreplay import trl_replay
from rlreplay.cli import main

FIXTURE = Path(__file__).parent / "fixtures/trl-native.json"


@pytest.fixture
def bundle():
    return trl_replay.read_trl(FIXTURE)


def test_native_observations_survive_round_trip(bundle, tmp_path):
    path = trl_replay.write_trl(bundle, tmp_path / "bundles")
    restored = trl_replay.read_trl(path)
    assert restored == bundle
    assert (
        restored["scalar_advantages"]
        == [-0.9998999834060669, 0.0, 0.9998999834060669] * 2
    )
    assert len(restored["generation"]) == 9
    assert len(restored["evidence"]["samples"]) == 6


def test_changed_payload_is_rejected_before_replay(tmp_path):
    envelope = json.loads(FIXTURE.read_text())
    envelope["bundle"]["rewards"][0][0] += 1
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(envelope))
    with pytest.raises(ValueError, match="digest mismatch"):
        trl_replay.read_trl(path)


@pytest.mark.parametrize(
    "fault", ["groups", "beta", "extras", "rewards", "scalar", "mask", "factory"]
)
def test_unsupported_bundle_never_creates_output(bundle, tmp_path, fault):
    if fault == "groups":
        bundle["inputs"][1]["rlreplay_group_id"] = "another-group"
    elif fault == "beta":
        bundle["configuration"]["beta"] = 0.1
    elif fault == "extras":
        bundle["generation"][6] = {"unrecorded": "context"}
    elif fault == "rewards":
        bundle["rewards"][0] = [1.0, 2.0]
    elif fault == "scalar":
        bundle["scalar_advantages"][0] += 1
    elif fault == "mask":
        bundle["observed"]["completion_mask"][0][0] = 0
    else:
        bundle["trainer_factory"] = "untrusted_module:execute"
    output = tmp_path / "unsupported"
    with pytest.raises((ValueError, TypeError)):
        trl_replay.write_trl(bundle, output)
    assert not output.exists()


def test_bundle_budget_is_checked_on_write_and_read(bundle, tmp_path):
    bundle["inputs"][0]["prompt"] = "x" * trl_replay.MAX_CAPSULE_BYTES
    with pytest.raises(ValueError, match="byte budget"):
        trl_replay.write_trl(bundle, tmp_path / "oversized")
    assert not (tmp_path / "oversized").exists()
    path = tmp_path / "huge.json"
    path.write_bytes(b" " * (trl_replay.MAX_CAPSULE_BYTES + 1))
    with pytest.raises(ValueError, match="byte budget"):
        trl_replay.read_trl(path)


@pytest.mark.parametrize("instance_override", [False, True])
def test_provider_restores_ownership_and_preserves_original_exception(
    instance_override,
):
    class Trainer:
        def generate(self):
            return "class method"

    trainer = Trainer()
    original = lambda: "instance method"
    if instance_override:
        trainer.generate = original
    failure = RuntimeError("native failure")
    with (
        pytest.raises(RuntimeError) as raised,
        trl_replay._providers(trainer, generate=lambda: "frozen"),
    ):
        assert trainer.generate() == "frozen"
        raise failure
    assert raised.value is failure
    if instance_override:
        assert trainer.generate is original
    else:
        assert "generate" not in trainer.__dict__
        assert trainer.generate() == "class method"


@pytest.mark.parametrize(
    "mismatch", ["configuration", "tokenizer", "runtime", "source"]
)
def test_replay_identity_gates_before_framework_execution(
    bundle, monkeypatch, mismatch
):
    provenance = dict(bundle["evidence"]["provenance"])
    configuration = copy.deepcopy(bundle["configuration"])
    tokenizer = copy.deepcopy(bundle["tokenizer"])
    runtime = dict(bundle["runtime"])
    if mismatch == "configuration":
        configuration["pad_to_multiple_of"] = 8
    elif mismatch == "tokenizer":
        tokenizer["pad_token_id"] = 99
    elif mismatch == "runtime":
        runtime["transformers"] = "different"
    else:
        provenance["algorithm_sha256"] = "changed"
    monkeypatch.setattr(trl_replay, "_native", lambda trainer: None)
    monkeypatch.setattr(
        trl_replay, "implementation_provenance", lambda **kwargs: provenance
    )
    monkeypatch.setattr(trl_replay, "_configuration", lambda trainer: configuration)
    monkeypatch.setattr(trl_replay, "_tokenizer", lambda trainer: tokenizer)
    monkeypatch.setattr(trl_replay, "_runtime", lambda: runtime)
    with pytest.raises(ValueError, match="differs from capture"):
        trl_replay.run_trl(bundle, object())


def test_export_uses_explicit_factory_without_importing_it(tmp_path):
    target = tmp_path / "regression"
    path = trl_replay.export_trl_test(
        FIXTURE, target, trainer_factory="not_installed_trainer:make_trainer"
    )
    compile(path.read_text(), str(path), "exec")
    assert "not_installed_trainer:make_trainer" in path.read_text()
    assert trl_replay.read_trl(target / "trl-bundle.json") == trl_replay.read_trl(
        FIXTURE
    )
    with pytest.raises(FileExistsError):
        trl_replay.export_trl_test(FIXTURE, target, trainer_factory="trainer:build")


def test_cli_requires_explicit_factory_and_does_not_reinterpret_old_capsules(
    tmp_path, capsys
):
    assert main(["replay", str(FIXTURE), "--runner", "trl"]) == 4
    assert "trainer-factory" in json.loads(capsys.readouterr().err)["reason"]
    destination = tmp_path / "missing-factory"
    assert main(["export-test", str(FIXTURE), str(destination), "--runner", "trl"]) == 4
    assert not destination.exists()
    assert json.loads(capsys.readouterr().err)["state"] == "ERROR"
    assert main(["check", str(FIXTURE)]) == 4
    assert "capsule envelope" in json.loads(capsys.readouterr().err)["reason"]


@pytest.mark.parametrize(
    "fault",
    [None, "completion_mask", "completion_ids", "prompt_mask", "num_items_in_batch"],
)
def test_replay_finding_rejects_layout_changes_even_with_correct_scalar(
    bundle, monkeypatch, fault
):
    class Tensor:
        device = "cpu"
        dtype = "torch.float32"

        def __init__(self, values):
            self.values = values

        def detach(self):
            return self

        def tolist(self):
            return copy.deepcopy(self.values)

    class Trainer:
        def __init__(self):
            self.model = SimpleNamespace(forward=lambda: None)

        def _generate(self, prompts):
            raise AssertionError("generation must be frozen")

        def _calculate_rewards(self, *args):
            raise AssertionError("reward computation must be frozen")

        def _generate_and_score_completions(self, inputs):
            prompts = [row["prompt"] for row in inputs]
            generation = self._generate(prompts)
            self._calculate_rewards(inputs, prompts, generation[3], generation[1])
            output = copy.deepcopy(bundle["observed"])
            output["advantages"] = list(bundle["scalar_advantages"])
            if fault in ("completion_mask", "prompt_mask"):
                output[fault] = [[0] * len(row) for row in output[fault]]
            elif fault == "completion_ids":
                output[fault][0][0] += 1
            elif fault == "num_items_in_batch":
                output[fault] -= 1
            return {key: Tensor(value) for key, value in output.items()}

    monkeypatch.setattr(trl_replay, "_native", lambda trainer: None)
    monkeypatch.setattr(
        trl_replay,
        "implementation_provenance",
        lambda **kwargs: bundle["evidence"]["provenance"],
    )
    monkeypatch.setattr(
        trl_replay, "_configuration", lambda trainer: bundle["configuration"]
    )
    monkeypatch.setattr(trl_replay, "_tokenizer", lambda trainer: bundle["tokenizer"])
    monkeypatch.setattr(trl_replay, "_runtime", lambda: bundle["runtime"])
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            tensor=lambda values, **kwargs: Tensor(values), float32="float32"
        ),
    )
    trainer = Trainer()
    result = trl_replay.run_trl(bundle, trainer)
    assert result["provider_calls"] == {"generation": 1, "rewards": 1}
    assert "_generate" not in trainer.__dict__
    if fault:
        assert result["finding"]["verdict"] == "CONTRACT_VIOLATION"
        assert result["finding"]["reason"] == "trl_frozen_generation_layout"
        assert result["layout_differences"] == [fault]
    else:
        assert result["finding"]["verdict"] == "PASS"
        assert result["native_observation_equal"]


def test_frozen_layout_uses_left_prompt_and_right_completion_padding(bundle):
    bundle["configuration"]["pad_to_multiple_of"] = 4
    bundle["generation"][0] = [[3, 4], [5]]
    bundle["generation"][1] = [[9], [7, 8]]
    assert trl_replay._expected_layout(bundle) == {
        "prompt_ids": [[0, 0, 3, 4], [0, 0, 0, 5]],
        "prompt_mask": [[0, 0, 1, 1], [0, 0, 0, 1]],
        "completion_ids": [[9, 0, 0, 0], [7, 8, 0, 0]],
        "completion_mask": [[1, 0, 0, 0], [1, 1, 0, 0]],
        "num_items_in_batch": 3,
    }
