"""Hook ownership, cadence and failure isolation without importing TRL or torch."""

import sys
from types import SimpleNamespace

import pytest

from rlreplay.adapters.trl_runtime import GRPOCaptureMixin, source_provenance
from rlreplay.artifacts import read
from rlreplay.capture import Recorder
from rlreplay.contracts.grpo import check


class Tensor:
    def __init__(self, values, dtype="torch.float32"):
        self.values = values
        self.dtype = dtype
        self.device = "cpu"
        self.shape = (len(values),)
        if values and isinstance(values[0], list):
            self.shape += (len(values[0]),)
        self.conversions = 0

    def detach(self):
        return self

    def tolist(self):
        self.conversions += 1
        return self.values

    def __getitem__(self, item):
        rows, column = item
        return Tensor([row[column] for row in self.values[rows]])


class NativeTrainer:
    def __init__(self, *, scale_rewards="group"):
        self.accelerator = SimpleNamespace(num_processes=1)
        self.reward_funcs = [object()]
        self.reward_weights = Tensor([1.0])
        self.scale_rewards = scale_rewards
        self.multi_objective_aggregation = "sum_then_normalize"
        self.mask_truncated_completions = False
        self.use_vllm = False
        self.tools = []
        self.environment_factories = None
        self._is_vlm = False
        self.rollout_func = None
        self.use_transformers_continuous_batching = False
        self.model = SimpleNamespace(training=True)
        self.args = SimpleNamespace(
            num_iterations=1, steps_per_generation=1, gradient_accumulation_steps=1
        )
        self.num_generations = 3
        self.state = SimpleNamespace(global_step=0)
        self.rewards = Tensor([[0.0], [1.0], [2.0]])
        self.output = {
            "advantages": Tensor([-0.9999000099990001, 0.0, 0.9999000099990001]),
            "completion_mask": Tensor([[1, 0], [1, 1], [1, 0]]),
        }
        self.native_calls = 0
        self.failure = None

    def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
        return self.rewards

    def _generate_and_score_completions(self, inputs):
        self.native_calls += 1
        self.returned_rewards = self._calculate_rewards(inputs, None, None, None)
        if self.failure is not None:
            raise self.failure
        return self.output


class ObservedTrainer(GRPOCaptureMixin, NativeTrainer):
    pass


@pytest.fixture
def trainer(tmp_path, monkeypatch, evidence):
    provenance = {**evidence.provenance, "framework": "trl"}
    from rlreplay.adapters.trl import SUPPORTED_REVISION

    provenance["framework_revision"] = SUPPORTED_REVISION
    monkeypatch.setattr(
        "rlreplay.adapters.trl_runtime.source_provenance", lambda _trainer: provenance
    )
    recorder = Recorder(tmp_path, check=True)
    observer = ObservedTrainer(capture_recorder=recorder, capture_run_id="test")
    yield observer
    recorder.close()


def inputs():
    return [{"rlreplay_group_id": "g"} for _ in range(3)]


def test_runtime_captures_without_changing_native_result(trainer):
    result = trainer._generate_and_score_completions(inputs())
    assert result is trainer.output
    assert trainer.returned_rewards is trainer.rewards
    assert trainer.native_calls == 1
    result["advantages"].values[0] = 999
    complete = next(
        event
        for event in trainer.rlreplay_recorder.close()
        if event["state"] == "COMPLETE"
    )
    captured = read(complete["path"])
    assert check(captured).verdict == "PASS"
    assert captured.samples[0].sample_id == "test/1/0"
    assert trainer.rlreplay_stats["captured_samples"] == 3
    assert trainer._rlreplay_pending is None


def test_native_failure_propagates_and_releases_reservation(trainer):
    trainer.failure = RuntimeError("private training exception")
    with pytest.raises(RuntimeError) as failure:
        trainer._generate_and_score_completions(inputs())
    assert failure.value is trainer.failure
    assert trainer.rlreplay_recorder.status()["reserved"] == 0
    assert trainer._rlreplay_pending is None
    assert not trainer.rlreplay_events


def test_bad_input_is_reported_and_training_runs(trainer):
    assert trainer._generate_and_score_completions([{}] * 3) is trainer.output
    assert trainer.native_calls == 1
    assert trainer.rlreplay_events[-1] == {
        "state": "ERROR",
        "reason": "KeyError",
        "phase": "before",
    }
    assert trainer.rlreplay_recorder.status()["reserved"] == 0


def test_sampling_and_admission_precede_payload_access(trainer, monkeypatch):
    trainer._rlreplay_cadence = 2
    trainer.rlreplay_stats["calls"] = 1
    assert trainer._generate_and_score_completions([{}] * 3) is trainer.output
    assert trainer.rlreplay_stats["cadence_skipped_calls"] == 1
    monkeypatch.setattr(trainer.rlreplay_recorder, "reserve", lambda: None)
    assert trainer._generate_and_score_completions([{}] * 3) is trainer.output
    assert trainer.rlreplay_stats["admission_rejected_calls"] == 1
    assert not trainer.rlreplay_events


def test_output_budget_rejects_before_tensor_conversion(trainer):
    trainer._rlreplay_limit = 3
    trainer._generate_and_score_completions(inputs())
    assert trainer.output["completion_mask"].conversions == 0
    assert trainer.output["advantages"].conversions == 0
    assert trainer.rlreplay_events[-1]["phase"] == "after"
    assert trainer.rlreplay_recorder.status()["reserved"] == 0


def test_reward_error_cannot_reuse_previous_observation(trainer):
    trainer.rewards.dtype = "torch.float64"
    trainer._generate_and_score_completions(inputs())
    assert trainer.rlreplay_events[-1]["phase"] == "rewards"
    assert trainer.rlreplay_stats["submitted_calls"] == 0
    trainer.rewards.dtype = "torch.float32"
    trainer._generate_and_score_completions(inputs())
    complete = [
        e for e in trainer.rlreplay_recorder.close() if e["state"] == "COMPLETE"
    ]
    assert len(complete) == 1
    assert read(complete[0]["path"]).samples[0].sample_id == "test/2/0"


def test_resolved_config_changes_are_rejected_per_call(trainer):
    trainer.scale_rewards = "batch"
    assert trainer._generate_and_score_completions(inputs()) is trainer.output
    assert trainer.rlreplay_events[-1]["phase"] == "before"
    assert trainer.rlreplay_stats["submitted_calls"] == 0


@pytest.mark.parametrize("outer_skipped", [False, True])
def test_reentrant_generation_cancels_capture_without_losing_native_calls(
    trainer, monkeypatch, outer_skipped
):
    original = NativeTrainer._generate_and_score_completions
    entered = False

    def reenter(self, batch):
        nonlocal entered
        if not entered:
            entered = True
            self._generate_and_score_completions(batch)
        return original(self, batch)

    monkeypatch.setattr(NativeTrainer, "_generate_and_score_completions", reenter)
    if outer_skipped:
        trainer._rlreplay_cadence = 2
        trainer.rlreplay_stats["calls"] = 1
    assert trainer._generate_and_score_completions(inputs()) is trainer.output
    assert trainer.native_calls == 2
    assert trainer.rlreplay_recorder.status()["reserved"] == 0
    assert trainer.rlreplay_stats["submitted_calls"] == 0
    assert trainer.rlreplay_stats["UNSUPPORTED"] == 1
    assert trainer._rlreplay_pending is None
    assert trainer._rlreplay_active is False


@pytest.mark.parametrize("mode", ["custom", "continuous", "eval", "tool_mask"])
def test_unqualified_runtime_paths_are_reported(trainer, mode):
    if mode == "custom":
        trainer.rollout_func = object()
    elif mode == "continuous":
        trainer.use_transformers_continuous_batching = True
    elif mode == "eval":
        trainer.model.training = False
    else:
        trainer.output["tool_mask"] = Tensor([[1, 0]] * 3)
    assert trainer._generate_and_score_completions(inputs()) is trainer.output
    assert trainer.rlreplay_stats["ERROR"] == 1
    assert trainer.rlreplay_stats["submitted_calls"] == 0
    assert trainer.rlreplay_recorder.status()["reserved"] == 0


def test_overridden_native_hooks_are_rejected_before_source_claim(monkeypatch):
    class Override(NativeTrainer):
        def _calculate_rewards(self, *args):
            return super()._calculate_rewards(*args)

    class Candidate(GRPOCaptureMixin, Override):
        pass

    monkeypatch.setitem(sys.modules, "trl", SimpleNamespace(GRPOTrainer=NativeTrainer))
    monkeypatch.setitem(sys.modules, "trl.trainer.utils", SimpleNamespace(nanstd=None))
    with pytest.raises(ValueError, match="native hook dispatch"):
        source_provenance(object.__new__(Candidate))
