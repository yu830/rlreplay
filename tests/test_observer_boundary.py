"""Observer failure isolation without optional framework dependencies."""

from collections import Counter
from types import SimpleNamespace

import pytest

from rlreplay.adapters.verl import SUPPORTED_REVISION, Snapshot, VerlObserver
from rlreplay.capture import Recorder
from rlreplay.core import Profile


def test_profile_is_bound_to_actual_call():
    observer = VerlObserver(None, Profile(normalize_std=False), {}, mode="sampled")
    sentinel = object()

    def training(data, **_kwargs):
        return data

    assert (
        observer.around(training, sentinel, (), adv_estimator="grpo_vectorized")
        is sentinel
    )
    assert observer.events[-1]["reason"] == "profile_call_mismatch"


@pytest.mark.parametrize(
    "profile", [Profile(epsilon=1e-4), Profile(ddof=0), Profile(implementation="rloo")]
)
def test_verl_rejects_unmapped_semantics(profile):
    observer = VerlObserver(None, profile, {})
    sentinel = object()

    def training(data, **_kwargs):
        return data

    assert (
        observer.around(training, sentinel, (), adv_estimator=profile.implementation)
        is sentinel
    )
    assert observer.events[-1]["reason"] == "profile_call_mismatch"


def test_observer_error_message_is_redacted(monkeypatch, caplog):
    observer = VerlObserver(None, Profile(), {})
    sentinel = object()

    def broken_capture(_data, _receipt):
        raise ValueError("SENSITIVE-REWARD-PAYLOAD")

    def training(data, **_kwargs):
        return data

    monkeypatch.setattr(observer, "before", broken_capture)
    assert (
        observer.around(training, sentinel, (), adv_estimator="grpo_vectorized")
        is sentinel
    )
    assert "ValueError" in caplog.text
    assert "SENSITIVE-REWARD-PAYLOAD" not in caplog.text
    assert observer.events[-1] == {
        "state": "ERROR",
        "reason": "ValueError",
        "phase": "before",
    }


def test_sampling_and_admission_skip_payload_access(monkeypatch):
    class DecliningRecorder:
        def __init__(self):
            self.stats = Counter()

        def reserve(self):
            self.stats["attempted"] += 1

    recorder = DecliningRecorder()
    observer = VerlObserver(recorder, Profile(), {}, sample_every=3)
    sentinel = object()

    def must_not_snapshot(*_args):
        pytest.fail("payload accessed after admission rejection")

    def training(data, **_kwargs):
        return data

    monkeypatch.setattr(observer, "_snapshot", must_not_snapshot)
    for _ in range(7):
        assert (
            observer.around(training, sentinel, (), adv_estimator="grpo_vectorized")
            is sentinel
        )
    coverage = observer.coverage()
    assert coverage["observer"]["calls"] == 7
    assert coverage["observer"]["cadence_skipped_calls"] == 4
    assert coverage["observer"]["admission_rejected_calls"] == 3
    assert coverage["recorder"]["attempted"] == 3


def test_training_failure_releases_capacity_and_preserves_error(tmp_path, monkeypatch):
    recorder = Recorder(tmp_path)
    observer = VerlObserver(recorder, Profile(), {})
    original_error = RuntimeError("training failed")

    def snapshot(_data, _receipt, reservation):
        return Snapshot((), (), reservation)

    def training(_data, **_kwargs):
        raise original_error

    monkeypatch.setattr(observer, "_snapshot", snapshot)
    with pytest.raises(RuntimeError) as raised:
        observer.around(training, None, (), adv_estimator="grpo_vectorized")
    assert raised.value is original_error
    assert recorder.stats["CANCELLED"] == 1
    assert any(
        event.get("reason") == "training_exception" for event in recorder.close()
    )


def test_budgets_precede_reward_and_output_tensor_copy(evidence):
    class NoCopyTensor:
        device = "cpu"
        dtype = "torch.float32"

        def __init__(self, shape):
            self.shape = shape

        def __getitem__(self, _index):
            pytest.fail("over-budget tensor was accessed")

    cancellations = []
    sink = SimpleNamespace(
        cancel=lambda reservation, **kwargs: cancellations.append((reservation, kwargs))
    )
    observer = VerlObserver(
        sink, Profile(), {"framework_revision": SUPPORTED_REVISION}, max_elements=8
    )
    observer._runtime_recorded = True
    data = SimpleNamespace(
        batch={
            "token_level_rewards": NoCopyTensor((2, 1000)),
            "response_mask": NoCopyTensor((2, 2)),
            "advantages": NoCopyTensor((2, 1000)),
        },
        non_tensor_batch={"rlreplay_sample_id": ["a0", "a1"], "uid": ["a", "a"]},
    )
    assert observer._snapshot(data, evidence.membership, "r") is None
    assert observer.events[-1]["reason"] == "element_budget"
    assert (
        observer.after(data, Snapshot(evidence.samples, evidence.membership, "r"))
        is None
    )
    assert cancellations == [("r", {"reason": "output_element_budget"})]
