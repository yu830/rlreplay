"""Explicit CPU TRL method replay with frozen generation and reward providers.

Bundles contain text and token IDs. They are opt-in debugging artifacts, separate
from the default numeric Evidence/Recorder path. The caller owns the real Trainer.
"""

import copy
import hashlib
import hmac
import importlib
import json
import math
import shutil
import uuid
from contextlib import contextmanager
from dataclasses import replace
from importlib.metadata import version
from pathlib import Path

from rlreplay._directory import writer_directory
from rlreplay.adapters.trl import (
    SUPPORTED_REVISION,
    from_frozen_batch,
    profile_from_config,
)
from rlreplay.adapters.trl_runtime import (
    implementation_provenance,
    resolved_configuration,
)
from rlreplay.artifacts import (
    MAX_CAPSULE_BYTES,
    _canonical,
    _unique_object,
    write_payload,
)
from rlreplay.contracts.grpo import check
from rlreplay.core import Evidence, Finding
from rlreplay.replay import diff

FORMAT = "rlreplay-trl-native/1"
OBSERVATIONS = (
    "prompt_ids",
    "prompt_mask",
    "completion_ids",
    "completion_mask",
    "num_items_in_batch",
)
FIELDS = {
    "configuration",
    "tokenizer",
    "runtime",
    "inputs",
    "generation",
    "rewards",
    "scalar_advantages",
    "observed",
    "evidence",
}


def _copy(value):
    payload = _canonical(value)
    if len(payload) > MAX_CAPSULE_BYTES:
        raise ValueError("TRL replay bundle exceeds byte budget")
    return json.loads(payload)


def _inputs(rows, generations):
    if not isinstance(rows, list) or not rows or len(rows) % generations:
        raise ValueError("TRL replay requires complete input groups")
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"prompt", "rlreplay_group_id"}:
            raise ValueError("TRL replay requires text prompts and explicit group IDs")
        if any(not isinstance(value, str) or not value for value in row.values()):
            raise ValueError(
                "TRL replay supports nonempty non-conversational text only"
            )
    groups = [row["rlreplay_group_id"] for row in rows]
    chunks = [groups[i : i + generations] for i in range(0, len(groups), generations)]
    if any(len(set(chunk)) != 1 for chunk in chunks) or len(
        {c[0] for c in chunks}
    ) != len(chunks):
        raise ValueError("TRL replay requires distinct, contiguous complete groups")


def _generation(value, count):
    if not isinstance(value, list) or len(value) != 9:
        raise ValueError("TRL generation must contain nine return values")
    if (
        any(value[i] is not None for i in (2, 4, 5, 7))
        or value[6] != {}
        or value[8] != []
    ):
        raise ValueError(
            "TRL replay excludes tools, sampling logprobs, images and extra fields"
        )
    for rows in value[:2]:
        if (
            not isinstance(rows, list)
            or len(rows) != count
            or any(
                not isinstance(row, list)
                or not row
                or any(
                    type(token) is not int or token < 0 or token >= 2**63
                    for token in row
                )
                for row in rows
            )
        ):
            raise ValueError("invalid frozen token IDs")
    if (
        not isinstance(value[3], list)
        or len(value[3]) != count
        or any(not isinstance(text, str) for text in value[3])
    ):
        raise ValueError("invalid frozen completion text")


def _configuration(trainer):
    configuration = resolved_configuration(trainer)
    configuration.update(
        num_iterations=trainer.num_iterations,
        beta=trainer.beta,
        pad_to_multiple_of=trainer.pad_to_multiple_of,
        gradient_checkpointing=trainer.args.gradient_checkpointing,
        per_device_train_batch_size=trainer.args.per_device_train_batch_size,
    )
    _check_configuration(configuration)
    if str(trainer.accelerator.device) != "cpu" or not trainer.model.training:
        raise ValueError("TRL replay requires CPU training mode")
    return configuration


def _check_configuration(configuration):
    if not isinstance(configuration, dict):
        raise TypeError("invalid TRL replay configuration")
    profile_from_config(configuration)
    if (
        configuration.get("beta") != 0
        or configuration.get("gradient_checkpointing") is not False
    ):
        raise ValueError("TRL replay requires beta=0 and checkpointing disabled")
    multiple = configuration.get("pad_to_multiple_of")
    if multiple is not None and (type(multiple) is not int or multiple < 1):
        raise ValueError("invalid padding multiple")


def _tokenizer(trainer):
    tokenizer = trainer._tokenizer
    if not tokenizer.is_fast:
        raise ValueError("TRL replay currently requires a fast tokenizer")
    return {
        "sha256": hashlib.sha256(
            _canonical(
                {
                    "backend": tokenizer.backend_tokenizer.to_str(),
                    "special_tokens": tokenizer.special_tokens_map,
                    "padding_side": tokenizer.padding_side,
                    "truncation_side": tokenizer.truncation_side,
                }
            )
        ).hexdigest(),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_ids": list(trainer.eos_token_ids),
    }


def _runtime():
    return {
        name: version(name)
        for name in ("trl", "transformers", "tokenizers", "accelerate")
    }


def _native(trainer):
    from trl import GRPOTrainer

    if not isinstance(trainer, GRPOTrainer):
        raise TypeError("TRL replay requires a real GRPOTrainer")
    for name in ("_generate", "_calculate_rewards", "_generate_and_score_completions"):
        if getattr(getattr(trainer, name), "__func__", None) is not getattr(
            GRPOTrainer, name
        ):
            raise ValueError("TRL replay requires native method dispatch")


@contextmanager
def _providers(trainer, **methods):
    absent = object()
    original = {name: trainer.__dict__.get(name, absent) for name in methods}
    try:
        for name, function in methods.items():
            setattr(trainer, name, function)
        yield
    finally:
        for name, value in original.items():
            if value is absent:
                delattr(trainer, name)
            else:
                setattr(trainer, name, value)


def _output(output):
    values = {}
    for name in (*OBSERVATIONS, "advantages"):
        tensor = output[name]
        if str(tensor.device) != "cpu":
            raise ValueError("TRL replay output must remain on CPU")
        if name == "advantages" and str(tensor.dtype) != "torch.float32":
            raise ValueError("TRL replay requires float32 advantages")
        values[name] = tensor.detach().tolist()
    return values


def _expected_layout(bundle):
    """Compute padding from frozen token lengths, independently of native outputs."""
    multiple = bundle["configuration"]["pad_to_multiple_of"]
    pad = bundle["tokenizer"]["pad_token_id"]
    expected = {}
    for name, rows, left in (
        ("prompt", bundle["generation"][0], True),
        ("completion", bundle["generation"][1], False),
    ):
        width = max(len(row) for row in rows)
        if multiple is not None:
            width = ((width + multiple - 1) // multiple) * multiple
        ids, masks = [], []
        for row in rows:
            padding = width - len(row)
            ids.append([pad] * padding + row if left else row + [pad] * padding)
            masks.append(
                [0] * padding + [1] * len(row)
                if left
                else [1] * len(row) + [0] * padding
            )
        expected[f"{name}_ids"] = ids
        expected[f"{name}_mask"] = masks
    expected["num_items_in_batch"] = sum(len(row) for row in bundle["generation"][1])
    return expected


def _evidence(inputs, rewards, scalars, masks, configuration, provenance, ids):
    return from_frozen_batch(
        sample_ids=ids,
        group_ids=[row["rlreplay_group_id"] for row in inputs],
        rewards=[row[0] for row in rewards],
        scalar_advantages=scalars,
        completion_masks=masks,
        membership=[
            (sample, row["rlreplay_group_id"])
            for sample, row in zip(ids, inputs, strict=True)
        ],
        configuration=configuration,
        provenance={**provenance, "framework_revision": SUPPORTED_REVISION},
    )


def _validate(bundle):
    if not isinstance(bundle, dict) or set(bundle) != FIELDS:
        raise ValueError("invalid TRL replay bundle fields")
    configuration = bundle["configuration"]
    _check_configuration(configuration)
    _inputs(bundle["inputs"], configuration["num_generations"])
    count = len(bundle["inputs"])
    _generation(bundle["generation"], count)
    rewards = bundle["rewards"]
    if (
        not isinstance(rewards, list)
        or len(rewards) != count
        or any(
            not isinstance(row, list)
            or len(row) != 1
            or type(row[0]) not in (int, float)
            or not math.isfinite(row[0])
            for row in rewards
        )
    ):
        raise ValueError("TRL replay requires one finite reward per sample")
    if not isinstance(bundle["observed"], dict) or set(bundle["observed"]) != set(
        OBSERVATIONS
    ):
        raise ValueError("invalid native observations")
    evidence = Evidence.from_dict(bundle["evidence"])
    rebuilt = _evidence(
        bundle["inputs"],
        rewards,
        bundle["scalar_advantages"],
        bundle["observed"]["completion_mask"],
        configuration,
        evidence.provenance,
        [row.sample_id for row in evidence.samples],
    )
    if rebuilt != evidence or not evidence.complete_groups:
        raise ValueError("TRL replay observations and Evidence disagree")
    if not isinstance(bundle["tokenizer"], dict) or not isinstance(
        bundle["runtime"], dict
    ):
        raise TypeError("missing tokenizer/runtime identity")
    if set(bundle["tokenizer"]) != {"sha256", "pad_token_id", "eos_token_ids"} or (
        type(bundle["tokenizer"]["pad_token_id"]) is not int
        or not 0 <= bundle["tokenizer"]["pad_token_id"] < 2**63
    ):
        raise ValueError("invalid tokenizer padding identity")
    return evidence


def capture_trl(trainer, inputs):
    """Call one native method and freeze its inputs. May generate text; not a training hook."""
    _native(trainer)
    provenance = implementation_provenance()
    configuration = _configuration(trainer)
    inputs = _copy(inputs)
    _inputs(inputs, configuration["num_generations"])
    frozen = {}
    generate, calculate = trainer._generate, trainer._calculate_rewards

    def generated(*args, **kwargs):
        if "generation" in frozen:
            raise ValueError("reentrant TRL generation is unsupported")
        value = generate(*args, **kwargs)
        frozen["generation"] = _copy(value)
        _generation(frozen["generation"], len(inputs))
        return value

    def rewarded(*args, **kwargs):
        if "rewards" in frozen:
            raise ValueError("repeated TRL reward provider is unsupported")
        value = calculate(*args, **kwargs)
        if str(value.device) != "cpu" or str(value.dtype) != "torch.float32":
            raise ValueError("TRL replay requires CPU float32 rewards")
        frozen["rewards"] = value.detach().tolist()
        return value

    with _providers(trainer, _generate=generated, _calculate_rewards=rewarded):
        output = _output(trainer._generate_and_score_completions(copy.deepcopy(inputs)))
    ids = [f"{uuid.uuid4()}:{i}" for i in range(len(inputs))]
    provenance.update(capture_mode="explicit_native_replay")
    evidence = _evidence(
        inputs,
        frozen["rewards"],
        output["advantages"],
        output["completion_mask"],
        configuration,
        provenance,
        ids,
    )
    bundle = _copy(
        {
            "configuration": configuration,
            "tokenizer": _tokenizer(trainer),
            "runtime": _runtime(),
            "inputs": inputs,
            **frozen,
            "scalar_advantages": output.pop("advantages"),
            "observed": output,
            "evidence": evidence.to_dict(),
        }
    )
    _validate(bundle)
    return bundle


def write_trl(bundle, directory):
    _validate(bundle)
    payload = _canonical(bundle)
    envelope = _canonical(
        {
            "format": FORMAT,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bundle": bundle,
        }
    )
    if len(envelope) > MAX_CAPSULE_BYTES:
        raise ValueError("TRL replay bundle exceeds byte budget")
    with writer_directory(directory):
        return write_payload(directory, str(uuid.uuid4()), envelope)


def read_trl(path):
    with Path(path).open("rb") as stream:
        payload = stream.read(MAX_CAPSULE_BYTES + 1)
    if len(payload) > MAX_CAPSULE_BYTES:
        raise ValueError("TRL replay bundle exceeds byte budget")
    value = json.loads(payload, object_pairs_hook=_unique_object)
    if (
        not isinstance(value, dict)
        or set(value) != {"format", "sha256", "bundle"}
        or value["format"] != FORMAT
    ):
        raise ValueError("invalid TRL replay envelope")
    digest = hashlib.sha256(_canonical(value["bundle"])).hexdigest()
    if not isinstance(value["sha256"], str) or not hmac.compare_digest(
        digest, value["sha256"]
    ):
        raise ValueError("TRL replay digest mismatch")
    _validate(value["bundle"])
    return value["bundle"]


def run_trl(bundle, trainer, *, allow_revision_change=False):
    """Execute the full native method with frozen providers on a caller-owned Trainer."""
    original = _validate(bundle)
    _native(trainer)
    actual = implementation_provenance(allow_revision_change=allow_revision_change)
    if (
        _configuration(trainer) != bundle["configuration"]
        or _tokenizer(trainer) != bundle["tokenizer"]
    ):
        raise ValueError("TRL replay configuration or tokenizer differs from capture")
    if _runtime() != bundle["runtime"] or any(
        actual[key] != original.provenance.get(key)
        for key in ("python_version", "torch_version", "numpy_version")
    ):
        raise ValueError("TRL replay runtime differs from capture")
    if not allow_revision_change and any(
        actual[key] != original.provenance.get(key)
        for key in ("algorithm_sha256", "groupwise_sha256")
    ):
        raise ValueError("TRL replay source differs from capture")
    import torch

    calls = {"generation": 0, "rewards": 0}

    def generated(prompts):
        calls["generation"] += 1
        if prompts != [row["prompt"] for row in bundle["inputs"]]:
            raise ValueError("native method changed the frozen prompt boundary")
        return tuple(copy.deepcopy(bundle["generation"]))

    def rewarded(inputs, prompts, completions, completion_ids_list):
        calls["rewards"] += 1
        if (
            inputs != bundle["inputs"]
            or prompts != [row["prompt"] for row in inputs]
            or completions != bundle["generation"][3]
            or completion_ids_list != bundle["generation"][1]
        ):
            raise ValueError("native method changed the frozen reward boundary")
        return torch.tensor(bundle["rewards"], dtype=torch.float32, device="cpu")

    def no_forward(*args, **kwargs):
        raise ValueError(
            "TRL replay attempted model/logprob computation outside its frozen boundary"
        )

    with (
        _providers(
            trainer,
            _generate=generated,
            _calculate_rewards=rewarded,
            _get_per_token_logps_and_entropies=no_forward,
        ),
        _providers(trainer.model, forward=no_forward),
    ):
        output = _output(
            trainer._generate_and_score_completions(copy.deepcopy(bundle["inputs"]))
        )
    if calls != {"generation": 1, "rewards": 1}:
        raise ValueError(
            "native method did not consume each frozen provider exactly once"
        )
    replayed = _evidence(
        bundle["inputs"],
        bundle["rewards"],
        output["advantages"],
        output["completion_mask"],
        bundle["configuration"],
        actual,
        [row.sample_id for row in original.samples],
    )
    replayed = replace(
        replayed, provenance={**actual, "capture_mode": "native_boundary_replay"}
    )
    # The scalar view remains observable even if a candidate returns all-zero masks.
    scalar_view = replace(
        replayed,
        samples=tuple(
            replace(row, mask=(1,), advantage=(scalar,))
            for row, scalar in zip(replayed.samples, output["advantages"], strict=True)
        ),
    )
    finding = check(replayed)
    if finding.verdict == "PASS":
        finding = check(scalar_view)
    layout_differences = [
        name
        for name, expected in _expected_layout(bundle).items()
        if output[name] != expected
    ]
    if layout_differences:
        finding = Finding(
            "CONTRACT_VIOLATION",
            "trl_frozen_generation_layout",
            tuple(row.sample_id for row in replayed.samples),
            checked_samples=len(replayed.samples),
            total_samples=len(replayed.samples),
        )
    return {
        "runner": "trl",
        "finding": finding.to_dict(),
        "comparison": diff(original, replayed),
        "scalar_advantages": output["advantages"],
        "source": actual,
        "native_observation_equal": output
        == {**bundle["observed"], "advantages": bundle["scalar_advantages"]},
        "provider_calls": calls,
        "layout_differences": layout_differences,
    }


def _factory_parts(reference):
    if not isinstance(reference, str):
        raise TypeError("TRL replay requires --trainer-factory module:function")
    module, separator, name = reference.partition(":")
    if (
        not separator
        or not name.isidentifier()
        or not all(part.isidentifier() for part in module.split("."))
    ):
        raise ValueError("trainer factory must be module:function")
    return module, name


def trainer_from_factory(reference):
    """Import only a factory explicitly supplied by the caller, never by a bundle."""
    module, name = _factory_parts(reference)
    factory = getattr(importlib.import_module(module), name, None)
    if not callable(factory):
        raise TypeError("trainer factory is not callable")
    return factory()


def export_trl_test(bundle_path, directory, *, trainer_factory):
    read_trl(bundle_path)
    _factory_parts(trainer_factory)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(bundle_path, directory / "trl-bundle.json")
    path = directory / "test_trl_replay.py"
    path.write_text(
        '"""Frozen-provider TRL regression; the explicit local factory must be importable."""\n'
        "from pathlib import Path\n"
        "from rlreplay.trl_replay import read_trl, run_trl, trainer_from_factory\n\n"
        "def test_native_trl_contract():\n"
        '    bundle = read_trl(Path(__file__).with_name("trl-bundle.json"))\n'
        f"    trainer = trainer_from_factory({trainer_factory!r})\n"
        "    result = run_trl(bundle, trainer, allow_revision_change=True)\n"
        '    assert result["finding"]["verdict"] == "PASS", result\n',
        encoding="utf-8",
    )
    return path
