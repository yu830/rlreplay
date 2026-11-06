"""Optional codec fidelity, archive boundaries and real writer integration."""

import io
import json
import math
import zipfile
from dataclasses import replace

import pytest

pytest.importorskip("numpy")
pytest.importorskip("safetensors")

from rlreplay.artifacts import decode, encode, read, write
from rlreplay.capture import Recorder
from rlreplay.contracts.grpo import check
from rlreplay.core import Sample
from rlreplay.replay import export_test


def canonical(evidence):
    return json.dumps(evidence.to_dict(), sort_keys=True, allow_nan=False)


def repack(payload, *, metadata=None, tensor_bytes=None, extra=None):
    with zipfile.ZipFile(io.BytesIO(payload)) as source:
        items = {name: source.read(name) for name in source.namelist()}
    if metadata is not None:
        items["metadata.json"] = metadata
    if tensor_bytes is not None:
        items["tensors.safetensors"] = tensor_bytes
    if extra is not None:
        items[extra] = b"unexpected"
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        for name, value in items.items():
            archive.writestr(name, value)
    return output.getvalue()


def test_both_formats_preserve_existing_contract(evidence, tmp_path):
    for codec in ("json", "safetensors"):
        path = write(evidence, tmp_path, codec=codec)
        restored = read(path)
        assert canonical(restored) == canonical(evidence)
        assert check(restored) == check(evidence)
    assert len(list(tmp_path.glob("*.json"))) == 1
    assert len(list(tmp_path.glob("*.rlrp"))) == 1


def test_special_numbers_and_ragged_faults_survive(evidence):
    samples = (
        Sample("a0", "a", 2**53 + 1, (True, -0.0, 0.5), (float("nan"), float("inf"))),
        Sample("a1", "a", -float("inf"), (), (-0.0,)),
    )
    mutated = replace(evidence, samples=samples, complete_groups=False)
    restored = decode(encode(mutated, codec="safetensors"))
    assert canonical(restored) == canonical(mutated)
    assert type(restored.samples[0].reward) is int
    assert type(restored.samples[0].mask[0]) is bool
    assert math.copysign(1, restored.samples[1].advantage[0]) == -1
    assert check(restored).verdict == "INCONCLUSIVE"


def test_binary_tamper_and_container_boundaries(evidence):
    payload = encode(evidence, codec="safetensors")
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        metadata = json.loads(archive.read("metadata.json"))
        tensors = archive.read("tensors.safetensors")
    metadata["metadata"]["provenance"]["phase"] = "changed"
    with pytest.raises(ValueError, match="metadata digest"):
        decode(repack(payload, metadata=json.dumps(metadata).encode()))
    with pytest.raises(ValueError, match="payload digest"):
        decode(repack(payload, tensor_bytes=tensors[:-1] + bytes([tensors[-1] ^ 1])))
    with pytest.raises(ValueError, match="unexpected archive"):
        decode(repack(payload, extra="../outside"))
    with pytest.raises(ValueError, match="invalid safetensors capsule"):
        decode(payload[:-20])


def test_binary_writer_uses_same_completion_contract(evidence, tmp_path):
    recorder = Recorder(tmp_path, codec="safetensors")
    recorder.submit(evidence)
    events = recorder.close()
    complete = [event for event in events if event["state"] == "COMPLETE"]
    assert len(complete) == 1
    assert complete[0]["path"].endswith(".rlrp")
    assert canonical(read(complete[0]["path"])) == canonical(evidence)


def test_binary_regression_export(evidence, tmp_path):
    path = write(evidence, tmp_path / "source", codec="safetensors")
    exported = export_test(path, tmp_path / "regression")
    assert canonical(read(exported.with_name("capsule.rlrp"))) == canonical(evidence)


@pytest.mark.parametrize("flag", [1, 0x20, 0x40])
def test_unsupported_zip_features_are_rejected(evidence, flag):
    payload = bytearray(encode(evidence, codec="safetensors"))
    central_directory = payload.index(b"PK\x01\x02")
    payload[central_directory + 8] |= flag
    with pytest.raises(ValueError, match="unsupported archive"):
        decode(bytes(payload))
