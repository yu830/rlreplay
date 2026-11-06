"""Bounded versioned capsules with content integrity and atomic completion."""

import hashlib
import hmac
import json
import os
import uuid
from pathlib import Path

from rlreplay._directory import writer_directory
from rlreplay.core import Evidence

MAX_CAPSULE_BYTES = 4 * 1024 * 1024
CAPSULE_SUFFIXES = {"json": ".json", "safetensors": ".rlrp"}


def prepare_codec(codec: str) -> str:
    if codec not in CAPSULE_SUFFIXES:
        raise ValueError("unsupported capsule codec")
    if codec == "safetensors":
        from rlreplay import _safetensors

        return _safetensors.FORMAT
    return "rlreplay-capsule/1"


def _canonical(value) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def encode(evidence: Evidence, *, codec="json") -> bytes:
    prepare_codec(codec)
    if codec == "safetensors":
        from rlreplay import _safetensors

        return _safetensors.encode(evidence)
    value = evidence.to_dict()
    # Round-trip types at the external boundary, including trusted callers' instances.
    value = json.loads(_canonical(value))
    Evidence.from_dict(value)
    payload = _canonical(value)
    result = _canonical(
        {
            "format": "rlreplay-capsule/1",
            "sha256": hashlib.sha256(payload).hexdigest(),
            "evidence": value,
        }
    )
    if len(result) > MAX_CAPSULE_BYTES:
        raise ValueError("capsule exceeds byte budget")
    return result


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def decode(payload: bytes) -> Evidence:
    if len(payload) > MAX_CAPSULE_BYTES:
        raise ValueError("capsule exceeds byte budget")
    if payload.startswith(b"PK"):
        from rlreplay import _safetensors

        return _safetensors.decode(payload)
    value = json.loads(payload, object_pairs_hook=_unique_object)
    if (
        not isinstance(value, dict)
        or set(value) != {"format", "sha256", "evidence"}
        or value["format"] != "rlreplay-capsule/1"
        or not isinstance(value["sha256"], str)
    ):
        raise ValueError("invalid capsule envelope")
    digest = hashlib.sha256(_canonical(value["evidence"])).hexdigest()
    if not hmac.compare_digest(digest, value["sha256"]):
        raise ValueError("capsule digest mismatch")
    return Evidence.from_dict(value["evidence"])


def read(path: str | Path) -> Evidence:
    with Path(path).open("rb") as stream:
        return decode(stream.read(MAX_CAPSULE_BYTES + 1))


def write_payload(
    directory: str | Path, name: str, payload: bytes, *, suffix=".json"
) -> Path:
    """Publish a complete capsule. The writer owns a dedicated output directory."""
    uuid.UUID(name)
    if suffix not in CAPSULE_SUFFIXES.values():
        raise ValueError("invalid capsule suffix")
    directory = Path(directory)
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    target = directory / f"{name}{suffix}"
    temporary = directory / f"{name}.partial"
    if os.path.lexists(target):
        raise FileExistsError("capture ID is already published")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def write(evidence: Evidence, directory: str | Path, *, codec="json") -> Path:
    with writer_directory(directory):
        payload = encode(evidence, codec=codec)
        return write_payload(
            directory, str(uuid.uuid4()), payload, suffix=CAPSULE_SUFFIXES[codec]
        )
