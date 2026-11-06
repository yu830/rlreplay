"""Optional lossless safetensors payloads with a bound JSON evidence manifest."""

import hashlib
import hmac
import io
import json
import zipfile
from dataclasses import asdict

import numpy as np
from safetensors import SafetensorError
from safetensors.numpy import load as safe_load
from safetensors.numpy import save as safe_save

from rlreplay import artifacts
from rlreplay.core import Evidence

LIMIT = artifacts.MAX_CAPSULE_BYTES
FORMAT = "rlreplay-capsule/safetensors/1"


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def digest(value):
    return hashlib.sha256(value).hexdigest()


def _lists(value):
    if isinstance(value, tuple):
        return [_lists(item) for item in value]
    if isinstance(value, list):
        return [_lists(item) for item in value]
    if isinstance(value, dict):
        return {key: _lists(item) for key, item in value.items()}
    return value


def pack(evidence):
    # Same Evidence field/type validation as JSON, without a numeric JSON codec pass.
    evidence = Evidence.from_dict(_lists(evidence.to_dict()))
    arrays = {}
    layout = {}
    rows = evidence.samples
    fields = {
        "reward": [[r.reward] for r in rows],
        "mask": [r.mask for r in rows],
        "advantage": [r.advantage for r in rows],
    }
    for field, values in fields.items():
        lengths = [len(row) for row in values]
        dense = len(set(lengths)) <= 1
        groups = (
            [[x for row in values for x in row]]
            if dense
            else [list(row) for row in values]
        )
        keys = []
        specials = {}
        for i, numbers in enumerate(groups):
            key = field if dense else f"{field}_{i}"
            keys.append(key)
            overrides = {
                str(j): {"kind": type(x).__name__, "value": str(x)}
                for j, x in enumerate(numbers)
                if type(x) in (int, bool)
            }
            # Exact integers (including >2**53) live in metadata, not a rounded float.
            raw = np.asarray(
                [0.0 if type(x) in (int, bool) else x for x in numbers], dtype="<f8"
            )
            if dense:
                raw = raw.reshape(len(rows), lengths[0] if lengths else 0)
            arrays[key] = np.ascontiguousarray(raw)
            if overrides:
                specials[key] = overrides
        layout[field] = {
            "dense": dense,
            "keys": keys,
            "lengths": lengths,
            "specials": specials,
        }
    meta = {
        "schema_version": evidence.schema_version,
        "profile": asdict(evidence.profile),
        "samples": [{"sample_id": r.sample_id, "group_id": r.group_id} for r in rows],
        "membership": [list(pair) for pair in evidence.membership],
        "provenance": dict(evidence.provenance),
        "complete_groups": evidence.complete_groups,
        "layout": layout,
        "arrays": {
            key: {"dtype": a.dtype.str, "shape": list(a.shape)}
            for key, a in arrays.items()
        },
    }
    return arrays, meta


def _read_meta(raw):
    envelope = json.loads(raw, object_pairs_hook=artifacts._unique_object)
    if (
        not isinstance(envelope, dict)
        or set(envelope) != {"format", "sha256", "metadata"}
        or envelope["format"] != FORMAT
    ):
        raise ValueError("invalid binary envelope")
    if not isinstance(envelope["sha256"], str) or not hmac.compare_digest(
        envelope["sha256"], digest(canonical(envelope["metadata"]))
    ):
        raise ValueError("metadata digest mismatch")
    meta = envelope["metadata"]
    expected = {
        "schema_version",
        "profile",
        "samples",
        "membership",
        "provenance",
        "complete_groups",
        "layout",
        "arrays",
    }
    expected.add("payload_sha256")
    if not isinstance(meta, dict) or set(meta) != expected:
        raise ValueError("invalid metadata fields")
    return meta


def _archive(payload):
    if len(payload) > LIMIT:
        raise ValueError("capsule exceeds byte budget")
    archive = zipfile.ZipFile(io.BytesIO(payload))
    info = archive.infolist()
    if len({i.filename for i in info}) != len(info):
        archive.close()
        raise ValueError("duplicate archive member")
    if sum(i.file_size for i in info) > LIMIT or any(
        i.compress_type != zipfile.ZIP_STORED or i.flag_bits & ~0x808 for i in info
    ):
        archive.close()
        raise ValueError("unsupported archive or expanded byte budget")
    return archive


def _unpack(meta, arrays):
    if set(arrays) != set(meta["arrays"]):
        raise ValueError("unexpected or missing arrays")
    for key, a in arrays.items():
        desc = meta["arrays"][key]
        if (
            set(desc) != {"dtype", "shape"}
            or a.dtype.str != desc["dtype"]
            or list(a.shape) != desc["shape"]
            or a.dtype.str != "<f8"
        ):
            raise ValueError("array dtype/shape mismatch")
    n = len(meta["samples"])
    components = {}
    used = []
    if set(meta["layout"]) != {"reward", "mask", "advantage"}:
        raise ValueError("invalid layout fields")
    for field, layout in meta["layout"].items():
        if (
            set(layout) != {"dense", "keys", "lengths", "specials"}
            or type(layout["dense"]) is not bool
        ):
            raise ValueError("invalid layout")
        lengths, keys, specials = layout["lengths"], layout["keys"], layout["specials"]
        if len(lengths) != n or any(type(x) is not int or x < 0 for x in lengths):
            raise ValueError("invalid row lengths")
        if any(key not in keys for key in specials) or len(keys) != (
            1 if layout["dense"] else n
        ):
            raise ValueError("invalid layout references")
        used.extend(keys)
        parts = []
        for key in keys:
            values = arrays[key].reshape(-1).tolist()
            for position, item in specials.get(key, {}).items():
                j = int(position)
                if (
                    str(j) != position
                    or j < 0
                    or j >= len(values)
                    or set(item) != {"kind", "value"}
                ):
                    raise ValueError("invalid numeric override")
                if item["kind"] == "int":
                    values[j] = int(item["value"])
                elif item["kind"] == "bool" and item["value"] in ("True", "False"):
                    values[j] = item["value"] == "True"
                else:
                    raise ValueError("invalid numeric kind")
            parts.append(values)
        if layout["dense"]:
            if len(set(lengths)) > 1 or list(arrays[keys[0]].shape) != [
                n,
                lengths[0] if lengths else 0,
            ]:
                raise ValueError("dense layout shape mismatch")
            width = lengths[0] if lengths else 0
            parts = [parts[0][i * width : (i + 1) * width] for i in range(n)]
        elif any(
            list(arrays[key].shape) != [length]
            for key, length in zip(keys, lengths, strict=True)
        ):
            raise ValueError("ragged layout shape mismatch")
        if [len(x) for x in parts] != lengths or (
            field == "reward" and lengths != [1] * n
        ):
            raise ValueError("numeric field length mismatch")
        components[field] = parts
    if len(used) != len(set(used)) or set(used) != set(arrays):
        raise ValueError("duplicated or unused numeric array")
    value = {
        key: meta[key]
        for key in (
            "schema_version",
            "profile",
            "membership",
            "provenance",
            "complete_groups",
        )
    }
    value["samples"] = []
    for i, row in enumerate(meta["samples"]):
        if set(row) != {"sample_id", "group_id"}:
            raise ValueError("invalid sample metadata")
        value["samples"].append(
            {
                **row,
                "reward": components["reward"][i][0],
                "mask": components["mask"][i],
                "advantage": components["advantage"][i],
            }
        )
    # The existing Core number marker is used for NaN/Inf; no data are discarded.
    from rlreplay.core import _encoded_number

    for row in value["samples"]:
        row["reward"] = _encoded_number(row["reward"])
        for name in ("mask", "advantage"):
            row[name] = [_encoded_number(x) for x in row[name]]
    return Evidence.from_dict(value)


def encode(evidence: Evidence) -> bytes:
    arrays, metadata = pack(evidence)
    payload = safe_save(arrays)
    metadata["payload_sha256"] = digest(payload)
    envelope = {
        "format": FORMAT,
        "sha256": digest(canonical(metadata)),
        "metadata": metadata,
    }
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("metadata.json", canonical(envelope))
        archive.writestr("tensors.safetensors", payload)
    result = stream.getvalue()
    if len(result) > LIMIT:
        raise ValueError("capsule exceeds byte budget")
    return result


def decode(payload: bytes) -> Evidence:
    try:
        with _archive(payload) as archive:
            if set(archive.namelist()) != {"metadata.json", "tensors.safetensors"}:
                raise ValueError("unexpected archive members")
            metadata = _read_meta(archive.read("metadata.json"))
            raw = archive.read("tensors.safetensors")
            expected = metadata["payload_sha256"]
            if not isinstance(expected, str) or not hmac.compare_digest(
                expected, digest(raw)
            ):
                raise ValueError("payload digest mismatch")
            arrays = safe_load(raw)
        return _unpack(metadata, arrays)
    except (
        zipfile.BadZipFile,
        SafetensorError,
        KeyError,
        IndexError,
        AttributeError,
    ) as error:
        raise ValueError("invalid safetensors capsule") from error
