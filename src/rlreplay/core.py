"""Versioned, framework-independent evidence for outcome advantage checks."""

import math
from dataclasses import asdict, dataclass
from typing import Any

SCHEMA_VERSION = 1
PROFILE_ID = "grpo-outcome/sample-std-epsilon-outside/v1"
PROVENANCE_FIELDS = frozenset(
    {
        "framework",
        "framework_revision",
        "algorithm_sha256",
        "adapter_version",
        "phase",
        "run_id",
        "step",
        "device",
        "dtype",
        "receipt_origin",
        "entrypoint",
        "entrypoint_sha256",
        "groupwise_sha256",
        "python_version",
        "torch_version",
        "numpy_version",
        "capture_mode",
        "sample_every",
        "max_groups",
        "worker_id",
        "rank",
        "world_size",
    }
)


def _encoded_number(value):
    if type(value) is int and abs(value) > 3.4028234663852886e38:
        raise ValueError("integer exceeds supported float32 range")
    if math.isnan(value):
        return {"nonfinite": "nan"}
    if math.isinf(value):
        return {"nonfinite": "+inf" if value > 0 else "-inf"}
    return value


def _decoded_number(value):
    if type(value) is int and abs(value) > 3.4028234663852886e38:
        raise ValueError("integer exceeds supported float32 range")
    if type(value) in (int, float, bool):
        return value
    if isinstance(value, dict) and set(value) == {"nonfinite"}:
        variants = {"nan": float("nan"), "+inf": float("inf"), "-inf": -float("inf")}
        if isinstance(value["nonfinite"], str) and value["nonfinite"] in variants:
            return variants[value["nonfinite"]]
    raise ValueError("invalid numeric value")


@dataclass(frozen=True)
class Profile:
    """Supported semantic choices are explicit even when a check is unsupported."""

    implementation: str = "grpo_vectorized"
    normalize_std: bool = True
    profile_id: str = PROFILE_ID
    epsilon: float = 1e-6
    ddof: int = 1
    singleton: str = "mean0_std1"


@dataclass(frozen=True)
class Sample:
    sample_id: str
    group_id: str
    reward: float
    mask: tuple[int | float, ...]
    advantage: tuple[float, ...]


@dataclass(frozen=True)
class Evidence:
    profile: Profile
    samples: tuple[Sample, ...]
    membership: tuple[tuple[str, str], ...]
    provenance: dict[str, str]
    complete_groups: bool = True
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        for row in result["samples"]:
            row["reward"] = _encoded_number(row["reward"])
            row["mask"] = [_encoded_number(value) for value in row["mask"]]
            row["advantage"] = [_encoded_number(value) for value in row["advantage"]]
        return result

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Evidence":
        """Validate the serialized boundary, leaving semantic faults to the oracle."""
        if (
            not isinstance(value, dict)
            or type(value.get("schema_version")) is not int
            or value["schema_version"] != SCHEMA_VERSION
        ):
            raise ValueError("unsupported evidence schema")
        if set(value) != {
            "schema_version",
            "profile",
            "samples",
            "membership",
            "provenance",
            "complete_groups",
        }:
            raise ValueError("unexpected or missing evidence fields")
        profile_data = value["profile"]
        if not isinstance(profile_data, dict) or set(profile_data) != set(
            Profile.__dataclass_fields__
        ):
            raise ValueError("invalid profile fields")
        profile = Profile(**profile_data)
        if (
            type(profile.normalize_std) is not bool
            or type(profile.ddof) is not int
            or not isinstance(profile.epsilon, (int, float))
            or any(
                not isinstance(x, str)
                for x in (profile.profile_id, profile.implementation, profile.singleton)
            )
        ):
            raise ValueError("invalid profile types")
        if type(value["complete_groups"]) is not bool:
            raise ValueError("complete_groups must be boolean")
        provenance = value["provenance"]
        if (
            not isinstance(provenance, dict)
            or not set(provenance) <= PROVENANCE_FIELDS
            or any(not isinstance(v, str) for v in provenance.values())
        ):
            raise ValueError("invalid provenance fields")
        samples = []
        if not isinstance(value["samples"], list) or not isinstance(
            value["membership"], list
        ):
            raise TypeError("samples and membership must be arrays")
        for row in value["samples"]:
            if not isinstance(row, dict) or set(row) != set(
                Sample.__dataclass_fields__
            ):
                raise ValueError("invalid sample fields")
            if any(
                not isinstance(row[k], str) or not row[k]
                for k in ("sample_id", "group_id")
            ):
                raise ValueError("sample and group IDs must be nonempty strings")
            if not isinstance(row["mask"], list) or not isinstance(
                row["advantage"], list
            ):
                raise TypeError("mask and advantage must be arrays")
            reward = _decoded_number(row["reward"])
            masks = tuple(_decoded_number(value) for value in row["mask"])
            advantages = tuple(_decoded_number(value) for value in row["advantage"])
            if type(reward) is bool or any(type(value) is bool for value in advantages):
                raise ValueError("reward and advantage must be numbers")
            samples.append(
                Sample(
                    row["sample_id"],
                    row["group_id"],
                    reward,
                    masks,
                    advantages,
                )
            )
        membership = []
        for pair in value["membership"]:
            if (
                not isinstance(pair, list)
                or len(pair) != 2
                or any(not isinstance(v, str) or not v for v in pair)
            ):
                raise ValueError("invalid membership entry")
            membership.append(tuple(pair))
        return cls(
            profile,
            tuple(samples),
            tuple(membership),
            dict(provenance),
            value["complete_groups"],
        )


@dataclass(frozen=True)
class Finding:
    verdict: str
    reason: str
    sample_ids: tuple[str, ...] = ()
    max_abs_error: float | None = None
    checked_samples: int = 0
    total_samples: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
