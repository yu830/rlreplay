"""Outcome GRPO reference and qualified numerical comparisons.

The binary32 reference only explains one CPU sequential reduction model. Matching
it qualifies a discrepancy as inconclusive, rather than proving every backend correct.
"""

import math
import struct
from collections import defaultdict

from rlreplay.core import PROFILE_ID, Evidence, Finding

ATOL = 1e-6
RTOL = 1e-5


def _f32(value: float) -> float:
    rounded = struct.unpack("f", struct.pack("f", value))[0]
    if not math.isfinite(rounded):
        raise OverflowError("binary32 reference became nonfinite")
    return rounded


def reference(
    evidence: Evidence, *, binary32: bool = False
) -> dict[str, tuple[float, ...]]:
    """Compute centered sample statistics independently of the training framework."""
    groups: dict[str, list[float]] = defaultdict(list)
    for row in evidence.samples:
        groups[row.group_id].append(row.reward)
    stats = {}
    for group, values in groups.items():
        if len(values) == 1:
            stats[group] = (0.0, 1.0)
        elif binary32:
            total = 0.0
            for value in values:
                total = _f32(total + value)
            mean = _f32(total / len(values))
            variance_sum = 0.0
            for value in values:
                deviation = _f32(value - mean)
                variance_sum = _f32(variance_sum + _f32(deviation * deviation))
            stats[group] = (
                mean,
                _f32(math.sqrt(_f32(variance_sum / (len(values) - 1)))),
            )
        else:
            mean = math.fsum(values) / len(values)
            variance = math.fsum((value - mean) ** 2 for value in values) / (
                len(values) - 1
            )
            stats[group] = (mean, math.sqrt(variance))
    output = {}
    for row in evidence.samples:
        mean, std = stats[row.group_id]
        if binary32:
            advantage = _f32(row.reward - mean)
            if evidence.profile.normalize_std:
                advantage = _f32(advantage / _f32(std + _f32(evidence.profile.epsilon)))
        else:
            advantage = row.reward - mean
            if evidence.profile.normalize_std:
                advantage /= std + evidence.profile.epsilon
        output[row.sample_id] = tuple(advantage * mask for mask in row.mask)
    return output


def check(evidence: Evidence) -> Finding:
    """Return evidence-scoped verdicts, including explicit missing-input states."""
    rows = evidence.samples
    total = len(rows)

    def finding(verdict, reason, ids=(), error=None, checked=0):
        return Finding(verdict, reason, tuple(ids), error, checked, total)

    profile = evidence.profile
    if (
        profile.profile_id != PROFILE_ID
        or profile.ddof != 1
        or profile.epsilon not in (1e-6, 1e-4)
        or profile.singleton != "mean0_std1"
    ):
        return finding("INCONCLUSIVE", "unsupported_profile")
    if (
        evidence.provenance.get("device") != "cpu"
        or evidence.provenance.get("dtype") != "float32"
    ):
        return finding("INCONCLUSIVE", "unsupported_numeric_execution")
    if not rows or not evidence.complete_groups or not evidence.membership:
        return finding("INCONCLUSIVE", "missing_membership_or_incomplete_group")
    required = {
        "framework",
        "framework_revision",
        "algorithm_sha256",
        "phase",
        "receipt_origin",
    }
    if not required <= evidence.provenance.keys():
        return finding("INCONCLUSIVE", "missing_provenance")
    members = dict(evidence.membership)
    ids = [row.sample_id for row in rows]
    if (
        len(members) != len(evidence.membership)
        or len(ids) != len(set(ids))
        or set(ids) != set(members)
    ):
        return finding("CONTRACT_VIOLATION", "sample_identity", ids)
    for row in rows:
        if members[row.sample_id] != row.group_id:
            return finding("CONTRACT_VIOLATION", "group_identity", (row.sample_id,))
        if (
            not row.mask
            or len(row.mask) != len(row.advantage)
            or any(mask not in (0, 1) for mask in row.mask)
        ):
            return finding("CONTRACT_VIOLATION", "shape_or_mask", (row.sample_id,))
        if any(
            type(x) is int and abs(x) > 3.4028234663852886e38
            for x in (row.reward, *row.advantage)
        ):
            return finding(
                "CONTRACT_VIOLATION", "outside_float32_range", (row.sample_id,)
            )
        if not all(math.isfinite(x) for x in (row.reward, *row.advantage)):
            return finding("CONTRACT_VIOLATION", "nonfinite", (row.sample_id,))
        if any(abs(x) > 3.4028234663852886e38 for x in (row.reward, *row.advantage)):
            return finding(
                "CONTRACT_VIOLATION", "outside_float32_range", (row.sample_id,)
            )
        if any(
            mask == 0 and value != 0
            for mask, value in zip(row.mask, row.advantage, strict=True)
        ):
            return finding("CONTRACT_VIOLATION", "padding_advantage", (row.sample_id,))
    expected = reference(evidence)
    differences = {}
    for row in rows:
        for actual, target in zip(row.advantage, expected[row.sample_id], strict=True):
            error = abs(actual - target)
            if error > ATOL + RTOL * abs(target):
                differences[row.sample_id] = max(
                    error, differences.get(row.sample_id, 0.0)
                )
    if not differences:
        return finding("PASS", "within_reference_tolerance", checked=total)
    try:
        rounded = reference(evidence, binary32=True)
    except (OverflowError, ZeroDivisionError):
        return finding(
            "INCONCLUSIVE",
            "binary32_reference_out_of_range",
            differences,
            max(differences.values()),
            total,
        )
    explained = all(
        abs(actual - target) <= ATOL + RTOL * abs(target)
        for row in rows
        for actual, target in zip(row.advantage, rounded[row.sample_id], strict=True)
    )
    return finding(
        "INCONCLUSIVE" if explained else "SUSPECTED_MISMATCH",
        "binary32_roundoff_explains_difference"
        if explained
        else "independent_reference_mismatch",
        differences,
        max(differences.values()),
        total,
    )
