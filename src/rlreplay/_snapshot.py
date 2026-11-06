"""Own the schema's primitive values before handing work to an IPC feeder."""

from rlreplay.core import Evidence, Profile, Sample

DEFAULT_SNAPSHOT_BUDGET = 16 * 1024 * 1024
FLOAT32_MAX = 3.4028234663852886e38


class SnapshotBudgetExceeded(ValueError):
    """The bounded snapshot accounting limit was exceeded."""


def freeze(evidence: Evidence, budget: int) -> tuple[Evidence, int]:
    """Freeze lists/dicts, rejecting tensors and arbitrary objects in the IPC path.

    The conservative charge covers primitive/container storage and serialization
    allowance. It is an admission limit, not an assertion about total process RSS.
    The worker retains the format's complete schema/semantic validation.
    """
    if type(evidence) is not Evidence or type(evidence.profile) is not Profile:
        raise TypeError("capture requires Evidence and Profile instances")
    charged = 1024

    def charge(size):
        nonlocal charged
        charged += size
        if charged > budget:
            raise SnapshotBudgetExceeded("snapshot exceeds input budget")

    def text(value):
        if not isinstance(value, str):
            raise TypeError("snapshot metadata must contain strings")
        charge(4 * str.__len__(value) + 80)
        return str.__str__(value)

    def number(value, *, boolean=False):
        if type(value) is float:
            return value
        if type(value) is bool:
            if boolean:
                return value
            raise TypeError("snapshot numeric fields require Python numbers")
        if isinstance(value, int):
            value = int.__int__(value)
            if abs(value) > FLOAT32_MAX:
                raise ValueError("integer exceeds supported float32 range")
            return value
        if isinstance(value, float):
            return float.__float__(value)
        raise TypeError("snapshot numeric fields require Python numbers")

    def vector(values, *, boolean=False):
        if type(values) not in (tuple, list):
            raise TypeError("snapshot vectors must be lists or tuples")
        charge(64 * len(values) + 64)
        return tuple(number(value, boolean=boolean) for value in values)

    profile = evidence.profile
    if type(profile.normalize_std) is not bool or type(profile.ddof) is not int:
        raise TypeError("invalid snapshot profile types")
    owned_profile = Profile(
        text(profile.implementation),
        profile.normalize_std,
        text(profile.profile_id),
        number(profile.epsilon),
        number(profile.ddof),
        text(profile.singleton),
    )
    if (
        type(evidence.complete_groups) is not bool
        or type(evidence.schema_version) is not int
    ):
        raise TypeError("invalid snapshot state types")
    if type(evidence.samples) not in (tuple, list) or type(evidence.membership) not in (
        tuple,
        list,
    ):
        raise TypeError("snapshot rows and membership must be sequences")
    charge(512 * len(evidence.samples) + 128 * len(evidence.membership))
    samples = []
    for row in evidence.samples:
        if type(row) is not Sample:
            raise TypeError("snapshot rows must be Sample instances")
        samples.append(
            Sample(
                text(row.sample_id),
                text(row.group_id),
                number(row.reward),
                vector(row.mask, boolean=True),
                vector(row.advantage),
            )
        )
    members = []
    for pair in evidence.membership:
        if type(pair) not in (tuple, list) or len(pair) != 2:
            raise TypeError("invalid snapshot membership")
        members.append((text(pair[0]), text(pair[1])))
    if type(evidence.provenance) is not dict:
        raise TypeError("snapshot provenance must be a dictionary")
    provenance = {text(key): text(value) for key, value in evidence.provenance.items()}
    return Evidence(
        owned_profile,
        tuple(samples),
        tuple(members),
        provenance,
        evidence.complete_groups,
        evidence.schema_version,
    ), charged
