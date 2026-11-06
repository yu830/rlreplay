"""Small primitive IPC receipts; full diagnostics remain reproducible from capsules."""

import json

MAX_RECEIPT_BYTES = 384
STATES = ("COMPLETE", "DROPPED", "ERROR")
CHECK_STATES = ("NOT_RUN", "COMPLETE", "ERROR")
VERDICTS = ("", "PASS", "CONTRACT_VIOLATION", "SUSPECTED_MISMATCH", "INCONCLUSIVE")


def _code(value, limit):
    return "".join(
        char if char.isascii() and (char.isalnum() or char in "_-. ") else "_"
        for char in value[:limit]
    )


def pack(event: dict) -> bytes:
    finding = event.get("finding", {})
    data = [
        event.get("id"),
        STATES.index(event["state"]),
        _code(event.get("reason", ""), 48),
        _code(event.get("phase", ""), 16),
        event.get("bytes"),
        event.get("queue_wait_ns"),
        event.get("encode_ns"),
        event.get("write_ns"),
        event.get("check_ns"),
        CHECK_STATES.index(event.get("check_state", "NOT_RUN")),
        VERDICTS.index(finding.get("verdict", "")),
        _code(finding.get("reason", ""), 64),
        finding.get("max_abs_error"),
        finding.get("checked_samples"),
        finding.get("total_samples"),
        _code(event.get("check_error", ""), 48),
    ]
    payload = json.dumps(data, separators=(",", ":"), allow_nan=False).encode("ascii")
    if len(payload) > MAX_RECEIPT_BYTES:
        raise ValueError("internal receipt exceeds atomic message budget")
    return payload


def unpack(payload: bytes) -> dict:
    if not isinstance(payload, bytes) or len(payload) > MAX_RECEIPT_BYTES:
        raise ValueError("invalid internal receipt")
    row = json.loads(payload)
    if not isinstance(row, list) or len(row) != 16:
        raise ValueError("invalid internal receipt layout")
    event = {"state": STATES[row[1]], "check_state": CHECK_STATES[row[9]]}
    for key, index in (
        ("id", 0),
        ("reason", 2),
        ("phase", 3),
        ("bytes", 4),
        ("queue_wait_ns", 5),
        ("encode_ns", 6),
        ("write_ns", 7),
        ("check_ns", 8),
        ("check_error", 15),
    ):
        if row[index] is not None and row[index] != "":
            event[key] = row[index]
    if row[10]:
        event["finding_summary"] = {
            "verdict": VERDICTS[row[10]],
            "reason": row[11],
            "max_abs_error": row[12],
            "checked_samples": row[13],
            "total_samples": row[14],
            "details": "recompute_from_capsule",
        }
    return event
