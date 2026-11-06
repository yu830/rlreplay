"""Bounded asynchronous capture with owned inputs and durable completion receipts."""

import logging
import multiprocessing
import queue
import time
import uuid
from collections import Counter, deque
from pathlib import Path

from rlreplay._directory import writer_directory
from rlreplay._receipts import pack as pack_receipt
from rlreplay._receipts import unpack as unpack_receipt
from rlreplay._snapshot import DEFAULT_SNAPSHOT_BUDGET, SnapshotBudgetExceeded, freeze
from rlreplay.artifacts import (
    CAPSULE_SUFFIXES,
    MAX_CAPSULE_BYTES,
    encode,
    prepare_codec,
    write_payload,
)
from rlreplay.contracts.grpo import check as check_grpo
from rlreplay.core import Evidence

LOGGER = logging.getLogger(__name__)


def _send(outgoing, event):
    outgoing.put(pack_receipt(event))


def _stored_bytes(directory):
    return sum(
        path.stat().st_size
        for suffix in (*CAPSULE_SUFFIXES.values(), ".partial")
        for path in directory.glob(f"*{suffix}")
    )


def _consume(directory, incoming, outgoing, storage_budget, codec, do_check, parent):
    used = _stored_bytes(directory)
    while parent.is_alive():
        try:
            message = incoming.get(timeout=0.25)
        except queue.Empty:
            continue
        except (EOFError, OSError):
            if not parent.is_alive():
                return
            raise
        if message is None:
            return
        capture_id, evidence, enqueued_at = message
        started = time.perf_counter_ns()
        details = {
            "queue_wait_ns": max(0, started - enqueued_at),
            "check_state": "NOT_RUN",
        }
        phase = "encode"
        try:
            payload = encode(evidence, codec=codec)
            details["encode_ns"] = time.perf_counter_ns() - started
            if do_check:
                check_started = time.perf_counter_ns()
                try:
                    details.update(
                        check_state="COMPLETE", finding=check_grpo(evidence).to_dict()
                    )
                except MemoryError:
                    raise
                except Exception as error:
                    LOGGER.exception(
                        "capture oracle failed (%s)",
                        type(error).__name__,
                        exc_info=False,
                    )
                    details.update(
                        check_state="ERROR", check_error=type(error).__name__
                    )
                details["check_ns"] = time.perf_counter_ns() - check_started
            if used + len(payload) > storage_budget:
                _send(
                    outgoing,
                    {
                        "id": capture_id,
                        "state": "DROPPED",
                        "reason": "storage_budget",
                        **details,
                    },
                )
                continue
            phase = "write"
            write_started = time.perf_counter_ns()
            path = write_payload(
                directory, capture_id, payload, suffix=CAPSULE_SUFFIXES[codec]
            )
            details["write_ns"] = time.perf_counter_ns() - write_started
        except MemoryError:
            raise
        except Exception as error:
            LOGGER.exception(
                "capture %s failed (%s)", phase, type(error).__name__, exc_info=False
            )
            accounting_failed = False
            if phase == "write":
                try:
                    used = _stored_bytes(directory)
                except OSError:
                    accounting_failed = True
            _send(
                outgoing,
                {
                    "id": capture_id,
                    "state": "ERROR",
                    "reason": type(error).__name__,
                    "phase": phase,
                    **details,
                },
            )
            if accounting_failed:
                return
        else:
            used += len(payload)
            _send(
                outgoing,
                {
                    "id": capture_id,
                    "state": "COMPLETE",
                    "path": str(path),
                    "bytes": len(payload),
                    **details,
                },
            )


def _writer(directory, incoming, outgoing, storage_budget, codec, do_check):
    incoming._writer.close()
    outgoing._reader.close()
    parent = multiprocessing.parent_process()
    phase = "initialize"
    try:
        with writer_directory(directory):
            phase = "worker"
            _consume(
                directory, incoming, outgoing, storage_budget, codec, do_check, parent
            )
    except (OSError, EOFError) as error:
        _send(
            outgoing, {"state": "ERROR", "reason": type(error).__name__, "phase": phase}
        )
    finally:
        if not parent.is_alive():
            # A dead parent cannot consume acknowledgements from the feeder thread.
            outgoing.cancel_join_thread()


class Recorder:
    """Single-producer capture with bounded input ownership and an isolated worker.

    Reservations, queued items and the active worker are limited to queue_capacity
    + 1. Input charge and encoded file limits are separate from measured process RSS.
    submit() freezes inputs; encoding/checking failures arrive through poll().
    """

    def __init__(
        self,
        directory,
        *,
        queue_capacity=2,
        storage_budget=64 * 1024 * 1024,
        codec="json",
        snapshot_budget=DEFAULT_SNAPSHOT_BUDGET,
        check=False,
    ):
        if any(
            type(value) is not int or value < 1
            for value in (queue_capacity, storage_budget, snapshot_budget)
        ):
            raise ValueError("capture budgets must be positive integers")
        if type(check) is not bool:
            raise TypeError("check must be boolean")
        self.directory = Path(directory).resolve()
        prepare_codec(codec)
        self.codec = codec
        self.storage_budget = storage_budget
        self.snapshot_budget = snapshot_budget
        # This legacy field bounds encoded outputs, not in-memory Python snapshots.
        self.max_queued_bytes = (queue_capacity + 1) * MAX_CAPSULE_BYTES
        self.max_inflight_snapshot_charge = (queue_capacity + 1) * snapshot_budget
        self._capacity = queue_capacity + 1
        context = multiprocessing.get_context("spawn")
        self._incoming = context.Queue(maxsize=queue_capacity)
        self._outgoing = context.Queue(maxsize=queue_capacity + 1)
        self._process = context.Process(
            target=_writer,
            args=(
                self.directory,
                self._incoming,
                self._outgoing,
                storage_budget,
                codec,
                check,
            ),
            daemon=True,
        )
        self._closed = False
        self._stopping = False
        self._exit_seen = False
        self._channel_failed = False
        self._events = deque(maxlen=128)
        self.stats = Counter()
        self._pending = {}
        self._reservations = set()
        self._process.start()
        self._incoming._reader.close()
        self._outgoing._writer.close()

    def reserve(self) -> str | None:
        """Admit work before tensor copies, Python conversion, or serialization."""
        self._collect()
        self.stats["attempted"] += 1
        if self._closed or self._channel_failed or not self._process.is_alive():
            self._remember({"state": "ERROR", "reason": "writer_unavailable"})
            return None
        if len(self._pending) + len(self._reservations) >= self._capacity:
            self._remember({"state": "DROPPED", "reason": "inflight_budget"})
            return None
        if self._incoming.full():
            self._remember({"state": "DROPPED", "reason": "queue_full"})
            return None
        capture_id = str(uuid.uuid4())
        self._reservations.add(capture_id)
        self.stats["reserved"] += 1
        return capture_id

    def cancel(self, capture_id: str, *, reason="cancelled") -> None:
        """Release an unsubmitted reservation, including on training exceptions."""
        if capture_id in self._reservations:
            self._reservations.remove(capture_id)
            self._remember({"id": capture_id, "state": "CANCELLED", "reason": reason})

    def submit(
        self, evidence: Evidence, *, reservation: str | None = None
    ) -> str | None:
        """Freeze supported values and queue them; worker failures arrive via poll()."""
        capture_id = reservation if reservation is not None else self.reserve()
        if capture_id is None:
            return None
        if capture_id not in self._reservations:
            raise ValueError("capture reservation is not active")
        if self._closed or self._channel_failed or not self._process.is_alive():
            self.cancel(capture_id, reason="writer_unavailable")
            return None
        try:
            owned, charged = freeze(evidence, self.snapshot_budget)
        except SnapshotBudgetExceeded:
            self._reservations.remove(capture_id)
            self._remember(
                {"id": capture_id, "state": "DROPPED", "reason": "snapshot_budget"}
            )
            return None
        except (ValueError, TypeError, OverflowError):
            self.cancel(capture_id, reason="snapshot_invalid")
            raise
        enqueued_at = time.perf_counter_ns()
        try:
            self._incoming.put_nowait((capture_id, owned, enqueued_at))
        except queue.Full:
            self._reservations.remove(capture_id)
            self._remember(
                {"id": capture_id, "state": "DROPPED", "reason": "queue_full"}
            )
            return None
        self._reservations.remove(capture_id)
        self._pending[capture_id] = (time.monotonic(), charged)
        self.stats["submitted"] += 1
        return capture_id

    def _remember(self, event):
        if event["state"] == "COMPLETE":
            event["path"] = str(
                self.directory / f"{event['id']}{CAPSULE_SUFFIXES[self.codec]}"
            )
        self.stats[event["state"]] += 1
        if event.get("check_state") == "ERROR":
            self.stats["check_errors"] += 1
        if "finding_summary" in event:
            self.stats["finding_" + event["finding_summary"]["verdict"]] += 1
        if len(self._events) == self._events.maxlen:
            self.stats["event_details_evicted"] += 1
        self._events.append(event)

    def _collect(self):
        if self._closed:
            return
        while True:
            try:
                event = unpack_receipt(self._outgoing.get_nowait())
            except queue.Empty:
                break
            except (EOFError, OSError, ValueError):
                if (
                    not self._channel_failed
                    and not self._exit_seen
                    and not self._stopping
                ):
                    self._remember(
                        {"state": "ERROR", "reason": "receipt_channel_closed"}
                    )
                self._channel_failed = True
                break
            self._remember(event)
            self._pending.pop(event.get("id"), None)
            if event["state"] == "ERROR" and "id" not in event:
                self._channel_failed = True
                self._exit_seen = True
        if self._channel_failed or not self._process.is_alive():
            if not self._exit_seen and not self._stopping and not self._channel_failed:
                self._remember(
                    {
                        "state": "ERROR",
                        "reason": "writer_exited",
                        "exitcode": self._process.exitcode,
                    }
                )
            self._exit_seen = True
            for capture_id in self._pending:
                self._remember(
                    {
                        "id": capture_id,
                        "state": "PARTIAL",
                        "reason": "receipt_unavailable"
                        if self._channel_failed
                        else "writer_exited",
                    }
                )
            self._pending.clear()

    def status(self) -> dict:
        """Report liveness and backlog without treating an alive process as healthy."""
        self._collect()
        now = time.monotonic()
        return {
            "closed": self._closed,
            "worker_alive": not self._closed and self._process.is_alive(),
            "receipt_channel_open": not self._closed and not self._channel_failed,
            "reserved": len(self._reservations),
            "pending": len(self._pending),
            "pending_snapshot_charge": sum(
                charge for _, charge in self._pending.values()
            ),
            "oldest_pending_seconds": max(
                (now - started for started, _ in self._pending.values()), default=0.0
            ),
        }

    def poll(self) -> list[dict]:
        self._collect()
        events = list(self._events)
        self._events.clear()
        return events

    def close(self, *, timeout=5.0) -> list[dict]:
        if not self._closed:
            self._stopping = True
            for capture_id in tuple(self._reservations):
                self.cancel(capture_id, reason="closed_before_submit")
            deadline = time.monotonic() + timeout
            sentinel_sent = False
            while self._process.is_alive() and time.monotonic() < deadline:
                self._collect()
                try:
                    if not sentinel_sent:
                        self._incoming.put_nowait(None)
                        sentinel_sent = True
                except queue.Full:
                    pass  # Drain receipts before retrying the bounded shutdown request.
                self._process.join(0.02)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(1.0)
                self._remember({"state": "ERROR", "reason": "writer_shutdown_timeout"})
                if self._process.is_alive():
                    self._remember(
                        {"state": "ERROR", "reason": "writer_termination_pending"}
                    )
            self._collect()
            for item in self._pending:
                self._remember(
                    {"id": item, "state": "PARTIAL", "reason": "writer_receipt_missing"}
                )
            events = self.poll()
            self._pending.clear()
            self._closed = True
            self._incoming.cancel_join_thread()
            self._incoming.close()
            self._outgoing.close()
            # Release SemLock ownership while the parent's resource tracker is alive.
            self._incoming = None
            self._outgoing = None
            if not self._process.is_alive():
                self._process.close()
            return events
        return []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._events.extend(self.close())
