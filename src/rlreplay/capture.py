"""Bounded asynchronous capture with owned inputs and publication receipts."""

import logging
import multiprocessing
import os
import queue
import time
import uuid
from collections import Counter, deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
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
_BUDGET_RESCAN_INTERVAL = 1.0


def _send(outgoing, event):
    outgoing.put(pack_receipt(event))


def _stored_bytes(directory):
    used = 0
    suffixes = tuple(
        os.path.normcase(suffix) for suffix in (*CAPSULE_SUFFIXES.values(), ".partial")
    )
    with os.scandir(directory) as entries:
        for entry in entries:
            if os.path.normcase(entry.name).endswith(suffixes):
                try:
                    # DirEntry.stat() can cache enumeration data on Windows.
                    used += os.stat(entry.path).st_size
                except FileNotFoundError:
                    # The owner may delete a file while we are accounting for it.
                    continue
    return used


def _encode_for_capture(evidence, codec, do_check, started, details):
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
                "capture oracle failed (%s)", type(error).__name__, exc_info=False
            )
            details.update(check_state="ERROR", check_error=type(error).__name__)
        details["check_ns"] = time.perf_counter_ns() - check_started
    return payload


def _consume(directory, incoming, outgoing, storage_budget, codec, do_check, parent):
    used = _stored_bytes(directory)
    last_budget_scan = None
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
            payload = _encode_for_capture(evidence, codec, do_check, started, details)
            if used + len(payload) > storage_budget:
                now = time.monotonic()
                if (
                    last_budget_scan is None
                    or now - last_budget_scan >= _BUDGET_RESCAN_INTERVAL
                ):
                    last_budget_scan = now
                    phase = "budget"
                    used = _stored_bytes(directory)
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


def _consume_concurrent(
    directory, incoming, outgoing, storage_budget, codec, do_check, parent, concurrency
):
    """Serialize admission and accounting while publishing distinct files in parallel."""
    used = _stored_bytes(directory)
    reserved = 0
    last_budget_scan = None
    accounting_dirty = False
    pending = {}

    def publish(capture_id, payload):
        started = time.perf_counter_ns()
        path = write_payload(
            directory, capture_id, payload, suffix=CAPSULE_SUFFIXES[codec]
        )
        return path, time.perf_counter_ns() - started

    def collect(*, block=False):
        nonlocal used, reserved, accounting_dirty
        if not pending:
            return
        if block:
            finished, _ = wait(pending, timeout=0.05, return_when=FIRST_COMPLETED)
        else:
            finished = {future for future in pending if future.done()}
        for future in finished:
            capture_id, size, details = pending.pop(future)
            reserved -= size
            try:
                path, write_ns = future.result()
            except MemoryError:
                raise
            except Exception as error:
                LOGGER.exception(
                    "capture write failed (%s)", type(error).__name__, exc_info=False
                )
                _send(
                    outgoing,
                    {
                        "id": capture_id,
                        "state": "ERROR",
                        "reason": type(error).__name__,
                        "phase": "write",
                        **details,
                    },
                )
                accounting_dirty = True
            else:
                used += size
                details["write_ns"] = write_ns
                _send(
                    outgoing,
                    {
                        "id": capture_id,
                        "state": "COMPLETE",
                        "path": str(path),
                        "bytes": size,
                        **details,
                    },
                )

    def drain():
        while pending and parent.is_alive():
            collect(block=True)
        return parent.is_alive()

    def reconcile():
        nonlocal used, accounting_dirty
        if not accounting_dirty:
            return True
        if not drain():
            return False
        try:
            used = _stored_bytes(directory)
        except OSError:
            return False
        accounting_dirty = False
        return True

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        while parent.is_alive():
            collect()
            if not reconcile():
                return
            if len(pending) >= concurrency:
                collect(block=True)
                continue
            try:
                message = incoming.get(timeout=0.01 if pending else 0.25)
            except queue.Empty:
                continue
            except (EOFError, OSError):
                if not parent.is_alive():
                    return
                raise
            if message is None:
                if drain():
                    reconcile()
                return

            capture_id, evidence, enqueued_at = message
            started = time.perf_counter_ns()
            details = {
                "queue_wait_ns": max(0, started - enqueued_at),
                "check_state": "NOT_RUN",
            }
            phase = "encode"
            payload = None
            try:
                payload = _encode_for_capture(
                    evidence, codec, do_check, started, details
                )
                if used + reserved + len(payload) > storage_budget:
                    now = time.monotonic()
                    if (
                        last_budget_scan is None
                        or now - last_budget_scan >= _BUDGET_RESCAN_INTERVAL
                    ):
                        phase = "budget"
                        # A scan must not double-count files currently being written.
                        if not drain():
                            return
                        last_budget_scan = time.monotonic()
                        used = _stored_bytes(directory)
                        accounting_dirty = False
                if used + reserved + len(payload) > storage_budget:
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
                size = len(payload)
                future = pool.submit(publish, capture_id, payload)
                pending[future] = (capture_id, size, details)
                reserved += size
            except MemoryError:
                raise
            except Exception as error:
                LOGGER.exception(
                    "capture %s failed (%s)",
                    phase,
                    type(error).__name__,
                    exc_info=False,
                )
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
                if accounting_dirty:
                    return
            finally:
                # Keep at most one encoded payload per configured I/O task.
                payload = None


def _writer(
    directory, incoming, outgoing, storage_budget, codec, do_check, io_concurrency=1
):
    incoming._writer.close()
    outgoing._reader.close()
    parent = multiprocessing.parent_process()
    phase = "initialize"
    try:
        with writer_directory(directory):
            phase = "worker"
            if io_concurrency == 1:
                _consume(
                    directory,
                    incoming,
                    outgoing,
                    storage_budget,
                    codec,
                    do_check,
                    parent,
                )
            else:
                _consume_concurrent(
                    directory,
                    incoming,
                    outgoing,
                    storage_budget,
                    codec,
                    do_check,
                    parent,
                    io_concurrency,
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

    Reservations, queued items and active writes are limited to queue_capacity
    + 1. Input charge and encoded file limits are separate from measured process RSS.
    submit() freezes inputs; encoding/checking failures arrive through poll().
    An optional inflight_snapshot_budget caps the aggregate charge of pending
    snapshots. Omitting it preserves the previous per-slot accounting bound.
    io_concurrency enables up to eight parallel publications; the default is one.
    """

    def __init__(
        self,
        directory,
        *,
        queue_capacity=2,
        storage_budget=64 * 1024 * 1024,
        codec="json",
        snapshot_budget=DEFAULT_SNAPSHOT_BUDGET,
        inflight_snapshot_budget=None,
        io_concurrency=1,
        check=False,
    ):
        if any(
            type(value) is not int or value < 1
            for value in (queue_capacity, storage_budget, snapshot_budget)
        ):
            raise ValueError("capture budgets must be positive integers")
        if type(check) is not bool:
            raise TypeError("check must be boolean")
        if type(io_concurrency) is not int or not 1 <= io_concurrency <= 8:
            raise ValueError("io_concurrency must be an integer between 1 and 8")
        if inflight_snapshot_budget is not None and (
            type(inflight_snapshot_budget) is not int or inflight_snapshot_budget < 1
        ):
            raise ValueError("inflight snapshot budget must be a positive integer")
        self.directory = Path(directory).resolve()
        prepare_codec(codec)
        self.codec = codec
        self.io_concurrency = io_concurrency
        self.storage_budget = storage_budget
        self.snapshot_budget = snapshot_budget
        # This legacy field bounds encoded outputs, not in-memory Python snapshots.
        self.max_queued_bytes = (queue_capacity + 1) * MAX_CAPSULE_BYTES
        nominal_charge = (queue_capacity + 1) * snapshot_budget
        self.max_inflight_snapshot_charge = (
            nominal_charge
            if inflight_snapshot_budget is None
            else min(nominal_charge, inflight_snapshot_budget)
        )
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
            )
            + ((io_concurrency,) if io_concurrency > 1 else ()),
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
        pending_charge = sum(charge for _, charge in self._pending.values())
        if pending_charge >= self.max_inflight_snapshot_charge:
            self._remember({"state": "DROPPED", "reason": "inflight_snapshot_budget"})
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
        if reservation is not None:
            self._collect()
        if self._closed or self._channel_failed or not self._process.is_alive():
            self.cancel(capture_id, reason="writer_unavailable")
            return None
        try:
            pending_charge = sum(charge for _, charge in self._pending.values())
            remaining = self.max_inflight_snapshot_charge - pending_charge
            owned, charged = freeze(evidence, min(self.snapshot_budget, remaining))
        except SnapshotBudgetExceeded:
            self._reservations.remove(capture_id)
            reason = (
                "inflight_snapshot_budget"
                if remaining < self.snapshot_budget
                else "snapshot_budget"
            )
            self._remember({"id": capture_id, "state": "DROPPED", "reason": reason})
            return None
        except (ValueError, TypeError, OverflowError):
            self.cancel(capture_id, reason="snapshot_invalid")
            raise
        except BaseException:
            self.cancel(capture_id, reason="snapshot_failed")
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
