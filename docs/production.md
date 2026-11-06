# Operating the capture runtime

This guide defines the current local runtime contract. Performance and resource
qualification apply to the measured configuration and workload.

## Lifecycle and policy

Create one `Recorder` per producer and output directory. Configure the codec and
budgets before the training loop. Setup errors, including invalid options or a
missing optional codec, raise in the control plane.

`VerlObserver.around()` uses an observing policy during training. It preserves the
training function's return value and exception, while recording capture failures.
The producer reserves capacity, freezes the input values, and queues the snapshot.
The isolated worker validates and encodes the snapshot, optionally checks it,
then publishes the complete file atomically.

`submit()` returns a capture ID for queued work. Schema and encoding failures
arrive asynchronously in `poll()` with `state=ERROR` and a phase. Invalid Python
input types fail synchronously while releasing their reservation. A snapshot
budget rejection returns `None` with a DROPPED receipt.

Use `close(timeout=...)` to drain acknowledged work and stop the owned worker.
It has a bounded graceful drain followed by up to one second of termination
cleanup. Unacknowledged items retain PARTIAL status. The closed object keeps its
metrics and releases its IPC objects. A failed instance can be closed and replaced
explicitly with a new Recorder.

The worker exits when its parent dies. A worker failure closes the receipt
channel, marks unconfirmed items PARTIAL and prevents new admission. Polling also
exposes worker liveness and pending age. A blocked local write can fill the bounded
backlog while training continues and subsequent captures are dropped.

Offline `check` and exported regressions supply CI failure gates through their
existing exit codes and assertions. Coordinated distributed training interruption
requires a framework safe-point contract beyond the current observing policy.

## Budgets and ownership

The defaults are two queue entries plus one active/reserved item, a 16 MiB
conservative input charge per snapshot, a 4 MiB encoded capsule limit, and a
64 MiB logical artifact storage budget.

Input charge accounts for primitive values, containers and string storage. It is
separate from actual parent/worker RSS, allocator retention, the encoder workspace
and filesystem allocation. `max_inflight_snapshot_charge` reports the input
accounting bound. The legacy `max_queued_bytes` field reports the encoded-output
bound. Resource measurements report actual RSS independently.

Only supported primitive schema values enter the internal multiprocessing channel.
Lists and provenance dictionaries are frozen before `submit()` returns. Tensors,
arbitrary objects and excessive snapshots are rejected before queue publication.
The caller owns the data during capture and provides a stable computation boundary.
Public capsules use JSON or safetensors, with explicit schema and format versions.

Storage accounting includes complete JSON/S files and stranded `.partial` files.
Failure evidence is retained and counts against a restarted writer's budget.
Filesystem block allocation and directory metadata are separate from logical file
byte counts. A published capture ID and an existing partial file are preserved on
collision. The caller can export or remove retained evidence under its own policy.

`sample_every` controls deterministic call sampling, while complete selected groups
preserve the advantage contract. Check the observed coverage alongside latency.

## Receipts and monitoring

The public receipt states distinguish publication, budget drops, cancellations,
errors and missing acknowledgements. COMPLETE means the artifact was published.
Input snapshot, submitted item, durable file and checked sample counts describe
different stages.

Enable `Recorder(..., check=True)` to run the independent GRPO check in the worker.
A complete receipt includes `check_state` and a `finding_summary` with verdict,
reason and numeric coverage. An Oracle error preserves the artifact and reports
`check_state=ERROR`. Default `check=False` is capture-only.

IPC receipts are deliberately small. They omit sample IDs and reconstruct paths in
the parent. Full findings can be recomputed from the saved capsule. Queue endpoints
are directionally closed so a peer exit produces EOF, and receipts stay within the
tested atomic message bound. This prevents a partially delivered large diagnostic
from blocking the producer's polling path.

`status()` exposes pending/reserved counts, pending input charge, oldest pending
age, process liveness and receipt-channel state. `stats` accumulates outcomes even
when the bounded recent-event buffer evicts details. Collect and route these data
through the application's existing telemetry.

## Multiple workers and local storage

Use paths such as `run-identifier/worker-0` and `run-identifier/worker-1` for separate
producers. Each directory has an OS advisory single-writer lease, including the
one-shot artifact writer. Contention produces an explicit error. The lock releases
when its process exits, and the lock file remains as a stable inode.

The tested contract covers local filesystems on Windows and Linux/WSL. Multi-host
shared filesystem locking and aggregate quota coordination require separate
qualification. Per-worker storage budgets add up across the run. Attach explicit
`worker_id`, `rank` and `world_size` provenance where the integration knows them.
Groups split across workers retain incomplete status until their required evidence
has been joined.

## Data handling

The default Evidence schema limits captured content to numeric observations, identities and source
metadata. Callers should use opaque sample/group IDs and review the sensitivity of
their metadata. Public error messages contain error type and stage rather than
raw exception values. Capsules remain local until the user explicitly exports them.

The separate [TRL native replay bundle](trl-native-replay.md) is an explicit debug
operation containing text and token IDs. It does not use the default Recorder or
its asynchronous observation policy; its capture errors propagate to the caller.

New directories/files request owner-only POSIX permissions on filesystems that
honor those modes. Windows volumes, including WSL-mounted Windows paths, use the
directory's Windows ACL. For a new local Windows output root, use
`tools/New-SecureCaptureDirectory.ps1 -Path <new-directory>`. It creates and verifies
an explicit ACL for the current user, SYSTEM and Administrators and rejects an
existing target. Subdirectories and files inherit those access rules.

Deployment owners select an appropriately restricted directory and retention
policy. Format digests check consistency, while source trust depends on the
producer and the recorded implementation/runtime evidence. Shared directory leases
across Windows and WSL runtimes or multiple hosts require separate qualification.
Use independent worker directories in the validated runtime.

The readers retain JSON v1 compatibility, reject unsupported schema/format versions,
and validate the optional archive's layout, bounds and integrity. Native replay
requires the declared framework and matching implementation/runtime, with explicit
opt-in for a candidate revision.
