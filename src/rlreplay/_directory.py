"""OS-released single-writer leases for local capture directories."""

import errno
import os
from contextlib import contextmanager
from pathlib import Path


def _lock(stream, *, unlock=False):
    stream.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(
            stream.fileno(), msvcrt.LK_UNLCK if unlock else msvcrt.LK_NBLCK, 1
        )
    else:
        import fcntl

        fcntl.flock(
            stream.fileno(), fcntl.LOCK_UN if unlock else fcntl.LOCK_EX | fcntl.LOCK_NB
        )


@contextmanager
def writer_directory(directory):
    """Hold a local advisory lock. The persistent lock file is intentionally kept."""
    directory = Path(directory)
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(directory / ".rlreplay.lock", os.O_CREAT | os.O_RDWR, 0o600)
    with os.fdopen(descriptor, "r+b") as stream:
        if os.fstat(stream.fileno()).st_size == 0:
            stream.write(b"\0")
            stream.flush()
        try:
            _lock(stream)
        except OSError as error:
            if error.errno in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                raise BlockingIOError(
                    "capture directory has an active writer"
                ) from error
            raise
        try:
            yield directory
        finally:
            _lock(stream, unlock=True)
