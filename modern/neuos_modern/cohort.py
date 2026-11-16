"""Cohort shared state: replaces the System V shm segments + ftok() of net.cpp.

Layout (same as the original): ``data[10]`` float32 (each job's epsilon / slack),
``approx[10]`` float32 (each job's max_score) and an int32 slot counter.
Slots are never cleared, as in the original.
"""
import contextlib
import fcntl
import os
import struct
import sys
import threading
from multiprocessing import resource_tracker, shared_memory

from .schedule import f32

SHM_SIZE = 10


def _next_slot(old):
    """__sync_fetch_and_add(&c,1) then __sync_val_compare_and_swap(&c,10,0).

    Returns ``(slot, new_counter)``. The slot is taken modulo SHM_SIZE so it can
    never index past the arrays (the original could, under a race).
    """
    new = old + 1
    if new == SHM_SIZE:
        new = 0
    return old % SHM_SIZE, new


class LocalCohort:
    """In-process cohort (tests, single-process experiments)."""

    def __init__(self):
        self.data = [0.0] * SHM_SIZE
        self.approx = [0.0] * SHM_SIZE
        self.counter = 0
        self._lock = threading.Lock()

    def acquire_slot(self):
        with self._lock:
            slot, self.counter = _next_slot(self.counter)
        return slot

    def publish(self, slot, epsilon, approx):
        self.data[slot] = f32(epsilon)
        self.approx[slot] = f32(approx)

    def read(self):
        return list(self.data)

    def close(self):
        pass


def _open_shm(name, size):
    kw = {"track": False} if sys.version_info >= (3, 13) else {}
    try:
        shm = shared_memory.SharedMemory(name=name, create=True, size=size, **kw)
    except FileExistsError:
        shm = shared_memory.SharedMemory(name=name, **kw)
    if not kw:  # keep the block alive after this process exits, like SysV shm
        resource_tracker.unregister(shm._name, "shared_memory")
    return shm


class SharedCohort:
    """Cross-process cohort on POSIX shared memory (create-or-attach by name).

    Blocks: ``<prefix>_data``, ``<prefix>_approx``, ``<prefix>_counter``.
    The atomic fetch-add is an ``fcntl.flock`` on ``<lock_dir>/<prefix>.lock``.
    """

    def __init__(self, prefix="neuos", lock_dir=None):
        self.prefix = prefix
        lock_dir = lock_dir or ("/dev/shm" if os.path.isdir("/dev/shm") else "/tmp")
        self._lockf = open(os.path.join(lock_dir, prefix + ".lock"), "a+")
        with self._locked():
            self._data = _open_shm(prefix + "_data", 4 * SHM_SIZE)
            self._approx = _open_shm(prefix + "_approx", 4 * SHM_SIZE)
            self._counter = _open_shm(prefix + "_counter", 4)

    @contextlib.contextmanager
    def _locked(self):
        fcntl.flock(self._lockf, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(self._lockf, fcntl.LOCK_UN)

    def acquire_slot(self):
        with self._locked():
            old = struct.unpack_from("i", self._counter.buf, 0)[0]
            slot, new = _next_slot(old)
            struct.pack_into("i", self._counter.buf, 0, new)
        return slot

    def publish(self, slot, epsilon, approx):
        struct.pack_into("f", self._data.buf, 4 * slot, epsilon)
        struct.pack_into("f", self._approx.buf, 4 * slot, approx)

    def read(self):
        return list(struct.unpack_from("%df" % SHM_SIZE, self._data.buf, 0))

    def read_approx(self):
        return list(struct.unpack_from("%df" % SHM_SIZE, self._approx.buf, 0))

    def close(self):
        for shm in (self._data, self._approx, self._counter):
            shm.close()
        self._lockf.close()

    @staticmethod
    def unlink(prefix="neuos", lock_dir=None):
        """Remove the shared blocks and the lock file (the ``ipcrm`` equivalent)."""
        for suffix in ("_data", "_approx", "_counter"):
            try:
                shm = shared_memory.SharedMemory(name=prefix + suffix)
            except FileNotFoundError:
                continue
            shm.close()
            shm.unlink()
        lock_dir = lock_dir or ("/dev/shm" if os.path.isdir("/dev/shm") else "/tmp")
        with contextlib.suppress(FileNotFoundError):
            os.remove(os.path.join(lock_dir, prefix + ".lock"))
