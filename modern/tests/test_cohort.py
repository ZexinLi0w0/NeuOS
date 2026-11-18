import multiprocessing as mp
import os

import pytest

from neuos_modern.cohort import SHM_SIZE, LocalCohort, SharedCohort


def test_local_counter_wraps_like_cas():
    c = LocalCohort()
    assert [c.acquire_slot() for _ in range(12)] == list(range(10)) + [0, 1]


def _worker(prefix, barrier, q):
    c = SharedCohort(prefix)
    slot = c.acquire_slot()
    c.publish(slot, 100.0 + slot, 1.0)
    barrier.wait()
    q.put((slot, c.read()))
    barrier.wait()
    c.close()


def test_shared_cohort_across_processes():
    prefix = "nt%d" % os.getpid()
    SharedCohort.unlink(prefix)
    ctx = mp.get_context("spawn")
    barrier, q = ctx.Barrier(2), ctx.Queue()
    try:
        ps = [ctx.Process(target=_worker, args=(prefix, barrier, q)) for _ in range(2)]
        for p in ps:
            p.start()
        results = [q.get(timeout=60) for _ in ps]
        for p in ps:
            p.join(timeout=60)
            assert p.exitcode == 0
        slots = sorted(r[0] for r in results)
        assert slots == [0, 1]
        for _, data in results:
            assert data[0] == 100.0 and data[1] == 101.0 and len(data) == SHM_SIZE
        c = SharedCohort(prefix)  # attach: state persists after the workers exit
        assert c.acquire_slot() == 2
        c.close()
    finally:
        SharedCohort.unlink(prefix)
    with pytest.raises(FileNotFoundError):
        from multiprocessing import shared_memory
        shared_memory.SharedMemory(name=prefix + "_data")
