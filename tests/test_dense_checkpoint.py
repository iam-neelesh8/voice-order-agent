"""Stage 2 -- the dense index checkpoint format.

Embedding 100k titles is a job measured in hours on a laptop, so the build
checkpoints each chunk and resumes. That only works if the chunk files land
under the exact names the resume logic looks for -- and numpy has a sharp
edge here that broke it once already.
"""

from __future__ import annotations

import numpy as np


def test_numpy_renames_paths_that_do_not_end_in_npy(tmp_path):
    """Documents the trap, so nobody 'simplifies' the write back into it.

    np.save silently appends '.npy' to a path lacking that suffix, so writing
    a temp file called 'c0.npy.partial' produces 'c0.npy.partial.npy' and the
    atomic rename then fails on a file that was never created.
    """
    target = tmp_path / "chunk.npy.partial"
    np.save(target, np.zeros((2, 3), dtype=np.float32))

    assert not target.exists()
    assert (tmp_path / "chunk.npy.partial.npy").exists()


def test_saving_through_a_handle_keeps_the_exact_name(tmp_path):
    """Which is why the build writes through an open file object."""
    target = tmp_path / "chunk.npy.partial"
    with target.open("wb") as fh:
        np.save(fh, np.zeros((2, 3), dtype=np.float32))

    assert target.exists()
    assert not (tmp_path / "chunk.npy.partial.npy").exists()


def test_a_checkpoint_round_trips(tmp_path):
    """Write -> atomic rename -> load must return the same vectors."""
    vectors = np.random.default_rng(0).random((5, 384)).astype(np.float32)
    part = tmp_path / "c00000000.npy"
    tmp = part.with_name(part.name + ".partial")

    with tmp.open("wb") as fh:
        np.save(fh, vectors)
    tmp.replace(part)

    assert np.array_equal(np.load(part), vectors)
    assert not tmp.exists()
