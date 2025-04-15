"""Tests for GridArchive."""

import numpy as np

from .grid_archive import GridArchive


def test_history():
    archive = GridArchive(solution_dim=2, dims=[10, 10], ranges=[(-1, 1)] * 2)

    archive.new_history_gen()
    archive.add_single([1, 2], 1.0, [0.5, 0.5])
    archive.add_single([1, 2], 0.0, [0.5, 0.5])
    archive.add_single([1, 2], 2.0, [0.5, 0.5])

    archive.new_history_gen()
    archive.add(
        [[1, 2], [1, 2], [1, 2]],
        [2.0, 3.0, 4.0],
        [[0.1, 0.1], [0.1, 0.1], [0.1, 0.1]],
    )

    h = archive.history()

    # Check h[0].

    assert len(h[0]) == 3

    assert h[0][0][0] == "add_single"
    assert h[0][0][1] == 1.0
    assert np.all(h[0][0][2] == [0.5, 0.5])

    assert h[0][1][0] == "add_single"
    assert h[0][1][1] == 0.0
    assert np.all(h[0][1][2] == [0.5, 0.5])

    assert h[0][2][0] == "add_single"
    assert h[0][2][1] == 2.0
    assert np.all(h[0][2][2] == [0.5, 0.5])

    # Check h[1].

    assert len(h[1]) == 1

    assert h[1][0][0] == "add"
    assert np.all(h[1][0][1] == [2.0, 3.0, 4.0])
    assert np.all(h[1][0][2] == [[0.1, 0.1], [0.1, 0.1], [0.1, 0.1]])
