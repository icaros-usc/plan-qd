"""Tests extra features of the scheduler."""

import pytest
from ribs.emitters import GaussianEmitter

from qd.archives import GridArchive

from .scheduler import Scheduler

# pylint: disable = redefined-outer-name


@pytest.fixture(params=["batch", "single"])
def scheduler(request):
    add_mode = request.param

    solution_dim = 2
    num_solutions = 2
    archive = GridArchive(
        solution_dim=solution_dim,
        dims=[100, 100],
        ranges=[(-1, 1), (-1, 1)],
        record_history=False,
    )
    result_archive = GridArchive(
        solution_dim=solution_dim,
        dims=[100, 100],
        ranges=[(-1, 1), (-1, 1)],
        record_history=False,
    )
    emitters = [
        GaussianEmitter(archive, sigma=0.1, x0=[0.0, 0.0], batch_size=num_solutions),
        GaussianEmitter(archive, sigma=0.1, x0=[0.0, 0.0], batch_size=num_solutions),
    ]
    return Scheduler(
        archive,
        emitters,
        add_mode=add_mode,
        result_archive=result_archive,
    )


def test_no_success_mask(scheduler):
    _ = scheduler.ask()
    scheduler.tell(
        [1, 1, 1, 1],
        [[-1, -1], [-1, 1], [1, -1], [1, 1]],
    )
    assert len(scheduler.archive) == 4
    assert len(scheduler.result_archive) == 4


def test_tell_success_mask(scheduler):
    _ = scheduler.ask()
    scheduler.tell(
        [1, 0.0, 0.0, 1],
        [[-1, -1], [-1, 1], [1, -1], [1, 1]],
        success_mask=[True, False, False, True],
    )
    assert len(scheduler.archive) == 2
    assert len(scheduler.result_archive) == 2
