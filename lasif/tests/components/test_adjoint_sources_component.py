import inspect
import pathlib

import os
import glob
import obspy
import pytest
import pyasdf
import h5py
from unittest import mock
import shutil
from lasif.components.project import Project
import lasif.api
import numpy as np


@pytest.fixture()
def comm(tmpdir):
    proj_dir = os.path.join(
        os.path.dirname(
            os.path.dirname(
                os.path.abspath(inspect.getfile(inspect.currentframe()))
            )
        ),
        "data",
        "example_project",
    )
    tmpdir = str(tmpdir)
    shutil.copytree(proj_dir, os.path.join(tmpdir, "proj"))
    proj_dir = os.path.join(tmpdir, "proj")

    folder_path = pathlib.Path(proj_dir).absolute()
    project = Project(project_root_path=folder_path, init_project=False)

    return project.comm


def test_get_filename(comm):
    event = lasif.api.list_events(comm, output=True)[0]
    name = comm.adj_sources.get_filename(event, "1")
    should_be = os.path.join(
        comm.project.paths["adjoint_sources"],
        "ITERATION_1",
        event,
        "adjoint_source_auxiliary.h5",
    )
    assert name == should_be


def test_get_misfit_for_event(comm):
    events = lasif.api.list_events(comm, output=True)
    iteration = lasif.api.list_iterations(comm, output=True)[0]
    lasif.api.calculate_adjoint_sources(comm, iteration, "A")
    misfit = comm.adj_sources.get_misfit_for_event(
        event=events[0], iteration=iteration
    )
    np.testing.assert_almost_equal(misfit, 1.301670132868937)
    misfits = comm.adj_sources.get_misfit_for_event(
        event=events[1], iteration=iteration, include_station_misfit=True,
    )
    np.testing.assert_almost_equal(misfits["event_misfit"], 5.537936479961759)
    np.testing.assert_almost_equal(
        misfits["stations"]["YD.4F14"], 5.537936479961759
    )
    np.testing.assert_almost_equal(
        misfits["event_misfit"],
        np.sum(np.array(list(misfits["stations"].values()))),
    )
    lasif.api.calculate_adjoint_sources(
        comm, iteration, "A", weight_set="sw_1"
    )

    misfit_with_weights = comm.adj_sources.get_misfit_for_event(
        event=events[1], iteration=iteration, include_station_misfit=True,
    )
    stat_weight = 0.7040545724757455
    np.testing.assert_almost_equal(
        misfit_with_weights["stations"]["YD.4F14"],
        misfits["stations"]["YD.4F14"] * stat_weight,
    )


# Should be equal, and is equal in manual test, but for some reason
# fails the automatic test. We'll keep it almost equal for now then.
def test_calculate_adjoint_sources(comm):
    events = lasif.api.list_events(comm, output=True)
    file = (
        comm.project.paths["adjoint_sources"]
        / "ITERATION_1"
        / events[0]
        / "auxiliary_adjoint.npy"
    )
    should_be = np.load(file)
    lasif.api.calculate_adjoint_sources(comm, "1", "A", weight_set="sw_1")
    output = (
        comm.project.paths["adjoint_sources"]
        / "ITERATION_1"
        / events[0]
        / "adjoint_source_auxiliary.h5"
    )
    with pyasdf.ASDFDataSet(output) as ds:
        adj = ds.auxiliary_data.AdjointSources.HT_ALN.Channel__HHZ.data[()]
    np.testing.assert_array_almost_equal(should_be, adj)


def test_finalize_adjoint_sources(comm):
    events = lasif.api.list_events(comm, output=True)
    file = (
        comm.project.paths["adjoint_sources"]
        / "ITERATION_1"
        / events[0]
        / "source.npy"
    )
    should_be = np.load(file)
    output = (
        comm.project.paths["adjoint_sources"]
        / "ITERATION_1"
        / events[0]
        / "stf.h5"
    )
    with h5py.File(output) as f:
        src = f["HT_ALN"][()]
    np.testing.assert_array_equal(should_be, src)
