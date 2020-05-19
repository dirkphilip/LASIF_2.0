import inspect
import pathlib

import os
import glob
import obspy
import pytest
from unittest import mock
import shutil
from lasif.components.project import Project
import lasif.api


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
    name = comm.adjoint_sources.get_filename(event, "1")
    should_be = os.path.join(
        comm.adjoint_sources._folder,
        "ITERATION_1",
        event,
        "adjoint_source_auxiliary.h5",
    )
    assert name == should_be


def test_get_misfit_for_event(comm):
    event = lasif.api.list_events(comm, output=True)[0]
    iteration = lasif.api.list_iterations(comm, output=True)[0]

    misfit = comm.adjoint_sources.get_misfit_for_event(
        event=event, iteration=iteration
    )
