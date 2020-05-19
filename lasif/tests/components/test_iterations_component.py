#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import

import inspect
import pathlib
import pytest
import shutil
import os
import lasif.api
import toml

from lasif.components.project import Project


@pytest.fixture()
def comm(tmpdir):
    """
    Most visualizations need a valid project in any case, so use one for the
    tests.
    """
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


def test_get_long_iteration_name(comm):
    assert comm.iterations.get_long_iteration_name("1") == "ITERATION_1"
    assert (
        comm.iterations.get_long_iteration_name("ITERATION_1") == "ITERATION_1"
    )


def test_setup_directories_for_iteration(comm):
    events = lasif.api.list_events(comm, output=True)
    comm.iterations.setup_directories_for_iteration(
        iteration_name="4", events=events
    )
    it = comm.iterations.get_long_iteration_name("4")
    directories = []
    directories.append(comm.project.paths["eq_synthetics"] / it)
    directories.append(comm.project.paths["adjoint_sources"] / it)
    directories.append(comm.project.paths["salvus_files"] / it)
    directories.append(comm.project.paths["gradients"] / it)
    directories.append(comm.project.paths["iterations"] / it)
    directories.append(comm.project.paths["models"] / it)
    for directory in directories:
        assert os.path.exists(directory)
    comm.iterations.setup_directories_for_iteration(
        iteration_name="4", events=events, remove_dirs=True
    )
    for directory in directories:
        assert not os.path.exists(directory)
    comm.iterations.setup_directories_for_iteration(
        iteration_name="4",
        events=events[1],
        remove_dirs=False,
        event_specific=True,
    )
    directories.append(comm.project.paths["models"] / it / events[1])
    for directory in directories:
        assert os.path.exists(directory)
    other_event = comm.project.paths["models"] / it / events[0]
    assert not os.path.exists(other_event)
    comm.iterations.setup_directories_for_iteration(
        iteration_name="4",
        events=events[1],
        remove_dirs=True,
        event_specific=True,
    )
    for directory in directories:
        assert not os.path.exists(directory)


def test_setup_iteration_toml(comm):
    lasif.api.set_up_iteration(comm, "4")
    iter_file = (
        comm.project.paths["iterations"] / f"ITERATION_4" / "central_info.toml"
    )
    info = toml.load(iter_file)
    assert "events" in info.keys()
    assert "simulations" in info.keys()


def test_setup_events_toml(comm):
    lasif.api.set_up_iteration(comm, "4")
    events = lasif.api.list_events(comm, output=True)
    event_file = (
        comm.project.paths["iterations"] / f"ITERATION_4" / "events_used.toml"
    )
    events_toml = toml.load(event_file)
    assert set(events_toml["events"]["events_used"]) == set(events)
    lasif.api.set_up_iteration(comm, "4", remove_dirs=True)
    lasif.api.set_up_iteration(comm, "5", events=events[0])
    event_file = (
        comm.project.paths["iterations"] / f"ITERATION_5" / "events_used.toml"
    )
    events_toml = toml.load(event_file)
    assert set(events_toml["events"]["events_used"]) != set(events)
    lasif.api.set_up_iteration(comm, "5", remove_dirs=True)
