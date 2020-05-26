#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import

import inspect
import pathlib

import os
import pytest
import shutil
import lasif.api
from lasif.exceptions import LASIFError
import toml

from lasif.components.project import Project
from ..testing_helpers import reset_matplotlib, images_are_identical
from matplotlib.testing.decorators import image_comparison

# from ..testing_helpers import images_are_identical
# The importing of matplotlib.testing.compare.compare_images is not
# working for some reason that's why these tests are not functional currently


def setup_function(function):
    """
    Reset matplotlib.
    """
    reset_matplotlib()


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
    os.chdir(os.path.abspath(folder_path))

    return project.comm


@pytest.fixture()
def comm_simple(tmpdir):
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
    toml_file = folder_path / "lasif_config.toml"
    config = toml.load(toml_file)
    config["lasif_project"]["solver_used"] = "other"
    with open(toml_file, "w") as fh:
        toml.dump(config, fh)

    project = Project(project_root_path=folder_path, init_project=False)
    os.chdir(os.path.abspath(folder_path))

    return project.comm


@pytest.mark.filterwarnings(
    "ignore: can't resolve package from __spec__ or __package__"
)
def test_plot_domain(comm):
    comm.visualizations.plot_domain(inner_boundary=False)
    images_are_identical(
        "domain", comm.project.paths["root"], tol=30, dpi=200,
    )


@pytest.mark.filterwarnings(
    "ignore: can't resolve package from __spec__ or __package__"
)
def test_event_plotting(comm):
    """
    Tests the plotting of all events.

    The commands supports three types of plots: Beachballs on a map and depth
    and time distribution histograms.
    """
    comm.visualizations.plot_events()
    images_are_identical(
        "events_plot", comm.project.paths["root"], tol=30, dpi=200
    )
    comm.visualizations.plot_events(inner_boundary=True)
    images_are_identical(
        "events_with_inner_boundary",
        comm.project.paths["root"],
        tol=30,
        dpi=200,
    )
    comm.visualizations.plot_events(iteration="2")
    images_are_identical(
        "events_2", comm.project.paths["root"], tol=30, dpi=200,
    )


@pytest.mark.filterwarnings(
    "ignore: can't resolve package from __spec__ or __package__"
)
def test_single_event_plot(comm):
    """
    Tests the plotting of a single event.
    """
    event = lasif.api.list_events(comm, output=True)[1]
    lasif.api.plot_event(comm, event)
    images_are_identical(
        "event_2", comm.project.paths["root"], tol=30, dpi=200
    )
    lasif.api.plot_event(comm, event, weight_set_name="sw_1")
    images_are_identical(
        "event_2_station_weights", comm.project.paths["root"], tol=30, dpi=200
    )


@pytest.mark.filterwarnings(
    "ignore: can't resolve package from __spec__ or __package__"
)
def test_plot_raydensity(comm):
    """
    Test plotting a simple raydensity map.
    """
    comm.visualizations.plot_raydensity(save_plot=False, plot_stations=True)
    # Use a low dpi to keep the test filesize in check.
    images_are_identical(
        "raydensity", comm.project.paths["root"], tol=30, dpi=200
    )


@pytest.mark.filterwarnings(
    "ignore: can't resolve package from __spec__ or __package__"
)
def test_plot_all_rays(comm):
    """
    Test plotting a simple raydensity map.
    """
    lasif.api.plot_all_rays(comm, plot_stations=True)
    # Use a low dpi to keep the test filesize in check.
    images_are_identical(
        "all_rays", comm.project.paths["root"], tol=30, dpi=200
    )


@pytest.mark.filterwarnings(
    "ignore: can't resolve package from __spec__ or __package__"
)
def test_plot_windows(comm):
    event = lasif.api.list_events(".")[0]
    comm.visualizations.plot_windows(
        event=event, window_set_name="A", show=False
    )
    images_are_identical(
        "windows", comm.project.paths["root"], tol=30, dpi=200
    )


@pytest.mark.filterwarnings(
    "ignore: can't resolve package from __spec__ or __package__"
)
def test_plot_window_statistics(comm):
    events = lasif.api.list_events(".")
    comm.visualizations.plot_window_statistics("A", events=events, show=False)
    images_are_identical(
        "window_statistics", comm.project.paths["root"], tol=30, dpi=200
    )


@pytest.mark.filterwarnings(
    "ignore: can't resolve package from __spec__ or __package__"
)
def test_simple_domain_plotting(comm_simple):
    comm_simple.visualizations.plot_domain(inner_boundary=False)
    images_are_identical(
        "simple_domain", comm_simple.project.paths["root"], tol=30, dpi=200
    )
    with pytest.raises(LASIFError) as excinfo:
        comm_simple.visualizations.plot_domain(inner_boundary=True)
        assert "Inner boundary is not" in str(excinfo.value)
