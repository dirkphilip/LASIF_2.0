#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import

import inspect
import pathlib

import os
import pytest
import shutil
import lasif.api

from lasif.components.project import Project
from ..testing_helpers import reset_matplotlib

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


# @pytest.mark.skip(reason="Domain plots are unstable")
# def test_event_plotting(comm):
#     """
#     Tests the plotting of all events.

#     The commands supports three types of plots: Beachballs on a map and depth
#     and time distribution histograms.
#     """
#     comm.visualizations.plot_events(plot_type="map")
#     images_are_identical("events_plot", comm.project.paths["root"], tol=30)


# @pytest.mark.skip(reason="Domain plots are unstable")
# def test_single_event_plot(comm):
#     """
#     Tests the plotting of a single event.
#     """
#     event = lasif.api.list_events(comm, output=True)[1]
#     lasif.api.plot_event(comm, event)
#     images_are_identical(
#         "event_2", comm.project.paths["root"], tol=30
#     )
#     lasif.api.plot_event(comm, event, weight_set_name="sw_1")
#     images_are_identical(
#         "event_2_station_weights", comm.project.paths["root"], tol=30
#     )


# def test_simple_raydensity(comm):
#     """
#     Test plotting a simple raydensity map.
#     """
#     comm.visualizations.plot_raydensity(save_plot=False)
#     # Use a low dpi to keep the test filesize in check.
#     images_are_identical("raydensity",
#                          comm.project.paths["root"],
#                          tol=30)


# def test_plot_all_rays(comm):
#     """
#     Test plotting a simple raydensity map.
#     """
#     lasif.api.plot_all_rays(comm, plot_stations=True)
#     # Use a low dpi to keep the test filesize in check.
#     images_are_identical(
#         "all_rays", comm.project.paths["root"], tol=30
#     )


# def test_complex_domain(comm):
#     """
#     Test plotting a simple raydensity map.
#     """
#     lasif.api.plot_domain(comm)
#     # Use a low dpi to keep the test filesize in check.
#     images_are_identical(
#         "complex_domain_no_convex_hull", comm.project.paths["root"], tol=30
#     )
#     lasif.api.plot_domain(comm, inner_boundary=True)
#     images_are_identical(
#         "complex_domain_with_convex_hull", comm.project.paths["root"], tol=30
#     )


# def test_simple_raydensity_with_stations(comm):
#     """
#     Test plotting a simple raydensity map with stations.
#     """
#     comm.visualizations.plot_raydensity(save_plot=False, plot_stations=True)
#     # Use a low dpi to keep the test filesize in check.
#     images_are_identical("simple_raydensity_plot_with_stations",
#                          comm.project.paths["root"], dpi=25,
#                          tol=30)
