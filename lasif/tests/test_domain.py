#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test suite for the domain definitions in LASIF.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2014
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
from __future__ import absolute_import

# from unittest import mock

import inspect
import os
import pathlib
import shutil
import toml
from lasif.domain import HDF5Domain, SimpleDomain

# from lasif.domain import HDF5Domain
from lasif.scripts import lasif_cli
from lasif.tests.testing_helpers import reset_matplotlib

# images_are_identical

import pytest

from lasif.components.project import Project

# Get a list of all available commands.
CMD_LIST = [
    key.replace("lasif_", "")
    for (key, value) in lasif_cli.__dict__.items()
    if (key.startswith("lasif_") and callable(value))
]


@pytest.fixture()
def comm(tmpdir):
    proj_dir = os.path.join(
        os.path.dirname(
            os.path.dirname(
                os.path.abspath(inspect.getfile(inspect.currentframe()))
            )
        ),
        "tests",
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
        "tests",
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


def setup_function(function):
    """
    Reset matplotlib.
    """
    reset_matplotlib()


# def test_global_domain_check(comm):
#     """
#     Check whether the domain is assumed to be global. Should not be global
#     according to mesh used in example project
#     """

#     globe = comm.project.domain.is_global_domain()
#     assert not globe


# def test_global_domain_with_a_globe():
#     """
#     Check whether global check works for a global mesh
#     """
#     path_to_mesh = os.path.join(os.path.dirname(os.path.dirname(
#         os.path.abspath(inspect.getfile(inspect.currentframe())))),
#         "tests", "data", "example_project", "MODELS", "Globalmesh.e")

#     from lasif.domain import ExodusDomain

#     global_domain = ExodusDomain(path_to_mesh, 7)

#     assert global_domain.is_global_domain()
#     assert global_domain.min_lat == -90.0
#     assert global_domain.max_lat == 90.0
#     assert global_domain.min_lon == -180.0
#     assert global_domain.max_lon == 180.0
#     assert global_domain.side_set_names == ['r0', 'r1']


def test_point_in_domain(comm):
    """
    Check whether points exist inside domain or not.
    """
    assert isinstance(comm.project.domain, HDF5Domain)
    event_list = comm.events.list()
    for event_name in event_list:
        event_dir = comm.events.get(event_name)
        in_domain = comm.project.domain.point_in_domain(
            longitude=event_dir["longitude"],
            latitude=event_dir["latitude"],
            depth=event_dir["depth_in_km"] * 1000.0,
        )
        # assert event_dir["depth_in_km"] == 12.0
        assert in_domain

    # Make conditions to let the test fail due to depth.
    long = event_dir["longitude"]
    lat = event_dir["latitude"]
    depth = 3500.0 * 1000
    in_domain = comm.project.domain.point_in_domain(long, lat, depth)
    assert not in_domain

    # Fail by longitude
    long = 60.0
    depth = event_dir["depth_in_km"] * 1000
    in_domain = comm.project.domain.point_in_domain(long, lat, depth)
    assert not in_domain

    # Fail by latitude
    long = event_dir["longitude"]
    lat = 62.0
    in_domain = comm.project.domain.point_in_domain(long, lat, depth)
    assert not in_domain

    # Fail by point inside minmax latlon but not in complex domain
    lon = 32.155
    lat = 43.24
    depth = 300.0
    in_domain = comm.project.domain.point_in_domain(lon, lat, depth)
    assert not in_domain


def test_2D_check(comm):
    """
    Make sure domain checks in 2D are working.
    """
    event_list = comm.events.list()
    for event_name in event_list:
        event_dir = comm.events.get(event_name)
        in_domain = comm.project.domain.point_in_domain(
            longitude=event_dir["longitude"], latitude=event_dir["latitude"],
        )
        assert in_domain

    # Fail by longitude
    long = 90.0
    lat = event_dir["latitude"]
    in_domain = comm.project.domain.point_in_domain(long, lat)
    assert not in_domain

    # Fail by latitude
    long = event_dir["longitude"]
    lat = 82.0
    in_domain = comm.project.domain.point_in_domain(long, lat)
    assert not in_domain


@pytest.mark.parametrize("latitude", [-70.0, -30.0, 0.0, 30.0, 70.0])
@pytest.mark.parametrize("longitude", [-160.0, -80.0, 0.0, 80.0, 160.0])
@pytest.mark.parametrize("depth", [10.0, 40.0, 100.0])
def test_point_in_global_domain(latitude, longitude, depth):
    """
    Use the Global mesh to make sure all possible points are inside domain.
    This one is a bit slow and should maybe be changed by having fewer points.
    """
    path_to_mesh = os.path.join(
        os.path.dirname(
            os.path.dirname(
                os.path.abspath(inspect.getfile(inspect.currentframe()))
            )
        ),
        "tests",
        "data",
        "example_project",
        "MODELS",
        "Global.h5",
    )

    from lasif.domain import HDF5Domain

    global_domain = HDF5Domain(path_to_mesh, 7)

    assert global_domain.point_in_domain(longitude, latitude, depth * 1000.0)


def test_simple_domain(comm_simple):
    """
    While Salvus is not used, different kind of domains are used.
    That will be tested here.
    """

    assert isinstance(comm_simple.project.domain, SimpleDomain)


@pytest.mark.parametrize("latitude", [50.0, 30.0, 25.0])
@pytest.mark.parametrize("longitude", [-1.0, 30.0, 50.0])
@pytest.mark.parametrize("depth", [10.0, 100.0])
def test_point_in_simple_domain(comm_simple, latitude, longitude, depth):

    assert comm_simple.project.domain.point_in_domain(
        longitude, latitude, depth * 1000.0
    )


def test_point_out_of_simple_domain(comm_simple):

    latitude = 30.0
    longitude = 30.0
    depth = 100.0

    # fail by longitude
    longitude_f = 80.0
    assert not comm_simple.project.domain.point_in_domain(
        longitude_f, latitude, depth * 1000.0
    )

    # fail by latitude
    latitude_f = 80.0
    assert not comm_simple.project.domain.point_in_domain(
        longitude, latitude_f, depth * 1000.0
    )

    # fail by depth
    depth_f = 700.0
    assert not comm_simple.project.domain.point_in_domain(
        longitude, latitude, depth_f * 1000.0
    )
