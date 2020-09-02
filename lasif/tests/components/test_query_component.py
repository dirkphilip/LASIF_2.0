#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import

import inspect
import pathlib

import os
import pytest
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


def test_get_all_stations_for_event(comm):
    events = lasif.api.list_events(comm, output=True)
    stations = comm.query.get_all_stations_for_event(
        event_name=events[1], list_only=True
    )
    assert set(stations) == set(["GE.TIRR", "II.KIV", "MN.IDI", "YD.4F14"])

    stations = comm.query.get_all_stations_for_event(
        event_name=events[1], list_only=False
    )
    assert stations["GE.TIRR"] == {
        "elevation_in_m": 77.0,
        "latitude": 44.458099,
        "longitude": 28.4128,
    }


def test_get_coordinates_for_station(comm):
    events = lasif.api.list_events(comm, output=True)
    station = "GE.TIRR"
    coordinates = comm.query.get_coordinates_for_station(events[1], station)
    assert coordinates == {
        "elevation_in_m": 77.0,
        "latitude": 44.458099,
        "longitude": 28.4128,
    }


def test_get_stations_for_all_events(comm):
    results = comm.query.get_stations_for_all_events()
    events = lasif.api.list_events(comm, output=True)
    should_be = {}
    for event in events:
        should_be[event] = comm.query.get_all_stations_for_event(
            event, list_only=True
        )

    assert results == should_be
