from __future__ import absolute_import

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


def test_generate_restrictions(comm):
    event = lasif.api.list_events(comm, output=True)[0]
    event_time = comm.events.get(event)["origin_time"]
    ds = comm.project.lasif_config["download_settings"]
    starttime = event_time - ds["seconds_before_event"]
    endtime = event_time + ds["seconds_after_event"]
    restrictions = comm.downloads.generate_restrictions(starttime, endtime, ds)
    assert restrictions.starttime == starttime
    assert restrictions.endtime == endtime
    assert restrictions.station_starttime == starttime - 86400 * 1
    assert restrictions.station_endtime == endtime + 86400 * 1
    assert restrictions.reject_channels_with_gaps
    assert restrictions.minimum_length == 0.95
