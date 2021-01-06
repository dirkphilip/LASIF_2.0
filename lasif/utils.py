#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Some utility functionality.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2013
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
from collections import namedtuple
from geographiclib import geodesic
from fnmatch import fnmatch
import os
import obspy
import numpy as np
from tqdm import tqdm
import pathlib
from typing import Union
import pyasdf
import sys
import math

from lasif.exceptions import LASIFError, LASIFNotFoundError


def is_mpi_env():
    """
    Returns True if currently in an MPI environment.
    """
    from mpi4py import MPI

    if MPI.COMM_WORLD.size == 1 and MPI.COMM_WORLD.rank == 0:
        return False
    return True


def channel_in_parser(parser_object, channel_id, starttime, endtime):
    """
    Simply function testing if a given channel is part of a Parser object.

    Returns True or False.

    :type parser_object: :class:`obspy.io.xseed.Parser`
    :param parser_object: The parser object.
    """
    channels = parser_object.get_inventory()["channels"]
    for chan in channels:
        if not fnmatch(chan["channel_id"], channel_id):
            continue
        if starttime < chan["start_date"]:
            continue
        if chan["end_date"] and (endtime > chan["end_date"]):
            continue
        return True
    return False


def table_printer(header, data):
    """
    Pretty table printer.

    :type header: A list of strings
    :param data: A list of lists containing data items.
    """
    row_format = "{:>15}" * (len(header))
    print(row_format.format(*(["=" * 15] * len(header))))
    print(row_format.format(*header))
    print(row_format.format(*(["=" * 15] * len(header))))
    for row in data:
        print(row_format.format(*row))


def sizeof_fmt(num):
    """
    Handy formatting for human readable filesize.

    From http://stackoverflow.com/a/1094933/1657047
    """
    for x in ["bytes", "KB", "MB", "GB"]:
        if num < 1024.0 and num > -1024.0:
            return "%3.1f %s" % (num, x)
        num /= 1024.0
    return "%3.1f %s" % (num, "TB")


Point = namedtuple("Point", ["lat", "lng"])


def greatcircle_points(
    point_1: float,
    point_2: float,
    max_extension: float = None,
    max_npts: int = 3000,
):
    """
    Generator yielding a number points along a greatcircle from point_1 to
    point_2. Max extension is the normalization factor. If the distance between
    point_1 and point_2 is exactly max_extension, then 3000 points will be
    returned, otherwise a fraction will be returned.

    If max_extension is not given, the generator will yield exactly max_npts
    points.

    :param point_1: Point 1 to draw the greatcircle between
    :type point_1: float
    :param point_2: Point 2 to draw the greatcircle between
    :type point_2: float
    :param max_extension: Fraction of max_npts to return, defaults to None
    :type max_extension: float, optional
    :param max_npts: Maximum number of points to return, defaults to 3000
    :type max_npts: int, optional
    """
    point = geodesic.Geodesic.WGS84.Inverse(
        lat1=point_1.lat, lon1=point_1.lng, lat2=point_2.lat, lon2=point_2.lng
    )
    line = geodesic.Geodesic.WGS84.Line(
        point_1.lat, point_1.lng, point["azi1"]
    )

    if max_extension:
        npts = int((point["a12"] / float(max_extension)) * max_npts)
    else:
        npts = max_npts - 1
    if npts == 0:
        npts = 1
    for i in range(npts + 1):
        line_point = line.Position(i * point["s12"] / float(npts))
        yield Point(line_point["lat2"], line_point["lon2"])


def channel2station(value: str):
    """
    Helper function converting a channel id to a station id. Will not change
    a passed station id.

    :param value: The channel id as a string.
    :type value: str

    >>> channel2station("BW.FURT.00.BHZ")
    'BW.FURT'
    >>> channel2station("BW.FURT")
    'BW.FURT'
    """
    return ".".join(value.split(".")[:2])


def progress(count, total, status=""):
    """
    Simple progress bar.

    :param count: current count
    :param total: total of job
    :param status: reported status
    """
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = "=" * filled_len + "-" * (bar_len - filled_len)

    sys.stdout.write("[%s] %s%s ...%s\r" % (bar, percents, "%", status))
    sys.stdout.flush()


def select_component_from_stream(st: obspy.core.Stream, component: str):
    """
    Helper function selecting a component from a Stream an raising the proper
    error if not found.

    This is a bit more flexible then stream.select() as it works with single
    letter channels and lowercase channels.

    :param st: Obspy stream
    :type st: obspy.core.Stream
    :param component: Name of component of stream
    :type component: str
    """
    component = component.upper()
    component = [tr for tr in st if tr.stats.channel[-1].upper() == component]
    if not component:
        raise LASIFNotFoundError(
            "Component %s not found in Stream." % component
        )
    elif len(component) > 1:
        raise LASIFNotFoundError(
            "More than 1 Trace with component %s found "
            "in Stream." % component
        )
    return component[0]


def get_event_filename(event: object, prefix: str):
    """
    Helper function generating a descriptive event filename.

    :param event: The event object.
    :type event: object
    :param prefix: A prefix for the file, denoting e.g. the event catalog.
    :type prefix: str

    >>> from obspy import read_events
    >>> event = read_events()[0]
    >>> print(get_event_filename(event, "GCMT"))
    GCMT_event_KYRGYZSTAN-XINJIANG_BORDER_REG._Mag_4.4_2012-4-4-14.h5
    """
    from obspy.geodetics import FlinnEngdahl

    mag = event.preferred_magnitude() or event.magnitudes[0]
    org = event.preferred_origin() or event.origins[0]

    # Get the flinn_engdahl region for a nice name.
    fe = FlinnEngdahl()
    region_name = fe.get_region(org.longitude, org.latitude)
    region_name = region_name.replace(" ", "_")
    # Replace commas, as some file systems cannot deal with them.
    region_name = region_name.replace(",", "")

    return "%s_event_%s_Mag_%.1f_%s-%s-%s-%s.h5" % (
        prefix,
        region_name,
        mag.mag,
        org.time.year,
        org.time.month,
        org.time.day,
        org.time.hour,
    )


def write_custom_stf(stf_path: Union[pathlib.Path, str], comm: object):
    """
    Write the custom source-time-function specified in lasif config into
    the correct file

    :param stf_path: File path of the STF function
    :type stf_path: Union[pathlib.Path, str]
    :param comm: Lasif communicator
    :type comm: object
    """
    import h5py

    freqmax = 1.0 / comm.project.simulation_settings["minimum_period_in_s"]
    freqmin = 1.0 / comm.project.simulation_settings["maximum_period_in_s"]

    delta = comm.project.simulation_settings["time_step_in_s"]
    npts = comm.project.simulation_settings["number_of_time_steps"]

    stf_fct = comm.project.get_project_function("source_time_function")
    stf = comm.project.simulation_settings["source_time_function"]
    if stf == "bandpass_filtered_heaviside":
        stf = stf_fct(npts=npts, delta=delta, freqmin=freqmin, freqmax=freqmax)
    elif stf == "heaviside":
        stf = stf_fct(npts=npts, delta=delta)
    else:
        raise LASIFError(
            f"{stf} is not supported by lasif. Use either "
            f"bandpass_filtered_heaviside or heaviside."
        )

    stf_mat = np.zeros((len(stf), 6))
    # for i, moment in enumerate(moment_tensor):
    #     stf_mat[:, i] = stf * moment
    # Now we add the spatial weights into salvus
    for i in range(6):
        stf_mat[:, i] = stf

    heaviside_file_name = os.path.join(stf_path)
    f = h5py.File(heaviside_file_name, "w")

    source = f.create_dataset("source", data=stf_mat)
    source.attrs["dt"] = delta
    source.attrs["sampling_rate_in_hertz"] = 1 / delta
    # source.attrs["location"] = location
    source.attrs["spatial-type"] = np.string_("moment_tensor")
    # Start time in nanoseconds
    source.attrs["start_time_in_seconds"] = comm.project.simulation_settings[
        "start_time_in_s"
    ]

    f.close()


def load_receivers(comm: object, event: str):
    """
    Loads receivers which have already been written into a json file

    :param comm: LASIF communicator object
    :type comm: object
    :param event: The name of the event for which to generate the
        input files.
    :type event: str
    """
    import json

    filename = (
        comm.project.paths["salvus_files"]
        / "RECEIVERS"
        / event
        / "receivers.json"
    )
    if not os.path.exists(filename):
        raise LASIFNotFoundError()
    with open(filename, "r") as fh:
        receivers = json.load(fh)
    return receivers


def place_receivers(comm: object, event: str, write_to_file: bool = False):
    """
    Generates a list of receivers with the required fields
    for a salvus simulation.

    :param comm: LASIF communicator object
    :type comm: object
    :param event: The name of the event for which to generate the
        input files.
    :type event: str
    :param write_to_file: Writes receivers to json file, allows for faster
        loading next time receivers are used, defaults to False
    :type write_to_file: bool, optional
    """

    event_stations = comm.query.get_all_stations_for_event(event)
    recs = []
    for station, info in event_stations.items():
        net, sta = station.split(".")
        rec_dict = {
            "network-code": net,
            "station-code": sta,
            "medium": "solid",
            "latitude": elliptic_to_geocentric_latitude(info["latitude"]),
            "longitude": info["longitude"],
        }
        recs.append(rec_dict)

    print(f"Wrote {len(recs)} receivers into a list of dictionaries")
    if write_to_file:
        import json

        filename = (
            comm.project.paths["salvus_files"]
            / "RECEIVERS"
            / event
            / "receivers.json"
        )
        if not os.path.exists(filename.parent):
            os.makedirs(filename.parent)
        with open(filename, "w+") as fh:
            json.dump(recs, fh)
        print(f"Wrote {len(recs)} receivers into file")
    return recs


def prepare_source(comm: object, event: str, iteration: str):
    """
    Gather important information on the source

    :param comm: Project communicator
    :type comm: object
    :param event: Name of event
    :type event: str
    :param iteration: Name of iteration for simulation
    :type iteration: str
    """
    import h5py

    iteration = comm.iterations.get_long_iteration_name(iteration)

    # Check if STF exists
    write_stf = True
    stf_path = os.path.join(
        comm.project.paths["salvus_files"], iteration, "stf.h5"
    )
    if os.path.exists(path=stf_path):
        with h5py.File(stf_path, "r") as f:
            if "source" in f:
                write_stf = False

    if write_stf:
        write_custom_stf(stf_path, comm)

    event_info = comm.events.get(event)

    source = [
        {
            "latitude": event_info["latitude"],
            "longitude": event_info["longitude"],
            "depth_in_m": event_info["depth_in_km"] * 1000.0,
            "mrr": event_info["m_rr"],
            "mtt": event_info["m_tt"],
            "mpp": event_info["m_pp"],
            "mtp": event_info["m_tp"],
            "mrp": event_info["m_rp"],
            "mrt": event_info["m_rt"],
            "stf": "Custom",
            "stf_file": stf_path,
            "dataset": "source",
        }
    ]

    return source


def process_two_files_without_parallel_output(
    ds: pyasdf.ASDFDataSet,
    other_ds: pyasdf.ASDFDataSet,
    process_function,
    traceback_limit=3,
):
    import traceback
    import sys
    from mpi4py import MPI

    """
    Process data in two data sets. Based on pyasdf

    This is mostly useful for comparing data in two data sets in any
    number of scenarios. It again takes a function and will apply it on
    each station that is common in both data sets.

    Can only be run with MPI.

    :param ds: The dataset to process
    :type ds: pyasdf.ASDFDataSet
    :param other_ds: The data set to compare to.
    :type other_ds: pyasdf.ASDFDataSet
    :param process_function: The processing function takes two
        parameters: The station group from this data set and
        the matching station group from the other data set.
    :param traceback_limit: The length of the traceback printed if an
        error occurs in one of the workers, defaults to 3
    :type traceback_limit: int, optional
    :return: A dictionary for each station with gathered values. Will
        only be available on rank 0.
    """

    # Collect the work that needs to be done on rank 0.
    if MPI.COMM_WORLD.rank == 0:

        def split(container, count):
            """
            Simple function splitting a container into equal length
            chunks.
            Order is not preserved but this is potentially an advantage
            depending on the use case.
            """
            return [container[_i::count] for _i in range(count)]

        this_stations = set(ds.waveforms.list())
        other_stations = set(other_ds.waveforms.list())

        # Usable stations are those that are part of both.
        usable_stations = list(this_stations.intersection(other_stations))
        total_job_count = len(usable_stations)
        jobs = split(usable_stations, MPI.COMM_WORLD.size)
    else:
        jobs = None

    # Scatter jobs.
    jobs = MPI.COMM_WORLD.scatter(jobs, root=0)

    # Dictionary collecting results.
    results = {}

    for _i, station in enumerate(jobs):

        if MPI.COMM_WORLD.rank == 0:
            print(
                " -> Processing approximately task %i of %i ..."
                % ((_i * MPI.COMM_WORLD.size + 1), total_job_count),
                flush=True,
            )
        try:
            result = process_function(
                getattr(ds.waveforms, station),
                getattr(other_ds.waveforms, station),
            )
        except Exception:
            # If an exception is raised print a good error message
            # and traceback to help diagnose the issue.
            msg = (
                "\nError during the processing of station '%s' "
                "on rank %i:"
                % (
                    station,
                    MPI.COMM_WORLD.rank,
                )
            )

            # Extract traceback from the exception.
            exc_info = sys.exc_info()
            stack = traceback.extract_stack(limit=traceback_limit)
            tb = traceback.extract_tb(exc_info[2])
            full_tb = stack[:-1] + tb
            exc_line = traceback.format_exception_only(*exc_info[:2])
            tb = (
                "Traceback (At max %i levels - most recent call "
                "last):\n" % traceback_limit
            )
            tb += "".join(traceback.format_list(full_tb))
            tb += "\n"
            tb += "".join(
                _i.decode(errors="ignore") if hasattr(_i, "decode") else _i
                for _i in exc_line
            )

            # These potentially keep references to the HDF5 file
            # which in some obscure way and likely due to
            # interference with internal HDF5 and Python references
            # prevents it from getting garbage collected. We
            # explicitly delete them here and MPI can finalize
            # afterwards.
            del exc_info
            del stack

            print(msg, flush=True)
            print(tb, flush=True)
        else:
            # print("Else!", flush=True)
            results[station] = result
    # barrier but better be safe than sorry.
    MPI.COMM_WORLD.Barrier()
    if MPI.COMM_WORLD.rank == 0:
        print("All ranks finished", flush=True)
    return results


def normalize_coordinates(
    x: Union[float, np.array],
    y: Union[float, np.array],
    z: Union[float, np.array],
):
    """
    Take coordinates which are defined on some sphere to being defined on
    the unit sphere

    :param x: X coordinate
    :type x: Union[float, np.array]
    :param y: Y coordinate
    :type y: Union[float, np.array]
    :param z: Z coordinate
    :type z: Union[float, np.array]
    """
    curr_rad = np.max(np.sqrt(np.square(x) + np.square(y) + np.square(z)))
    x *= 1.0 / curr_rad
    y *= 1.0 / curr_rad
    z *= 1.0 / curr_rad
    return x, y, z


def elliptic_to_geocentric_latitude(
    lat: float, axis_a: float = 6378137.0, axis_b: float = 6356752.314245
) -> float:
    """
    Convert latitudes defined on an ellipsoid to a geocentric one.
    Based on Salvus Seismo

    :param lat: Latitude to convert
    :type lat: float
    :param axis_a: Major axis of planet in m, defaults to 6378137.0
    :type axis_a: float, optional
    :param axis_b: Minor axis of planet in m, defaults to 6356752.314245
    :type axis_b: float, optional
    :return: Converted latitude
    :rtype: float

    >>> elliptic_to_geocentric_latitude(0.0)
    0.0
    >>> elliptic_to_geocentric_latitude(90.0)
    90.0
    >>> elliptic_to_geocentric_latitude(-90.0)
    -90.0
    """
    _f = (axis_a - axis_b) / axis_a
    E_2 = 2 * _f - _f ** 2
    if abs(lat) < 1e-6 or abs(lat - 90) < 1e-6 or abs(lat + 90) < 1e-6:
        return lat

    return math.degrees(math.atan((1 - E_2) * math.tan(math.radians(lat))))


class Receiver(object):
    from lasif.utils import elliptic_to_geocentric_latitude

    def __init__(self, network, station, latitude, longitude, depth_in_m=0.0):
        self.latitude = float(latitude)
        self.longitude = float(longitude)
        self.depth_in_m = float(depth_in_m)
        if self.depth_in_m <= 0.0:
            self.depth_in_m = 0.0
        self.network = network
        self.network = self.network.strip()
        assert len(self.network) <= 2
        self.station = station
        self.station = self.station.strip()

    @staticmethod
    def parse(obj: object, network_code: str = None):
        """
        Based on the receiver parser of Salvus Seismo by Lion Krischer and
        Martin van Driel.
        It aims to parse an obspy inventory object into a list of
        Receiver objects
        Maybe we want to remove this elliptic to geocentric latitude thing
        at some point. Depends on what solver wants.

        :param obj: Obspy inventory object
        :type obj: object
        :param network_code: Used to keep information about network when
            at the station level, defaults to None
        :type network_code: str, optional
        """
        receivers = []

        if isinstance(obj, obspy.core.inventory.Inventory):
            for network in obj:
                receivers.extend(Receiver.parse(network))
            return receivers

        elif isinstance(obj, obspy.core.inventory.Network):
            for station in obj:
                receivers.extend(
                    Receiver.parse(station, network_code=obj.code)
                )
            return receivers

        elif isinstance(obj, obspy.core.inventory.Station):
            # If there are no channels, use the station coordinates
            if not obj.channels:
                return [
                    Receiver(
                        latitude=elliptic_to_geocentric_latitude(obj.latitude),
                        longitude=obj.longitude,
                        network=network_code,
                        station=obj.code,
                    )
                ]
            # Otherwise we use channel information
            else:
                coords = set(
                    (_i.latitude, _i.longitude, _i.depth)
                    for _i in obj.channels
                )
                if len(coords) != 1:
                    raise LASIFError(
                        f"Coordinates of channels of station "
                        f"{network_code}.{obj.code} are not identical"
                    )
                coords = coords.pop()
                return [
                    Receiver(
                        latitude=elliptic_to_geocentric_latitude(coords[0]),
                        longitude=coords[1],
                        depth_in_m=coords[2],
                        network=network_code,
                        station=obj.code,
                    )
                ]
