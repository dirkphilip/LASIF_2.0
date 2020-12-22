#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Visualization scripts.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2013

:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
from itertools import chain
from matplotlib import cm
import matplotlib.pyplot as plt
import cartopy as cp

# from cartopy.mpl.geoaxes import GeoAxes
import numpy as np
from obspy.signal.tf_misfit import plot_tfr
from lasif.exceptions import LASIFError
import cmasher as cmr
from typing import Union, List, Dict, Tuple


def project_points(
    projection: cp.crs.Projection,
    lon: Union[np.ndarray, float],
    lat: Union[np.ndarray, float],
):
    """
    Define the correct projection function depending on name of projection

    :param projection: Projection to be used
    :type projection: cp.crs.Projection
    :param lon: Longitude coordinate
    :type lon: Union[np.ndarray, float]
    :param lat: Latitude coordinate
    :type lat: Union[np.ndarray, float]
    :return: projected lon, lat points
    :rtype: np.ndarray, np.ndarray
    """
    import pyproj

    proj_dict = projection.proj4_params

    # projection = pyproj.crs.CRS.from_dict(proj_dict)
    event_loc = pyproj.Proj(proj_dict, preserve_units=True)
    x, y = event_loc(lon, lat)
    if isinstance(x, list):
        x = np.array(x)
        y = np.array(y)
    return x, y


def xy_to_lonlat(
    x: Union[float, np.ndarray],
    y: Union[float, np.ndarray],
    projection: cp.crs.Projection,
):
    """
    Change x and y to latitude and longitude, based on Earth radius

    :param x: X coordinate in the correct projection, but in meters
    :type x: Union[float, np.ndarray]t
    :param y: Y coordinate in the correct projection, but in meters
    :type y: Union[float, np.ndarray]t
    :param projection: Cartopy projection object
    :type projection: cp.crs.Projection
    """
    # if projection.proj4_params["proj"] == "moll":
    # return x, y
    earth_radius = 6371000.0
    radians_to_degrees = 180.0 / np.pi
    lat = (y / earth_radius) * radians_to_degrees
    if "lat_0" in projection.proj4_params.keys():
        center_lat = projection.proj4_params["lat_0"]
    else:
        center_lat = 0.0
    r = earth_radius * np.cos((lat - center_lat) / radians_to_degrees)
    lon = (x / r) * radians_to_degrees
    return lon, lat


def plot_events(
    events: List[object], map_object,
):
    """
    Plot event stars on a map

    :param events: Event information
    :type events: List[object]
    :param map_object: The already made map object from the domain component
    :type map_object: cartopy.mpl.geoaxes.GeoAxes
    """

    for event in events:

        plotted_events = map_object.scatter(
            x=event["longitude"],
            y=event["latitude"],
            zorder=22,
            marker="*",
            c="yellow",
            transform=cp.crs.PlateCarree(),
            s=180,
            edgecolors="black",
        )

    return plotted_events


def plot_raydensity(
    map_object,
    station_events: List[Tuple[dict, dict]],
    domain: object,
    projection: cp.crs.Projection,
):
    """
    Create a ray-density plot for all events and all stations.

    This function is potentially expensive and will use all CPUs available.
    Does require geographiclib to be installed.

    :param map_object: The cartopy domain plot object
    :type map_object: cp.mpl.geoaxes.GeoAxes
    :param station_events: A list of tuples with two dictionaries
    :type station_events: List[Tuple[dict, dict]]
    :param domain: An object with the domain plot
    :type domain: object
    :param projection: cartopy projection object
    :type projection: cp.crs.Projection
    """
    import ctypes as C
    from lasif.tools.great_circle_binner import GreatCircleBinner
    from lasif.utils import Point
    import multiprocessing
    import progressbar
    from scipy.stats import scoreatpercentile

    # Merge everything so that a list with coordinate pairs is created. This
    # list is then distributed among all processors.
    station_event_list = []
    for event, stations in station_events:

        e_point = Point(event["latitude"], event["longitude"])
        for station in stations.values():

            p = Point(station["latitude"], station["longitude"])
            station_event_list.append((e_point, p))

    circle_count = len(station_event_list)

    # The granularity of the latitude/longitude discretization for the
    # raypaths. Attempt to get a somewhat meaningful result in any case.
    if circle_count < 1000:
        lat_lng_count = 1000
    elif circle_count < 10000:
        lat_lng_count = 2000
    else:
        lat_lng_count = 3000

    cpu_count = multiprocessing.cpu_count()

    def to_numpy(raw_array, dtype, shape):
        data = np.frombuffer(raw_array.get_obj())
        data.dtype = dtype
        return data.reshape(shape)

    print(
        "\nLaunching %i great circle calculations on %i CPUs..."
        % (circle_count, cpu_count)
    )

    widgets = [
        "Progress: ",
        progressbar.Percentage(),
        progressbar.Bar(),
        "",
        progressbar.ETA(),
    ]
    pbar = progressbar.ProgressBar(
        widgets=widgets, maxval=circle_count
    ).start()

    def great_circle_binning(
        sta_evs, bin_data_buffer, bin_data_shape, lock, counter
    ):
        new_bins = GreatCircleBinner(
            domain.min_lat,
            domain.max_lat,
            lat_lng_count,
            domain.min_lon,
            domain.max_lon,
            lat_lng_count,
        )
        for event, station in sta_evs:
            with lock:
                counter.value += 1
            if not counter.value % 25:
                pbar.update(counter.value)
            new_bins.add_greatcircle(event, station)

        bin_data = to_numpy(bin_data_buffer, np.uint32, bin_data_shape)
        with bin_data_buffer.get_lock():
            bin_data += new_bins.bins

    # Split the data in cpu_count parts.
    def chunk(seq, num):
        avg = len(seq) / float(num)
        out = []
        last = 0.0
        while last < len(seq):
            out.append(seq[int(last) : int(last + avg)])
            last += avg
        return out

    chunks = chunk(station_event_list, cpu_count)

    # One instance that collects everything.
    collected_bins = GreatCircleBinner(
        domain.min_lat,
        domain.max_lat,
        lat_lng_count,
        domain.min_lon,
        domain.max_lon,
        lat_lng_count,
    )

    # Use a multiprocessing shared memory array and map it to a numpy view.
    collected_bins_data = multiprocessing.Array(
        C.c_uint32, collected_bins.bins.size
    )
    collected_bins.bins = to_numpy(
        collected_bins_data, np.uint32, collected_bins.bins.shape
    )

    # Create, launch and join one process per CPU. Use a shared value as a
    # counter and a lock to avoid race conditions.
    processes = []
    lock = multiprocessing.Lock()
    counter = multiprocessing.Value("i", 0)
    for _i in range(cpu_count):
        processes.append(
            multiprocessing.Process(
                target=great_circle_binning,
                args=(
                    chunks[_i],
                    collected_bins_data,
                    collected_bins.bins.shape,
                    lock,
                    counter,
                ),
            )
        )
    for process in processes:
        process.start()
    for process in processes:
        process.join()

    pbar.finish()

    stations = chain.from_iterable(
        (_i[1].values() for _i in station_events if _i[1])
    )
    # Remove duplicates
    stations = [(_i["latitude"], _i["longitude"]) for _i in stations]
    stations = set(stations)
    title = "%i Events, %i unique raypaths, " "%i unique stations" % (
        len(station_events),
        circle_count,
        len(stations),
    )
    plt.title(title, size="xx-large")

    data = collected_bins.bins.transpose()

    if data.max() >= 10:
        data = np.log10(np.clip(data, a_min=0.5, a_max=data.max()))
        data[data >= 0.0] += 0.1
        data[data < 0.0] = 0.0
        max_val = scoreatpercentile(data.ravel(), 99)
    else:
        max_val = data.max()

    cmap = cm.get_cmap("gist_heat")
    cmap._init()
    cmap._lut[:120, -1] = np.linspace(0, 1.0, 120) ** 2

    lngs, lats = collected_bins.coordinates
    ln, la = project_points(projection, lngs, lats)

    map_object.pcolormesh(
        ln, la, data, cmap=cmap, vmin=0, vmax=max_val, zorder=10
    )
    # Draw the coastlines so they appear over the rays. Otherwise things are
    # sometimes hard to see.
    map_object.add_feature(cp.feature.COASTLINE, zorder=13)
    map_object.add_feature(cp.feature.BORDERS, linestyle=":", zorder=13)


def plot_all_rays(map_object, station_events: List[Tuple[dict, dict]]):
    """
    Plot all the rays in the project on a plot. Each event has a different
    color of rays.

    :param map_object: Cartopy plot object
    :type map_object: cp.mpl.geoaxes.GeoAxes
    :param station_events: List of tuples with two dictionaries with
        event and station information
    :type station_events: List[Tuple[dict, dict]]
    """
    from tqdm import tqdm

    # Maybe not make color completely random, I might need to use a colormap
    c = np.random.rand(len(station_events), 3)
    # print(station_events)
    i = 0
    for event, stations in station_events:
        for station in tqdm(stations.values()):
            map_object.plot(
                [event["longitude"], station["longitude"]],
                [event["latitude"], station["latitude"]],
                c=c[i],
                transform=cp.crs.PlateCarree(),
                alpha=0.8,
                linewidth=0.6,
                zorder=19,
            )
        i += 1
        print(f"{i} done from {len(station_events)}\n")


def plot_stations_for_event(
    map_object,
    station_dict: Dict[str, Union[str, float]],
    event_info: Dict[str, Union[str, float]],
    color: str = "green",
    alpha: float = 1.0,
    raypaths: bool = True,
    weight_set: str = None,
    plot_misfits: bool = False,
    print_title: bool = True,
):
    """
    Plot all stations for one event

    :param map_object: Cartopy plotting object
    :type map_object: cp.mpl.geoaxes.GeoAxes
    :param station_dict: Dictionary with station information
    :type station_dict: Dict[str, Union[str, float]]
    :param event_info: Dictionary with event information
    :type event_info: Dict[str, Union[str, float]]
    :param color: Color to plot stations with, defaults to "green"
    :type color: str, optional
    :param alpha: How transparent the stations are, defaults to 1.0
    :type alpha: float, optional
    :param raypaths: Should raypaths be plotted?, defaults to True
    :type raypaths: bool, optional
    :param weight_set: Do we colorcode stations with their respective
        weights, defaults to None
    :type weight_set: str, optional
    :param plot_misfits: Color code stations with their respective
        misfits, defaults to False
    :type plot_misfits: bool, optional
    :param print_title: Have a title on the figure, defaults to True
    """
    import re

    # Check inputs:
    if weight_set and plot_misfits:
        raise LASIFError("Can't plot both weight set and misfit")

    # Loop as dicts are unordered.
    lngs = []
    lats = []
    station_ids = []

    for key, value in station_dict.items():
        lngs.append(value["longitude"])
        lats.append(value["latitude"])
        station_ids.append(key)

    event = event_info["event_name"]
    if weight_set:
        # If a weight set is specified, stations will be color coded.
        weights = []
        for id in station_ids:
            weights.append(
                weight_set.events[event]["stations"][id]["station_weight"]
            )
        cmap = cmr.heat
        stations = map_object.scatter(
            lngs,
            lats,
            c=weights,
            cmap=cmap,
            s=35,
            marker="v",
            alpha=alpha,
            zorder=5,
            transform=cp.crs.PlateCarree(),
        )
        cbar = plt.colorbar(stations)
        cbar.ax.set_ylabel("Station Weights", rotation=-90)

    elif plot_misfits:
        misfits = [station_dict[x]["misfit"] for x in station_dict.keys()]
        cmap = cmr.heat
        # cmap = cm.get_cmap("seismic")
        stations = map_object.scatter(
            lngs,
            lats,
            c=misfits,
            cmap=cmap,
            s=35,
            marker="v",
            alpha=alpha,
            zorder=5,
            transform=cp.crs.PlateCarree(),
        )
        # from mpl_toolkits.axes_grid1 import make_axes_locatable

        # divider = make_axes_locatable(map_object)
        # cax = divider.append_axes("right", "5%", pad="3%")
        # im_ratio = map_object.shape[0] / map_object.shape[1]
        cbar = plt.colorbar(stations)
        cbar.ax.set_ylabel("Station misfits", rotation=-90)
        # plt.tight_layout()

    else:
        stations = map_object.scatter(
            lngs,
            lats,
            color=color,
            s=35,
            marker="v",
            alpha=alpha,
            zorder=5,
            transform=cp.crs.PlateCarree(),
        )
        # Setting the picker overwrites the edgecolor attribute on certain
        # matplotlib and basemap versions. Fix it here.
        stations._edgecolors = np.array([[0.0, 0.0, 0.0, 1.0]])
        stations._edgecolors_original = "black"

    # Plot the ray paths.
    if raypaths:
        for sta_lng, sta_lat in zip(lngs, lats):
            map_object.plot(
                [event_info["longitude"], sta_lng],
                [event_info["latitude"], sta_lat],
                lw=2,
                alpha=0.3,
                transform=cp.crs.PlateCarree(),
            )
    title = "Event in %s, at %s, %.1f Mw, with %i stations." % (
        event_info["region"],
        re.sub(r":\d{2}\.\d{6}Z", "", str(event_info["origin_time"])),
        event_info["magnitude"],
        len(station_dict),
    )

    weights_title = f"Event in {event_info['region']}. Station Weights"
    if print_title:
        if weight_set is not None:
            map_object.set_title(weights_title, size="large")
        elif plot_misfits:
            misfit_title = (
                f"Event in {event_info['region']}. "
                f"Total misfit: '%.2f'" % (np.sum(misfits))
            )
            map_object.set_title(misfit_title, size="large")
        else:
            map_object.set_title(title, size="large")

    return stations


def plot_all_stations(map_object, event_stations: List[Tuple[dict, dict]]):
    """
    Add all stations to a map object

    :param map_object: A cartopy map object
    :type map_object: cp.mpl.geoaxes.GeoAxes
    :param event_stations: a list of dictionary tuples with events and stations
    :type event_stations: List[Tuple[dict, dict]]
    """
    stations = chain.from_iterable(
        (_i[1].values() for _i in event_stations if _i[1])
    )
    # Remove duplicates
    stations = [(_i["latitude"], _i["longitude"]) for _i in stations]
    stations = set(stations)
    x, y = [_i[1] for _i in stations], [_i[0] for _i in stations]

    map_object.scatter(
        x,
        y,
        s=10,
        color="#333333",
        edgecolor="#111111",
        alpha=0.6,
        zorder=12,
        marker="v",
        transform=cp.crs.PlateCarree(),
    )
    plt.tight_layout()


def plot_tf(
    data: np.ndarray,
    delta: float,
    freqmin: float = None,
    freqmax: float = None,
):
    """
    Plots a time frequency representation of any time series. Right now it is
    basically limited to plotting source time functions.

    :param data: Signal represented in a numpy array
    :type data: np.ndarray
    :param delta: Time discretization in seconds
    :type delta: float
    :param freqmin: minimum frequency of signal, defaults to None
    :type freqmin: float, optional
    :param freqmax: maximum frequency of signal, defaults to None
    :type freqmax: float, optional
    """
    npts = len(data)

    fig = plot_tfr(
        data,
        dt=delta,
        fmin=1.0 / (npts * delta),
        fmax=1.0 / (2.0 * delta),
        show=False,
    )

    # Get the different axes...use some kind of logic to determine which is
    # which. This is super flaky as dependent on the ObsPy version and what
    # not.
    axes = {}
    for ax in fig.axes:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        # Colorbar.
        if xlim == ylim:
            continue

        # Spectral axis.
        elif xlim[0] > xlim[1]:
            axes["spec"] = ax

        elif ylim[0] < 0:
            axes["time"] = ax

        else:
            axes["tf"] = ax

    fig.suptitle("Source Time Function")

    if len(axes) != 3:
        msg = "Could not plot frequency limits!"
        print(msg)
        plt.gcf().patch.set_alpha(0.0)
        plt.show()
        return

    axes["spec"].grid()
    axes["time"].grid()
    axes["tf"].grid()

    axes["spec"].xaxis.tick_top()
    axes["spec"].set_ylabel("Frequency [Hz]")

    axes["time"].set_xlabel("Time [s]")
    axes["time"].set_ylabel("Velocity [m/s]")

    if freqmin is not None and freqmax is not None:
        xmin, xmax = axes["tf"].get_xlim()
        axes["tf"].hlines(freqmin, xmin, xmax, color="green", lw=2)
        axes["tf"].hlines(freqmax, xmin, xmax, color="red", lw=2)
        axes["tf"].text(
            xmax - (0.02 * (xmax - xmin)),
            freqmin,
            "%.1f s" % (1.0 / freqmin),
            color="green",
            horizontalalignment="right",
            verticalalignment="top",
        )
        axes["tf"].text(
            xmax - (0.02 * (xmax - xmin)),
            freqmax,
            "%.1f s" % (1.0 / freqmax),
            color="red",
            horizontalalignment="right",
            verticalalignment="bottom",
        )

        xmin, xmax = axes["spec"].get_xlim()
        axes["spec"].hlines(freqmin, xmin, xmax, color="green", lw=2)
        axes["spec"].hlines(freqmax, xmin, xmax, color="red", lw=2)

    plt.gcf().patch.set_alpha(0.0)
    plt.show()


def plot_heaviside(data: np.ndarray, delta: float):
    """
    Make a simple plot to show how the source time function looks when it
    is unfiltered.

    :param data: Signal represented in a numpy array
    :type data: np.ndarray
    :param delta: Time discretization in seconds
    :type delta: float
    """
    # For visualization we append a few zeros at the beginning of the stf.
    data = np.insert(data, 0, np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))

    npts = len(data)
    time = np.arange(-(delta * 7), delta * (npts - 7), delta)

    plt.plot(time, data, color="black")
    plt.title("Source Time Function (6 extra zero samples before)")
    plt.xlabel("Time (Seconds)")
    plt.ylabel("Injected Force (Normalized)")
    plt.show()


def plot_event_histogram(
    events: List[Dict[str, Union[str, float]]], plot_type: str
):
    """
    Plot a histogram of the distribution of events in either time or depth

    :param events: A list of event informations
    :type events: List[Dict[str, Union[str, float]]]
    :param plot_type: time or depth
    :type plot_type: str
    """
    from matplotlib.dates import date2num, num2date
    from matplotlib import ticker

    plt.figure(figsize=(12, 4))

    values = []
    for event in events:
        if plot_type == "depth":
            values.append(event["depth_in_km"])
        elif plot_type == "time":
            values.append(date2num(event["origin_time"].datetime))

    plt.hist(values, bins=250)

    if plot_type == "time":
        plt.gca().xaxis.set_major_formatter(
            ticker.FuncFormatter(
                lambda numdate, _: num2date(numdate).strftime("%Y-%d-%m")
            )
        )
        plt.gcf().autofmt_xdate()
        plt.xlabel("Origin time (UTC)")
        plt.title("Origin time distribution (%i events)" % len(events))
    elif plot_type == "depth":
        plt.xlabel("Event depth in km")
        plt.title("Hypocenter depth distribution (%i events)" % len(events))

    plt.tight_layout()
