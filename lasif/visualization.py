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
import numpy as np
from obspy.imaging.beachball import beach
from obspy.signal.tf_misfit import plot_tfr
from lasif.exceptions import LASIFError


def project_points(projection, lon, lat):
    """
    Define the correct projection function depending on name of projection
    
    :param projection: Cartopy projection object
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


def xy_to_lonlat(x: float, y: float, projection):
    """
    Change x and y to latitude and longitude, based on Earth radius
    
    :param x: X coordinate in the correct projection, but in meters
    :type x: float
    :param y: Y coordinate in the correct projection, but in meters
    :type y: float
    :param projection: Cartopy projection object
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
    events: list, map_object, projection, domain, beachball_size=0.02
):
    """
    Plot event stars on a map
    
    :param events: Event information
    :type events: list
    :param map_object: The already made map object from the domain component
    :type map_object: cartopy plot object
    :param projection: Projection object used by cartopy
    :type projection: Cartopy projection object
    :param beachball_size: Size of beachball, defaults to 0.02
    :type beachball_size: float, optional
    """

    for event in events:

        plotted_events = map_object.scatter(
            x=event["longitude"],
            y=event["latitude"],
            zorder=22,
            marker="*",
            c="yellow",
            transform=cp.crs.Geodetic(),
            s=180,
            edgecolors="black",
        )

    return plotted_events


def plot_raydensity(map_object, station_events, domain, projection):
    """
    Create a ray-density plot for all events and all stations.

    This function is potentially expensive and will use all CPUs available.
    Does require geographiclib to be installed.
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


def plot_all_rays(map_object, station_events, domain, projection):
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
                transform=cp.crs.Geodetic(),
                alpha=0.8,
                linewidth=0.4,
                zorder=19,
            )
        i += 1
        print(f"{i} done from {len(station_events)}\n")


def plot_stations_for_event(
    map_object,
    station_dict,
    event_info,
    projection,
    color="green",
    alpha=1.0,
    raypaths=True,
    weight_set=None,
    print_title=True,
):
    """
    Plots all stations for one event.

    :param station_dict: A dictionary whose values at least contain latitude
        and longitude keys.
    """
    import re

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
        cmap = cm.get_cmap("seismic")
        stations = map_object.scatter(
            lngs,
            lats,
            c=weights,
            cmap=cmap,
            s=35,
            marker="v",
            alpha=alpha,
            zorder=5,
            transform=cp.crs.Geodetic(),
        )
        plt.colorbar(stations)

    else:
        stations = map_object.scatter(
            lngs,
            lats,
            color=color,
            s=35,
            marker="v",
            alpha=alpha,
            zorder=5,
            transform=cp.crs.Geodetic(),
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
                transform=cp.crs.Geodetic(),
            )
            # map_object.drawgreatcircle(
            #     event_info["longitude"],
            #     event_info["latitude"],
            #     sta_lng,
            #     sta_lat,
            #     lw=2,
            #     alpha=0.3,
            # )

    title = "Event in %s, at %s, %.1f Mw, with %i stations." % (
        event_info["region"],
        re.sub(r":\d{2}\.\d{6}Z", "", str(event_info["origin_time"])),
        event_info["magnitude"],
        len(station_dict),
    )
    if print_title:
        map_object.set_title(title, size="large")
    return stations


def plot_all_stations(map_object, event_stations: list):
    """
    Add all stations to a map object
    
    :param map_object: A cartopy map object
    :type map_object: object
    :param event_stations: a list of dictionary tuples with events and stations
    :type event_stations: list
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
        transform=cp.crs.Geodetic(),
    )
    plt.tight_layout()


def plot_tf(data, delta, freqmin=None, freqmax=None):
    """
    Plots a time frequency representation of any time series. Right now it is
    basically limited to plotting source time functions.
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


def plot_heaviside(data, delta):
    """
    Make a simple plot to show how the source time function looks when it
    is unfiltered.
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


def plot_event_histogram(events, plot_type):
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
