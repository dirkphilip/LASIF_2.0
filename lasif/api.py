#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
An api to communicate directly with lasif functions without using the
command line interface.

:copyright:
    Solvi Thrastarson (soelvi.thrastarson@erdw.ethz.ch), 2018
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)

"""
import os
import pathlib
from typing import Union
import colorama
import toml
import numpy as np
from typing import Union, List

from lasif.components.communicator import Communicator
from lasif.components.project import Project

from lasif.exceptions import (
    LASIFError,
    LASIFNotFoundError,
    LASIFCommandLineException,
)


def find_project_comm(folder):
    """
    Will search upwards from the given folder until a folder containing a
    LASIF root structure is found. The absolute path to the root is returned.

    :param folder: Path to folder where you want to search from
    :type folder: Union[str, pathlib.Path, object]
    """
    if isinstance(folder, Communicator):
        return folder

    folder = pathlib.Path(folder).absolute()
    max_folder_depth = 10
    folder = folder
    for _ in range(max_folder_depth):
        if (folder / "lasif_config.toml").exists():
            return Project(folder).get_communicator()
        folder = folder.parent
    msg = "Not inside a LASIF project."
    raise LASIFCommandLineException(msg)


def plot_domain(lasif_root, save=False, inner_boundary=False):
    """
    Plot the studied domain specified in config file

    :param lasif_root: path to lasif root directory.
    :type lasif_root: Union[str, pathlib.Path, object]
    :param save: save file, defaults to False
    :type save: bool, optional
    :param inner_boundary: binary whether the inner boundary should be drawn
        Only works well for convex domains, defaults to False
    :type inner_boundary: bool, optional
    """
    import matplotlib.pyplot as plt

    comm = find_project_comm(lasif_root)

    comm.visualizations.plot_domain(inner_boundary)

    if save:
        outfile = os.path.join(
            comm.project.get_output_folder(
                type="domain_plot", tag="domain", timestamp=True
            ),
            "domain.png",
        )
        plt.savefig(outfile, dpi=200, transparent=True)
        print(f"Saved picture at {outfile}")
    else:
        plt.show()


def plot_event(
    lasif_root,
    event_name: str,
    weight_set_name: str = None,
    save: bool = False,
    intersection_override: bool = None,
    inner_boundary: bool = False,
):
    """
    Plot a single event including stations on a map. Events can be
    color coded based on their weight

    :param lasif_root: path to lasif root directory
    :type lasif_root: Union[str, pathlib.Path, object]
    :param event_name: name of event to plot
    :type event_name: str
    :param weight_set_name: name of station weight set, defaults to None
    :type weight_set_name: str, optional
    :param save: if figure should be saved, defaults to False
    :type save: bool, optional
    :param intersection_override: boolean to require to have the same
        stations recording all events, i.e. the intersection of receiver
        sets. The intersection will consider two stations equal i.f.f. the
        station codes AND coordinates (LAT, LON, Z) are equal. If None is
        passed, the value use_only_intersection from the projects'
        configuration file is used, defaults to None
    :type intersection_override: bool, optional
    :param inner_boundary: binary whether the inner boundary should be drawn
        Only works well for convex domains, defaults to False
    :type inner_boundary: bool, optional
    """
    import matplotlib.pyplot as plt

    comm = find_project_comm(lasif_root)

    if save:
        plt.switch_backend("agg")

    comm.visualizations.plot_event(
        event_name,
        weight_set_name,
        intersection_override=intersection_override,
        inner_boundary=inner_boundary,
    )

    if save:
        outfile = os.path.join(
            comm.project.get_output_folder(
                type="event_plots", tag="event", timestamp=False
            ),
            f"{event_name}.png",
        )
        plt.savefig(outfile, dpi=200, transparent=True)
        print("Saved picture at %s" % outfile)
    else:
        plt.show()


def plot_events(
    lasif_root,
    type_of_plot: str = "map",
    iteration: str = None,
    save: bool = False,
    inner_boundary: bool = False,
):
    """
    Plot a all events on the domain

    :param lasif_root: path to lasif root directory
    :type lasif_root: Union[str, pathlib.Path, object]
    :param type_of_plot: type of plot, options are 'map', 'depth' and 'time', 
        defaults to 'map'
    :type type_of_plot: str, optional
    :param iteration: plot all events of an iteration, defaults to None
    :type iteration: str, optional
    :param save: if figure should be saved, defaults to False
    :type save: bool, optional
    :param inner_boundary: binary weather the inner boundary should be drawn
        Only works well for convex domains, defaults to False
    :type inner_boundary: bool, optional
    """
    import matplotlib.pyplot as plt

    comm = find_project_comm(lasif_root)

    comm.visualizations.plot_events(
        type_of_plot, iteration=iteration, inner_boundary=inner_boundary
    )

    if save:
        if iteration:
            file = f"events_{iteration}.png"
            timestamp = False
        else:
            file = "events.png"
            timestamp = True
        outfile = os.path.join(
            comm.project.get_output_folder(
                type="event_plots", tag="events", timestamp=timestamp
            ),
            file,
        )
        plt.savefig(outfile, dpi=200, transparent=True)
        print("Saved picture at %s" % outfile)
    else:
        plt.show()


def plot_station_misfits(lasif_root, event: str, iteration: str, save=False):
    """
    Plot a map of a specific event where all the stations are colour coded by
    their respective misfit value. You need to compute adjoint sources for
    the respective iteration prior to making this plot.
    Keep in mind that station with no windows will not get plotted, these
    stations might be the ones with the largest misfits in reality

    :param lasif_root: Lasif root directory or communicator object
    :type lasif_root: Union[str, pathlib.Path, object]
    :param event: Name of event
    :type event: str
    :param iteration: Name of iteration
    :type iteration: str
    :param save: You want to save the plot?, defaults to False
    :type save: bool, optional
    """
    import matplotlib.pyplot as plt

    comm = find_project_comm(lasif_root)

    comm.visualizations.plot_station_misfits(
        event_name=event, iteration=iteration
    )

    if save:
        file = f"misfit_{event}_{iteration}.png"
        timestamp = False

        outfile = os.path.join(
            comm.project.get_output_folder(
                type="event_plots", tag="events", timestamp=timestamp
            ),
            file,
        )
        plt.savefig(outfile, dpi=200, transparent=True)
        print("Saved picture at %s" % outfile)
    else:
        plt.show()


def plot_raydensity(
    lasif_root,
    plot_stations: bool,
    iteration: str = None,
    save: bool = True,
    intersection_override: bool = None,
):
    """
    Plot a distribution of earthquakes and stations with great circle rays
    plotted underneath.

    :param lasif_root: Lasif root directory
    :type lasif_root: Union[str, pathlib.Path, object]
    :param plot_stations: boolean argument whether stations should be plotted
    :type plot_stations: bool
    :param iteration: If you only want events from a certain iteration, if
        None is passed every event will be used, defaults to None
    :type iteration: str, optional
    :param save: Whether you want to save the figure, if False, it gets
        plotted and not saved, defaults to True
    :type save: bool, optional
    :param intersection_override: boolean to require to have the same
        stations recording all events, i.e. the intersection of receiver
        sets. The intersection will consider two stations equal i.f.f. the
        station codes AND coordinates (LAT, LON, Z) are equal. If None is
        passed, the value use_only_intersection from the projects'
        configuration file is used, defaults to None
    :type intersection_override: bool, optional
    """

    comm = find_project_comm(lasif_root)

    comm.visualizations.plot_raydensity(
        plot_stations=plot_stations,
        iteration=iteration,
        save_plot=save,
        intersection_override=intersection_override,
    )


def plot_all_rays(
    lasif_root,
    plot_stations: bool,
    iteration: str = None,
    save: bool = True,
    intersection_override: bool = None,
):
    """
    Plot all the rays that are in the project or in a specific iteration.
    This is typically slower than the plot_raydensity function

    :param lasif_root: Lasif root directory
    :type lasif_root: Union[pathlib.Path, str, object]
    :param plot_stations: True/False whether stations should be plotted
    :type plot_stations: bool
    :param iteration: If you only want events from a certain iteration, if
        None is passed every event will be used, defaults to None
    :type iteration: str, optional
    :param save: Whether you want to save the figure, if False, it gets
        plotted and not saved, defaults to True
    :type save: bool, optional
    :param intersection_override: boolean to require to have the same
        stations recording all events, i.e. the intersection of receiver
        sets. The intersection will consider two stations equal i.f.f. the
        station codes AND coordinates (LAT, LON, Z) are equal. If None is
        passed, the value use_only_intersection from the projects'
        configuration file is used, defaults to None
    :type intersection_override: bool, optional
    """

    comm = find_project_comm(lasif_root)

    comm.visualizations.plot_all_rays(
        plot_stations=plot_stations,
        iteration=iteration,
        save_plot=save,
        intersection_override=None,
    )


def add_gcmt_events(
    lasif_root,
    count: int,
    min_mag: float,
    max_mag: float,
    min_dist: float,
    min_year: int = None,
    max_year: int = None,
):
    """
    Add events to the project.

    :param lasif_root: path to lasif root directory
    :type lasif_root: Union[str, pathlib.Path, object]
    :param count: Nunber of events
    :type count: int
    :param min_mag: Minimum magnitude
    :type min_mag: float
    :param max_mag: Maximum magnitude
    :type max_mag: float
    :param min_dist: Minimum distance between events in km
    :type min_dist: float
    :param min_year: Year to start event search, defaults to None
    :type min_year: float, optional
    :param max_year: Year to end event search, defaults to None
    :type max_year: float, optional
    """

    from lasif.tools.query_gcmt_catalog import add_new_events

    comm = find_project_comm(lasif_root)

    add_new_events(
        comm=comm,
        count=count,
        min_magnitude=min_mag,
        max_magnitude=max_mag,
        min_year=min_year,
        max_year=max_year,
        threshold_distance_in_km=min_dist,
    )


def add_spud_event(lasif_root, url: str):
    """
    Adds events from the iris spud service, provided a link to the event

    :param lasif_root: Path to lasif project
    :type lasif_root: Union[str, pathlib.Path, object]
    :param url: URL to the spud event
    :type url: str
    """
    from lasif.scripts.iris2quakeml import iris2quakeml

    comm = find_project_comm(lasif_root)

    iris2quakeml(url, comm.project.paths["eq_data"])


def project_info(lasif_root):
    """
    Print a summary of the project

    :param lasif_root: path to lasif root directory
    :type lasif_root: Union[str, pathlib.Path, object]
    """

    comm = find_project_comm(lasif_root)
    print(comm.project)


def download_data(
    lasif_root,
    event_name: Union[str, List[str]] = None,
    providers: List[str] = None,
):
    """
    Download available data for events

    :param lasif_root: path to lasif root directory
    :type lasif_root: Union[str, pathlib.Path, object]
    :param event_name: Name of event(s), if you want to pass all events,
        pass None, you can pass either an event or a list of events,
        defaults to None.
    :type event_name: Union[str, List[str]], optional
    :param providers: A list of providers to download from, if nothing passed
        it will query all known providers that FDSN knows, defaults to None
    :type providers: List[str]
    """

    comm = find_project_comm(lasif_root)
    if event_name is None:
        event_name = comm.events.list()
    if not isinstance(event_name, list):
        event_name = [event_name]
    for event in event_name:
        comm.downloads.download_data(event, providers=providers)


def list_events(
    lasif_root,
    just_list: bool = True,
    iteration: str = None,
    output: bool = False,
):
    """
    Print a list of events in project

    :param lasif_root: path to lasif root directory
    :type lasif_root: Union[str, pathlib.Path, object]
    :param just_list: Show only a plain list of events, if False it gives
        more information on the events, defaults to True
    :type just_list: bool, optional
    :param iteration: Show only events for specific iteration, defaults to None
    :type iteration: str, optional
    :param output: Do you want to output the list into a variable, defaults
        to False
    :type output: bool, optional
    """

    comm = find_project_comm(lasif_root)
    if just_list:
        if output:
            return comm.events.list(iteration=iteration)
        else:
            for event in sorted(comm.events.list(iteration=iteration)):
                print(event)

    else:
        if output:
            raise LASIFError("You can only output a basic list")
        print(
            f"%i event%s in %s:"
            % (
                comm.events.count(),
                "s" if comm.events.count() != 1 else "",
                "iteration" if iteration else "project",
            )
        )

        from lasif.tools.prettytable import PrettyTable

        tab = PrettyTable(["Event Name", "Lat", "Lng", "Depth (km)", "Mag"])

        for event in comm.events.list(iteration=iteration):
            ev = comm.events.get(event)
            tab.add_row(
                [
                    event,
                    "%6.1f" % ev["latitude"],
                    "%6.1f" % ev["longitude"],
                    "%3i" % int(ev["depth_in_km"]),
                    "%3.1f" % ev["magnitude"],
                ]
            )
        tab.align = "r"
        tab.align["Event Name"] = "l"
        print(tab)


def event_info(lasif_root, event_name: str, verbose: bool = False):
    """
    Print information about a single event

    :param lasif_root: path to lasif root directory
    :type lasif_root: Union[str, pathlib.Path, object]
    :param event_name: Name of event
    :type event_name: str
    :param verbose: Print station information as well, defaults to False
    :type verbose: bool, optional
    """

    comm = find_project_comm(lasif_root)
    if not comm.events.has_event(event_name):
        msg = "Event '%s' not found in project." % event_name
        raise LASIFError(msg)

    event_dict = comm.events.get(event_name)

    print(
        "Earthquake with %.1f %s at %s"
        % (
            event_dict["magnitude"],
            event_dict["magnitude_type"],
            event_dict["region"],
        )
    )
    print(
        "\tLatitude: %.3f, Longitude: %.3f, Depth: %.1f km"
        % (
            event_dict["latitude"],
            event_dict["longitude"],
            event_dict["depth_in_km"],
        )
    )
    print("\t%s UTC" % str(event_dict["origin_time"]))

    try:
        stations = comm.query.get_all_stations_for_event(event_name)
    except LASIFError:
        stations = {}

    if verbose:
        from lasif.utils import table_printer

        print(
            "\nStation and waveform information available at %i "
            "stations:\n" % len(stations)
        )
        header = ["ID", "Latitude", "Longitude", "Elevation_in_m"]
        keys = sorted(stations.keys())
        data = [
            [
                key,
                stations[key]["latitude"],
                stations[key]["longitude"],
                stations[key]["elevation_in_m"],
            ]
            for key in keys
        ]
        table_printer(header, data)
    else:
        print(
            "\nStation and waveform information available at %i stations. "
            "Use '-v' to print them." % len(stations)
        )


def plot_stf(lasif_root):
    """
    Plot the source time function

    :param lasif_root: path to lasif root directory
    :type lasif_root: Union[str, pathlib.Path, object]
    """
    import lasif.visualization

    comm = find_project_comm(lasif_root)

    freqmax = 1.0 / comm.project.simulation_settings["minimum_period_in_s"]
    freqmin = 1.0 / comm.project.simulation_settings["maximum_period_in_s"]

    stf_fct = comm.project.get_project_function("source_time_function")

    delta = comm.project.simulation_settings["time_step_in_s"]
    npts = comm.project.simulation_settings["number_of_time_steps"]
    stf_type = comm.project.simulation_settings["source_time_function"]

    stf = {"delta": delta}
    if stf_type == "heaviside":
        stf["data"] = stf_fct(npts=npts, delta=delta)
        lasif.visualization.plot_heaviside(stf["data"], stf["delta"])
    elif stf_type == "bandpass_filtered_heaviside":
        stf["data"] = stf_fct(
            npts=npts, delta=delta, freqmin=freqmin, freqmax=freqmax
        )
        lasif.visualization.plot_tf(
            stf["data"], stf["delta"], freqmin=freqmin, freqmax=freqmax
        )
    else:
        raise LASIFError(
            f"{stf_type} is not supported by lasif. Check your"
            f"config file and make sure the source time "
            f"function is either 'heaviside' or "
            f"'bandpass_filtered_heaviside'."
        )


def init_project(project_path: Union[str, pathlib.Path]):
    """
    Create a new project

    :param project_path: Path to project root directory. Can use absolute
        paths or relative paths from current working directory.
    :type project_path: Union[str, pathlib.Path]
    """

    project_path = pathlib.Path(project_path).absolute()

    if project_path.exists():
        msg = "The given PROJECT_PATH already exists. It must not exist yet."
        raise LASIFError(msg)
    try:
        os.makedirs(project_path)
    except Exception:
        msg = f"Failed creating directory {project_path}. Permissions?"
        raise LASIFError(msg)

    Project(project_root_path=project_path, init_project=project_path.name)
    print(f"Initialized project in {project_path.name}")


def calculate_adjoint_sources(
    lasif_root,
    iteration: str,
    window_set: str,
    weight_set: str = None,
    events: Union[str, List[str]] = None,
):
    """
    Calculate adjoint sources for a given iteration

    :param lasif_root: path to lasif root directory
    :type lasif_root: Union[str, pathlib.Path, object]
    :param iteration: name of iteration
    :type iteration: str
    :param window_set: name of window set
    :type window_set: str
    :param weight_set: name of station weight set, defaults to None
    :type weight_set: str, optional
    :param events: Name of event or list of events. To get all events for
        the iteration, pass None, defaults to None
    :type events: Union[str, List[str]]
    """
    from mpi4py import MPI

    comm = find_project_comm(lasif_root)

    # some basic checks
    if not comm.windows.has_window_set(window_set):
        if MPI.COMM_WORLD.rank == 0:
            raise LASIFNotFoundError(
                "Window set {} not known to LASIF".format(window_set)
            )
        return

    if not comm.iterations.has_iteration(iteration):
        if MPI.COMM_WORLD.rank == 0:
            raise LASIFNotFoundError(
                "Iteration {} not known to LASIF".format(iteration)
            )
        return

    if events is None:
        events = comm.events.list(iteration=iteration)
    if isinstance(events, str):
        events = [events]

    for _i, event in enumerate(events):
        if not comm.events.has_event(event):
            if MPI.COMM_WORLD.rank == 0:
                print(
                    "Event '%s' not known to LASIF. No adjoint sources for "
                    "this event will be calculated. " % event
                )
            continue

        if MPI.COMM_WORLD.rank == 0:
            print(
                "\n{green}"
                "==========================================================="
                "{reset}".format(
                    green=colorama.Fore.GREEN, reset=colorama.Style.RESET_ALL
                )
            )
            print(
                "Starting adjoint source calculation for event %i of "
                "%i..." % (_i + 1, len(events))
            )
            print(
                "{green}"
                "==========================================================="
                "{reset}\n".format(
                    green=colorama.Fore.GREEN, reset=colorama.Style.RESET_ALL
                )
            )

        # Get adjoint sources_filename
        # filename = comm.adj_sources.get_filename(
        #     event=event, iteration=iteration
        # )

        # remove adjoint sources if they already exist
        if MPI.COMM_WORLD.rank == 0:
            filename = comm.adj_sources.get_filename(
                event=event, iteration=iteration
            )
            if os.path.exists(filename):
                os.remove(filename)

        MPI.COMM_WORLD.barrier()
        comm.adj_sources.calculate_adjoint_sources(
            event, iteration, window_set
        )
        MPI.COMM_WORLD.barrier()
        if MPI.COMM_WORLD.rank == 0:
            comm.adj_sources.finalize_adjoint_sources(
                iteration, event, weight_set
            )


def select_windows(
    lasif_root,
    iteration: str,
    window_set: str,
    events: Union[str, List[str]] = None,
):
    """
    Autoselect windows for a given iteration and event combination

    :param lasif_root: path to lasif root directory
    :type lasif_root: Union[str, pathlib.Path, object]
    :param iteration: name of iteration
    :type iteration: str
    :param window_set: name of window set
    :type window_set: str
    :param events: An event or a list of events. To get all of them pass 
        None, defaults to None
    :type events: Union[str, List[str]], optional
    """

    comm = find_project_comm(lasif_root)  # Might have to do this mpi

    if events is None:
        events = comm.events.list(iteration=iteration)
    if isinstance(events, str):
        events = [events]

    for event in events:
        print(f"Selecting windows for event: {event}")
        comm.windows.select_windows(event, iteration, window_set)


def open_gui(lasif_root):
    """
    Open up the misfit gui

    :param lasif_root: path to lasif root directory
    :type lasif_root: Union[str, pathlib.Path, object]
    """

    comm = find_project_comm(lasif_root)

    from lasif.misfit_gui.misfit_gui import launch

    launch(comm)


def create_weight_set(lasif_root, weight_set: str):
    """
    Create a new set of event and station weights

    :param lasif_root: path to lasif root directory
    :type lasif_root: Union[str, pathlib.Path, object]
    :param weight_set: name of weight set
    :type weight_set: str
    """

    comm = find_project_comm(lasif_root)

    comm.weights.create_new_weight_set(
        weight_set_name=weight_set,
        events_dict=comm.query.get_all_stations_for_events(),
    )


def compute_station_weights(
    lasif_root,
    weight_set: str,
    events: Union[str, List[str]] = None,
    iteration: str = None,
):
    """
    Compute weights for stations based on amount of neighbouring stations

    :param lasif_root: path to lasif root directory
    :type lasif_root: Union[str, pathlib.Path, object]
    :param weight_set: name of weight set to compute into
    :type weight_set: str
    :param events: An event or a list of events. To get all of them pass 
        None, defaults to None
    :type events: Union[str, List[str]], optional
    :param iteration: name of iteration to compute weights for, controls
        the events which are picked for computing weights for.
    :type iteration: str, optional
    """

    comm = find_project_comm(lasif_root)

    if events is None:
        events = comm.events.list(iteration=iteration)
    if isinstance(events, str):
        events = [events]
    events_dict = {}
    if not comm.weights.has_weight_set(weight_set):
        for event in events:
            events_dict[event] = comm.query.get_all_stations_for_event(event)
        comm.weights.create_new_weight_set(
            weight_set_name=weight_set, events_dict=events_dict,
        )

    w_set = comm.weights.get(weight_set)
    from tqdm import tqdm

    for event in events:
        print(f"Calculating station weights for event: {event}")
        if not comm.events.has_event(event):
            raise LASIFNotFoundError(f"Event: {event} is not known to LASIF")
        stations = comm.query.get_all_stations_for_event(event)
        events_dict[event] = list(stations.keys())
        locations = np.zeros((2, len(stations.keys())), dtype=np.float64)
        for _i, station in enumerate(stations):
            locations[0, _i] = stations[station]["latitude"]
            locations[1, _i] = stations[station]["longitude"]

        sum_value = 0.0

        for station in tqdm(stations):
            weight = comm.weights.calculate_station_weight(
                lat_1=stations[station]["latitude"],
                lon_1=stations[station]["longitude"],
                locations=locations,
            )
            sum_value += weight
            w_set.events[event]["stations"][station]["station_weight"] = weight
        for station in stations:
            w_set.events[event]["stations"][station]["station_weight"] *= (
                len(stations) / sum_value
            )
        if len(stations.keys()) == 1:
            w_set.events[event]["stations"][stations[station]][
                "station_weight"
            ] = 1.0

    comm.weights.change_weight_set(
        weight_set_name=weight_set, weight_set=w_set, events_dict=events_dict,
    )


def set_up_iteration(
    lasif_root,
    iteration: str,
    events: Union[str, List[str]] = None,
    event_specific: bool = False,
    remove_dirs: bool = False,
):
    """
    Creates or removes directory structure for an iteration

    :param lasif_root: path to lasif root directory
    :type lasif_root: Union[str, pathlib.Path, object]
    :param iteration: name of iteration
    :type iteration: str
    :param events: An event or a list of events. To get all of them pass 
        None, defaults to None
    :type events: Union[str, List[str]], optional
    :param event_specific: If the inversion needs a specific model for each
        event, defaults to False
    :type event_specific: bool, optional
    :param remove_dirs: boolean value to remove dirs, defaults to False
    :type remove_dirs: bool, optional
    """

    comm = find_project_comm(lasif_root)

    if events is None:
        events = comm.events.list()
    if isinstance(events, str):
        events = [events]

    iterations = list_iterations(comm, output=True)
    if isinstance(iterations, list):
        if iteration in iterations:
            if not remove_dirs:
                print(f"{iteration} already exists")
                return
    comm.iterations.setup_directories_for_iteration(
        iteration_name=iteration,
        remove_dirs=remove_dirs,
        events=events,
        event_specific=event_specific,
    )
    iteration = comm.iterations.get_long_iteration_name(iteration)

    if not remove_dirs:
        comm.iterations.setup_iteration_toml(iteration_name=iteration)
        comm.iterations.setup_events_toml(
            iteration_name=iteration, events=events
        )


def write_misfit(
    lasif_root,
    iteration: str,
    weight_set: str = None,
    window_set: str = None,
    events: Union[str, List[str]] = None,
):
    """
    Write misfit for iteration

    :param lasif_root: path to lasif root directory
    :type lasif_root: Union[str, pathlib.Path, object]
    :param iteration: name of iteration
    :param weight_set: name of weight set [optional]
    :param window_set: name of window set [optional]
    :param events: An event or a list of events. To get all of them pass 
        None, defaults to None
    :type events: Union[str, List[str]], optional
    """

    comm = find_project_comm(lasif_root)

    if weight_set:
        if not comm.weights.has_weight_set(weight_set):
            raise LASIFNotFoundError(
                f"Weights {weight_set} not known" f"to LASIF"
            )
    # Check if iterations exists
    if not comm.iterations.has_iteration(iteration):
        raise LASIFNotFoundError(
            f"Iteration {iteration} " f"not known to LASIF"
        )

    long_iter_name = comm.iterations.get_long_iteration_name(iteration)

    path = comm.project.paths["iterations"]
    toml_filename = os.path.join(path, long_iter_name, "misfits.toml")
    total_misfit = 0.0

    if not events:
        events = comm.events.list(iteration=iteration)
        iteration_dict = {"event_misfits": {}}
    if isinstance(events, str):
        events = [events]
    else:
        # Check to see whether iteration_toml previously existed
        if os.path.isfile(toml_filename):
            iteration_dict = toml.load(toml_filename)
            other_events = iteration_dict["event_misfits"].keys() - events
            for event in other_events:
                total_misfit += iteration_dict["event_misfits"][event]
        else:
            iteration_dict = {"event_misfits": {}}

    for event in events:
        event_misfit = comm.adj_sources.get_misfit_for_event(
            event, iteration, weight_set
        )
        iteration_dict["event_misfits"][event] = float(event_misfit)
        total_misfit += event_misfit

    iteration_dict["total_misfit"] = float(total_misfit)
    iteration_dict["weight_set_name"] = weight_set
    iteration_dict["window_set_name"] = window_set

    with open(toml_filename, "w") as fh:
        toml.dump(iteration_dict, fh)


def list_iterations(lasif_root, output: bool = False):
    """
    List iterations in project

    :param lasif_root: path to lasif root directory
    :type lasif_root: Union[str, pathlib.Path, object]
    :param output: If the function should return the list, defaults to False
    :type output: bool, optional
    
    """

    comm = find_project_comm(lasif_root)

    iterations = comm.iterations.list()
    if len(iterations) == 0:
        print("There are no iterations in this project")
    else:
        if output:
            return iterations
        if len(iterations) == 1:
            print(f"There is {len(iterations)} iteration in this project")
            print("Iteration known to LASIF: \n")
        else:
            print(f"There are {len(iterations)} iterations in this project")
            print("Iterations known to LASIF: \n")
    for iteration in iterations:
        print(comm.iterations.get_long_iteration_name(iteration), "\n")


def compare_misfits(
    lasif_root,
    from_it: str,
    to_it: str,
    events: Union[str, List[str]] = None,
    weight_set: str = None,
    print_events: bool = False,
):
    """
    Compares the total misfit between two iterations.

    Total misfit is used regardless of the similarity of the picked windows
    from each iteration. This might skew the results but should
    give a good idea unless the windows change excessively between
    iterations.

    If windows are weighted in the calculation of the adjoint
    sources. That should translate into the calculated misfit
    value.

    :param lasif_root: path to lasif root directory
    :type lasif_root: Union[str, pathlib.Path, object]
    :param from_it: evaluate misfit from this iteration
    :type from_it: str
    :param to_it: to this iteration
    :type to_it: str
    :param events: An event or a list of events. To get all of them pass 
        None, defaults to None
    :type events: Union[str, List[str]], optional
    :param weight_set: Set of station and event weights, defaults to None
    :type weight_set: str, optional
    :param print_events: compare misfits for each event, defaults to False
    :type print_events: bool, optional
    """
    comm = find_project_comm(lasif_root)

    if events is None:
        events = comm.events.list()
    if isinstance(events, str):
        events = [events]

    if weight_set:
        if not comm.weights.has_weight_set(weight_set):
            raise LASIFNotFoundError(
                f"Weights {weight_set} not known" f"to LASIF"
            )
    # Check if iterations exist
    if not comm.iterations.has_iteration(from_it):
        raise LASIFNotFoundError(f"Iteration {from_it} not known to LASIF")
    if not comm.iterations.has_iteration(to_it):
        raise LASIFNotFoundError(f"Iteration {to_it} not known to LASIF")

    from_it_misfit = 0.0
    to_it_misfit = 0.0
    for event in events:
        from_it_misfit += comm.adj_sources.get_misfit_for_event(
            event, from_it, weight_set
        )
        to_it_misfit += comm.adj_sources.get_misfit_for_event(
            event, to_it, weight_set
        )
        if print_events:
            # Print information about every event.
            from_it_misfit_event = comm.adj_sources.get_misfit_for_event(
                event, from_it, weight_set
            )
            to_it_misfit_event = comm.adj_sources.get_misfit_for_event(
                event, to_it, weight_set
            )
            print(
                f"{event}: \n"
                f"\t iteration {from_it} has misfit: "
                f"{from_it_misfit_event} \n"
                f"\t iteration {to_it} has misfit: {to_it_misfit_event}."
            )
            rel_change = (
                (to_it_misfit_event - from_it_misfit_event)
                / from_it_misfit_event
                * 100.0
            )
            print(f"Relative change: {rel_change:.2f}%")

    print(f"Total misfit for iteration {from_it}: {from_it_misfit}")
    print(f"Total misfit for iteration {to_it}: {to_it_misfit}")
    rel_change = (to_it_misfit - from_it_misfit) / from_it_misfit * 100.0
    print(
        f"Relative change in total misfit from iteration {from_it} to "
        f"{to_it} is: {rel_change:.2f}"
    )
    n_events = len(comm.events.list())
    print(
        f"Misfit per event for iteration {from_it}: "
        f"{from_it_misfit/n_events}"
    )
    print(
        f"Misfit per event for iteration {to_it}: " f"{to_it_misfit/n_events}"
    )


def list_weight_sets(lasif_root):
    """
    Print a list of all weight sets in the project.

    :param lasif_root: path to lasif root directory
    :type lasif_root: Union[str, pathlib.Path, object]
    """

    comm = find_project_comm(lasif_root)

    it_len = comm.weights.count()

    print(
        "%i weight set(s)%s in project:" % (it_len, "s" if it_len != 1 else "")
    )
    for weights in comm.weights.list():
        print("\t%s" % weights)


def process_data(
    lasif_root, events: Union[str, List[str]] = None, iteration: str = None
):
    """
    Process recorded data

    :param lasif_root: path to lasif root directory
    :type lasif_root: Union[str, pathlib.Path, object]
    :param events: An event or a list of events. To get all of them pass 
        None, defaults to None
    :type events: Union[str, List[str]], optional
    :param iteration: Process data from events used in an iteration,
        defaults to None
    :type iteration: str, optional
    """
    comm = find_project_comm(lasif_root)

    if events is None:
        events = comm.events.list(iteration=iteration)
    if isinstance(events, str):
        events = [events]

    exceptions = []
    # if MPI.COMM_WORLD.rank == 0:
    # Check if the event ids are valid.
    if not exceptions and events:
        for event_name in events:
            if not comm.events.has_event(event_name):
                msg = "Event '%s' not found." % event_name
                exceptions.append(msg)
                break

    # exceptions = MPI.COMM_WORLD.bcast(exceptions, root=0)
    if exceptions:
        raise Exception(exceptions[0])

    # Make sure all the ranks enter the processing at the same time.
    # MPI.COMM_WORLD.barrier()
    comm.waveforms.process_data(events)


def plot_window_statistics(
    lasif_root,
    window_set: str,
    save: bool = False,
    events: Union[str, List[str]] = None,
    iteration: str = None,
):
    """
    Plot some statistics related to windows in a specific set.

    :param lasif_root: path to lasif root directory
    :type lasif_root: Union[str, pathlib.Path, object]
    :param window_set: name of window set
    :type window_set: str
    :param save: Saves the plot in a file, defaults to False,
    :type save: bool, optional
    :param events: An event or a list of events. To get all of them pass 
        None, defaults to None
    :type events: Union[str, List[str]], optional
    :param iteration: Plot statistics related to events in a specific iteration
        , defaults to None
    :type iteration: str, optional
    """
    import matplotlib.pyplot as plt

    comm = find_project_comm(lasif_root)

    if events is None:
        events = comm.events.list(iteration=iteration)
    if isinstance(events, str):
        events = [events]

    if save:
        plt.switch_backend("agg")

    if not comm.windows.has_window_set(window_set):
        raise LASIFNotFoundError("Could not find the specified window set")

    comm.visualizations.plot_window_statistics(
        window_set, events, ax=None, show=False
    )

    if save:
        outfile = os.path.join(
            comm.project.get_output_folder(
                type="window_statistics_plots", tag="windows", timestamp=False
            ),
            f"{window_set}.png",
        )
        plt.savefig(outfile, dpi=200, transparent=True)
        print("Saved picture at %s" % outfile)
    else:
        plt.show()


def plot_windows(
    lasif_root, event_name: str, window_set: str, distance_bins: int = 500
):
    """
    Plot the selected windows for a specific event

    :param lasif_root: path to lasif root directory
    :type lasif_root: Union[str, pathlib.Path, object]
    :param event_name: name of event
    :type event_name: str
    :param window_set: name of window set
    :type window_set: str
    :param distance_bins: number of bins on distance axis for combined plot
    :type distance_bins: int
    """

    comm = find_project_comm(lasif_root)

    comm.visualizations.plot_windows(
        event=event_name,
        window_set_name=window_set,
        ax=None,
        distance_bins=500,
        show=True,
    )


def write_stations_to_file(lasif_root):
    """
    This function writes a csv file with station name and lat,lon locations

    :param lasif_root: path to lasif root directory or communicator
    :type lasif_root: Union[str, pathlib.Path, object]
    """
    import pandas as pd

    comm = find_project_comm(lasif_root)

    events = comm.events.list()

    station_list = {"station": [], "latitude": [], "longitude": []}

    for event in events:
        try:
            data = comm.query.get_all_stations_for_event(
                event, list_only=False
            )
        except LASIFNotFoundError:
            continue
        for key in data:
            if key not in station_list["station"]:
                station_list["station"].append(key)
                station_list["latitude"].append(data[key]["latitude"])
                station_list["longitude"].append(data[key]["latitude"])

    stations = pd.DataFrame(data=station_list)
    output_path = os.path.join(
        comm.project.paths["output"], "station_list.csv"
    )
    stations.to_csv(path_or_buf=output_path, index=False)
    print(f"Wrote a list of stations to file: {output_path}")


def find_event_mesh(lasif_root, event: str):
    """
    See if there is a version of the event mesh which has been
    constructed already but not moved to iteration folder.
    If there is no mesh there it will return False and correct path.
    Otherwise it returns true and the path.

    :param lasif_root: Path to project root
    :type lasif_root: Union[str, pathlib.Path, object]
    :param event: Name of event
    :type event: str
    :return: bool whether mesh exists and path to mesh
    """
    comm = find_project_comm(lasif_root)

    models = comm.project.paths["models"]
    event_mesh = os.path.join(models, "EVENT_MESHES", event, "mesh.h5")
    if os.path.exists(event_mesh):
        return True, event_mesh
    else:
        return False, event_mesh


def get_simulation_mesh(lasif_root, event: str, iteration: str) -> str:
    """
    In the multi mesh approach, each event has a unique mesh.
    This function returns the path of the correct mesh.

    :param lasif_root: Path to project root
    :type lasif_root: Union[str, pathlib.Path, object]
    :param event: Name of event
    :type event: str
    :param iteration: Name of iteration
    :type iteration: str
    :return: path to simulation mesh
    :rtype: str
    """
    import os
    import toml

    comm = find_project_comm(lasif_root)

    models = comm.project.paths["models"]
    it_name = comm.iterations.get_long_iteration_name(iteration)
    iteration_path = os.path.join(comm.project.paths["iterations"], it_name)
    assert iteration in list_iterations(
        comm, output=True
    ), f"Iteration {iteration} not in project"

    events_in_iteration = toml.load(
        os.path.join(iteration_path, "events_used.toml")
    )
    assert (
        event in events_in_iteration["events"]["events_used"]
    ), f"Event {event} not in iteration: {iteration}"

    return os.path.join(models, it_name, event, "mesh.h5")


def get_receivers(lasif_root, event: str):
    """
    Get a list of receiver dictionaries which are compatible with Salvus.
    SalvusFlow can then use these dictionaries to place the receivers.

    :param lasif_root: Path to project root
    :type lasif_root: Union[str, pathlib.Path, object]
    :param event: Name of event which we want to find the receivers for
    :type event: str
    """
    from lasif.utils import place_receivers

    comm = find_project_comm(lasif_root)

    assert comm.events.has_event(event), f"Event: {event} not in project"

    return place_receivers(event, comm)


def get_source(lasif_root, event: str, iteration: str):
    """
    Provide source information to give to Salvus

    :param lasif_root: Path to project root
    :type lasif_root: Union[str, pathlib.Path, object]
    :param event: Name of event which we want to find the source for
    :type event: str
    :param iteration: Name of iteration
    :type iteration: str
    """
    from lasif.utils import prepare_source

    comm = find_project_comm(lasif_root)

    assert comm.events.has_event(event), f"Event: {event} not in project"

    return prepare_source(comm, event, iteration)


def get_subset(
    lasif_root,
    events: List[str],
    count: int,
    existing_events: List[str] = None,
):
    """
    This function gets an optimally distributed subset of events from a larger
    group of events. The distribution is based on Mitchel's best sampling
    algorithm. It's possible to exclude events and have already picked
    events.

    :param lasif_root: Path to project root
    :type lasif_root: Union[str, pathlib.Path, object]
    :param events: List of event names from which to choose from. These
        events must be known to LASIF.
    :type events: List[str]
    :param count: number of events to choose. (Size of subset)
    :type count: int
    :param existing_events: Events already in subset, They will be considered
        in spatial sampling, defaults to None
    :type existing_events: List[str]
    :return: list of selected events, defaults to None
    :rtype: List[str], optional
    """
    from lasif.tools.query_gcmt_catalog import get_subset_of_events

    comm = find_project_comm(lasif_root)

    return get_subset_of_events(comm, count, events, existing_events)


def create_salvus_simulation(
    lasif_root: Union[str, object],
    event: str,
    iteration: str,
    mesh: Union[str, pathlib.Path, object] = None,
    side_set: str = None,
    type_of_simulation: str = "forward",
):
    """
    Create a Salvus simulation object based on simulation and salvus
    specific parameters specified in config file.

    :param lasif_root: path to lasif root folder or the lasif communicator
        object
    :type lasif_root: Union[str, pathlib.Path, object]
    :param event: Name of event
    :type event: str
    :param iteration: Name of iteration
    :type iteration: str
    :param mesh: Path to mesh or Salvus mesh object, if None it will use
        the domain file from config file, defaults to None
    :type mesh: Union[str, pathlib.Path, object], optional
    :param side_set: Name of side set on mesh to place receivers,
        defaults to None.
    :type side_set: str, optional
    :param type_of_simulation: forward or adjoint, defaults to forward
    :type type_of_simulation: str, optional
    :return: Salvus simulation object
    :rtype: object
    """
    if type_of_simulation == "forward":
        from lasif.salvus_utils import create_salvus_forward_simulation as css
    elif type_of_simulation == "adjoint":
        from lasif.salvus_utils import create_salvus_adjoint_simulation as css
    else:
        raise LASIFError("Only types of simulations are forward or adjoint")

    comm = find_project_comm(lasif_root)

    return css(
        comm=comm,
        event=event,
        iteration=iteration,
        mesh=mesh,
        side_set=side_set,
    )


def submit_salvus_simulation(
    lasif_root: Union[str, object], simulations: Union[List[object], object]
) -> object:
    """
    Submit a Salvus simulation to the machine defined in config file
    with details specified in config file

    :param lasif_root: The Lasif communicator object or root file
    :type lasif_root: Union[str, pathlib.Path, object]
    :param simulations: Simulation object
    :type simulations: Union[List[object], object]
    :return: SalvusJob object or array of them
    :rtype: object
    """
    from lasif.salvus_utils import submit_salvus_simulation as sss

    comm = find_project_comm(lasif_root)

    return sss(comm=comm, simulations=simulations)


def validate_data(
    lasif_root,
    data_station_file_availability: bool = False,
    raypaths: bool = False,
    full: bool = False,
):
    """
    Validate the data currently in the project.

    This commands walks through all available data and checks it for validity.
    It furthermore does some sanity checks to detect common problems. These
    should be fixed.

    By default is only checks some things. A full check is recommended but
    potentially takes a very long time.

    Things the command does:

    Event files:
        * Validate against QuakeML 1.2 scheme.
        * Make sure they contain at least one origin, magnitude and focal
          mechanism object.
        * Check for duplicate ids amongst all QuakeML files.
        * Some simply sanity checks so that the event depth is reasonable and
          the moment tensor values as well. This is rather fragile and mainly
          intended to detect values specified in wrong units.

    :param lasif_root: path to lasif root directory
    :type lasif_root: Union[str, pathlib.Path, object]
    :param data_station_file_availability: asserts that all stations have
        corresponding station files and all stations have waveforms, very slow.
    :type data_station_file_availability: bool
    :param raypaths: Assert that all raypaths are within the set boundaries.
        Very slow.
    :type raypaths: bool
    :param full: Run all validations
    :type full: bool
    """

    comm = find_project_comm(lasif_root)

    # If full check, check everything.
    if full:
        data_station_file_availability = True
        raypaths = True

    comm.validator.validate_data(
        data_and_station_file_availability=data_station_file_availability,
        raypaths=raypaths,
    )


def clean_up(lasif_root, clean_up_file: str):
    """
    Clean up the lasif project. The required file can be created with
    the validate_data command.

    :param lasif_root: path to lasif root directory
    :type lasif_root: Union[str, pathlib.Path, object]
    :param clean_up_file: path to clean-up file
    :type clean_up_file: str
    """

    comm = find_project_comm(lasif_root)
    if not os.path.exists(clean_up_file):
        raise LASIFNotFoundError(
            f"Could not find {clean_up_file}\n"
            f"Please check that the specified file path "
            f"is correct."
        )

    comm.validator.clean_up_project(clean_up_file)


def update_catalog():
    """
    Update GCMT catalog
    """
    from lasif.tools.query_gcmt_catalog import update_GCMT_catalog

    update_GCMT_catalog()


def tutorial():
    """
    Open the lasif tutorial in a web-browser.
    """

    import webbrowser

    webbrowser.open("http://dirkphilip.github.io/LASIF_2.0/")
