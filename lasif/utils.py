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
from mpi4py import MPI
import os
import numpy as np

from lasif import LASIFError, LASIFNotFoundError


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
        if chan["end_date"] and \
                (endtime > chan["end_date"]):
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


def greatcircle_points(point_1, point_2, max_extension=None,
                       max_npts=3000):
    """
    Generator yielding a number points along a greatcircle from point_1 to
    point_2. Max extension is the normalization factor. If the distance between
    point_1 and point_2 is exactly max_extension, then 3000 points will be
    returned, otherwise a fraction will be returned.

    If max_extension is not given, the generator will yield exactly max_npts
    points.
    """
    point = geodesic.Geodesic.WGS84.Inverse(
        lat1=point_1.lat, lon1=point_1.lng, lat2=point_2.lat,
        lon2=point_2.lng)
    line = geodesic.Geodesic.WGS84.Line(
        point_1.lat, point_1.lng, point["azi1"])

    if max_extension:
        npts = int((point["a12"] / float(max_extension)) * max_npts)
    else:
        npts = max_npts - 1
    if npts == 0:
        npts = 1
    for i in range(npts + 1):
        line_point = line.Position(i * point["s12"] / float(npts))
        yield Point(line_point["lat2"], line_point["lon2"])


def channel2station(value):
    """
    Helper function converting a channel id to a station id. Will not change
    a passed station id.

    :param value: The channel id as a string.

    >>> channel2station("BW.FURT.00.BHZ")
    'BW.FURT'
    >>> channel2station("BW.FURT")
    'BW.FURT'
    """
    return ".".join(value.split(".")[:2])


def select_component_from_stream(st, component):
    """
    Helper function selecting a component from a Stream an raising the proper
    error if not found.

    This is a bit more flexible then stream.select() as it works with single
    letter channels and lowercase channels.
    """
    component = component.upper()
    component = [tr for tr in st if tr.stats.channel[-1].upper() == component]
    if not component:
        raise LASIFNotFoundError("Component %s not found in Stream." %
                                 component)
    elif len(component) > 1:
        raise LASIFNotFoundError("More than 1 Trace with component %s found "
                                 "in Stream." % component)
    return component[0]


def get_event_filename(event, prefix):
    """
    Helper function generating a descriptive event filename.

    :param event: The event object.
    :param prefix: A prefix for the file, denoting e.g. the event catalog.

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

    return "%s_event_%s_Mag_%.1f_%s-%s-%s-%s.h5" % \
        (prefix, region_name, mag.mag, org.time.year, org.time.month,
         org.time.day, org.time.hour)

def generate_input_files(iteration_name, event_name, comm,
                          simulation_type="forward", previous_iteration=None):
        """
        Generate the input files for one event.

        :param iteration_name: The name of the iteration.
        :param event_name: The name of the event for which to generate the
            input files.
        :param simulation_type: forward, adjoint, step_length
        :param previous_iteration: name of the iteration to copy input files
            from.
        """
        import shutil
        if comm.project.config["mesh_file"] == "multiple":
            mesh_file = os.path.join(comm.project.paths["models"],
                                     "EVENT_SPECIFIC", event_name, "mesh.e")
        else:
            mesh_file = comm.project.config["mesh_file"]

        input_files_dir = comm.project.paths['salvus_input']

        # If previous iteration specified, copy files over and update mesh_file
        # This part could be extended such that other parameters can be
        # updated as well.
        if previous_iteration:
            long_prev_iter_name = comm.iterations.get_long_iteration_name(
                previous_iteration)
            prev_it_dir = os.path.join(input_files_dir, long_prev_iter_name,
                                       event_name, simulation_type)
        if previous_iteration and os.path.exists(prev_it_dir):
            long_iter_name = comm.iterations.get_long_iteration_name(
                iteration_name)
            output_dir = os.path.join(input_files_dir, long_iter_name,
                                      event_name,
                                      simulation_type)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            if not prev_it_dir == output_dir:
                shutil.copyfile(os.path.join(prev_it_dir, "run_salvus.sh"),
                                os.path.join(output_dir, "run_salvus.sh"))
            else:
                print("Previous iteration is identical to current iteration.")
            with open(os.path.join(output_dir, "run_salvus.sh"), "r") as fh:
                cmd_string = fh.read()
            l = cmd_string.split(" ")
            l[l.index("--model-file") + 1] = mesh_file
            l[l.index("--mesh-file") + 1] = mesh_file
            cmd_string = " ".join(l)
            with open(os.path.join(output_dir, "run_salvus.sh"), "w") as fh:
                fh.write(cmd_string)
            return
        elif previous_iteration and not os.path.exists(prev_it_dir):
            print(f"Could not find previous iteration directory for event: "
                  f"{event_name}, generating input files")

        # =====================================================================
        # read weights toml file, get event and list of stations
        # =====================================================================
        asdf_file = comm.waveforms.get_asdf_filename(
            event_name=event_name, data_type="raw")

        import pyasdf
        ds = pyasdf.ASDFDataSet(asdf_file)
        event = ds.events[0]

        # Build inventory of all stations present in ASDF file
        stations = ds.waveforms.list()
        inv = ds.waveforms[stations[0]].StationXML
        for station in stations[1:]:
            inv += ds.waveforms[station].StationXML

        import salvus_seismo

        src_time_func = comm.project. \
            solver_settings["source_time_function_type"]

        if src_time_func == "bandpass_filtered_heaviside":
            salvus_seismo_src_time_func = "heaviside"
        else:
            salvus_seismo_src_time_func = src_time_func

        src = salvus_seismo.Source.parse(
            event,
            sliprate=salvus_seismo_src_time_func)
        recs = salvus_seismo.Receiver.parse(inv)

        solver_settings = comm.project.solver_settings
        if solver_settings["number_of_absorbing_layers"] == 0:
            num_absorbing_layers = None
        else:
            num_absorbing_layers = \
                solver_settings["number_of_absorbing_layers"]

        # Generate the configuration object for salvus_seismo
        if simulation_type == "forward":
            config = salvus_seismo.Config(
                mesh_file=mesh_file,
                start_time=solver_settings["start_time"],
                time_step=solver_settings["time_increment"],
                end_time=solver_settings["end_time"],
                salvus_call=comm.project.
                solver_settings["salvus_call"],
                polynomial_order=solver_settings["polynomial_order"],
                verbose=True,
                dimensions=3,
                num_absorbing_layers=num_absorbing_layers,
                with_anisotropy=comm.project.
                solver_settings["with_anisotropy"],
                wavefield_file_name="wavefield.h5",
                wavefield_fields="adjoint")

        elif simulation_type == "step_length":
            config = salvus_seismo.Config(
                mesh_file=mesh_file,
                start_time=solver_settings["start_time"],
                time_step=solver_settings["time_increment"],
                end_time=solver_settings["end_time"],
                salvus_call=comm.project.
                solver_settings["salvus_call"],
                polynomial_order=solver_settings["polynomial_order"],
                verbose=True,
                dimensions=3,
                num_absorbing_layers=num_absorbing_layers,
                with_anisotropy=comm.project.
                solver_settings["with_anisotropy"])

        # =====================================================================
        # output
        # =====================================================================
        long_iter_name = comm.iterations.get_long_iteration_name(
            iteration_name)

        output_dir = os.path.join(input_files_dir, long_iter_name, event_name,
                                  simulation_type)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        import shutil
        shutil.rmtree(output_dir)

        salvus_seismo.generate_cli_call(
            source=src, receivers=recs, config=config,
            output_folder=output_dir,
            exodus_file=mesh_file)

        write_custom_stf(output_dir, comm)

        run_salvus = os.path.join(output_dir, "run_salvus.sh")
        io_sampling_rate = comm.project. \
            solver_settings["io_sampling_rate_volume"]
        memory_per_rank = comm.project.\
            solver_settings["io_memory_per_rank_in_MB"]
        if comm.project.solver_settings["with_attenuation"]:
            with open(run_salvus, "a") as fh:
                fh.write(f" --with-attenuation")
        if simulation_type == "forward":
            with open(run_salvus, "a") as fh:
                fh.write(f" --io-sampling-rate-volume {io_sampling_rate}"
                         f" --io-memory-per-rank-in-MB {memory_per_rank}"
                         f" --io-file-format bin")


def write_custom_stf(output_dir, comm):
        import toml
        import h5py

        source_toml = os.path.join(output_dir, "source.toml")
        with open(source_toml, "r") as fh:
            source_dict = toml.load(fh)['source'][0]

        location = source_dict['location']
        moment_tensor = source_dict['scale']

        freqmax = 1.0 / comm.project.processing_params["highpass_period"]
        freqmin = 1.0 / comm.project.processing_params["lowpass_period"]

        delta = comm.project.solver_settings["time_increment"]
        npts = comm.project.solver_settings["number_of_time_steps"]

        stf_fct = comm.project.get_project_function(
            "source_time_function")
        stf = comm.project.processing_params["stf"]
        if stf == "bandpass_filtered_heaviside":
            stf = stf_fct(npts=npts, delta=delta,
                          freqmin=freqmin, freqmax=freqmax)
        elif stf == "heaviside":
            stf = stf_fct(npts=npts, delta=delta)
        else:
            raise LASIFError(f"{stf} is not supported by lasif. Use either "
                             f"bandpass_filtered_heaviside or heaviside.")

        stf_mat = np.zeros((len(stf), len(moment_tensor)))
        for i, moment in enumerate(moment_tensor):
            stf_mat[:, i] = stf * moment

        heaviside_file_name = os.path.join(output_dir, "Heaviside.h5")
        f = h5py.File(heaviside_file_name, 'w')

        source = f.create_dataset("source", data=stf_mat)
        source.attrs["dt"] = delta
        source.attrs["location"] = location
        source.attrs["spatial-type"] = np.string_("moment_tensor")
        # Start time in nanoseconds
        source.attrs["starttime"] = -delta * 1.0e9

        f.close()

        # remove source toml and write new one
        os.remove(source_toml)
        source_str = f"source_input_file = \"{heaviside_file_name}\"\n\n" \
                     f"[[source]]\n" \
                     f"name = \"source\"\n" \
                     f"dataset_name = \"/source\""

        with open(source_toml, "w") as fh:
            fh.write(source_str)


def process_two_files_without_parallel_output(ds, other_ds,
                                              process_function,
                                              traceback_limit=3):
    import traceback
    import sys
    """
    Process data in two data sets.

    This is mostly useful for comparing data in two data sets in any
    number of scenarios. It again takes a function and will apply it on
    each station that is common in both data sets. Please see the
    :doc:`parallel_processing` document for more details.

    Can only be run with MPI.

    :type other_ds: :class:`.ASDFDataSet`
    :param other_ds: The data set to compare to.
    :param process_function: The processing function takes two
        parameters: The station group from this data set and
        the matching station group from the other data set.
    :type traceback_limit: int
    :param traceback_limit: The length of the traceback printed if an
        error occurs in one of the workers.
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
        usable_stations = list(
            this_stations.intersection(other_stations))
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
            print(" -> Processing approximately task %i of %i ..." %
                  ((_i * MPI.COMM_WORLD.size + 1), total_job_count),
                  flush=True)
        try:
            result = process_function(
                getattr(ds.waveforms, station),
                getattr(other_ds.waveforms, station))
            # print("Working", flush=True)
        except Exception:
            # print("Not working", flush=True)
            # If an exception is raised print a good error message
            # and traceback to help diagnose the issue.
            msg = ("\nError during the processing of station '%s' "
                   "on rank %i:" % (station, MPI.COMM_WORLD.rank))

            # Extract traceback from the exception.
            exc_info = sys.exc_info()
            stack = traceback.extract_stack(
                limit=traceback_limit)
            tb = traceback.extract_tb(exc_info[2])
            full_tb = stack[:-1] + tb
            exc_line = traceback.format_exception_only(
                *exc_info[:2])
            tb = ("Traceback (At max %i levels - most recent call "
                  "last):\n" % traceback_limit)
            tb += "".join(traceback.format_list(full_tb))
            tb += "\n"
            # A bit convoluted but compatible with Python 2 and
            # 3 and hopefully all encoding problems.
            tb += "".join(
                _i.decode(errors="ignore")
                if hasattr(_i, "decode") else _i
                for _i in exc_line)

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
