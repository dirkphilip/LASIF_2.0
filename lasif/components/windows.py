#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import

import glob
import os
from typing import List

from .component import Component

from ..window_manager_sql import WindowGroupManager
from lasif.utils import process_two_files_without_parallel_output
from lasif.exceptions import LASIFNotFoundError


class WindowsComponent(Component):
    """
    Component dealing with the windows and adjoint sources.

    :param folder: The folder where the files are stored.
    :param communicator: The communicator instance.
    :param component_name: The name of this component for the communicator.
    """

    def __init__(self, communicator, component_name):
        super(WindowsComponent, self).__init__(communicator, component_name)

    def get(self, window_set_name: str):
        """
        Returns the window manager instance for a window set.

        :param window_set_name: The name of the window set.
        :type window_set_name: str
        """
        filename = self.get_window_set_filename(window_set_name)
        return WindowGroupManager(filename)

    def list(self):
        """
        Returns a list of window sets currently
        present within the LASIF project.
        """
        files = [
            os.path.abspath(_i)
            for _i in glob.iglob(
                os.path.join(self.comm.project.paths["windows"], "*.sqlite")
            )
        ]
        window_sets = [
            os.path.splitext(os.path.basename(_i))[0][:] for _i in files
        ]

        return sorted(window_sets)

    def has_window_set(self, window_set_name: str):
        """
        Checks whether a window set is already defined.
        Returns True or False

        :param window_set_name: The name of the window set.
        :type window_set_name: str
        """
        if window_set_name in self.list():
            return True
        return False

    def get_window_set_filename(self, window_set_name: str):
        """
        Retrieves the filename for a given window set

        :param window_set_name: The name of the window set.
        :type window_set_name: str
        :return: filename of the window set
        """
        filename = os.path.join(
            self.comm.project.paths["windows"], window_set_name + ".sqlite"
        )
        return filename

    def write_windows_to_sql(
        self, event_name: str, window_set_name: str, windows: dict
    ):
        """
        Writes windows to the sql database

        :param event_name: The name of the event
        :type event_name: str
        :param window_set_name: The name of the window set
        :type window_set_name: str
        :param windows: The actual windows, structured in a
            dictionary(stations) of dicts(channels) of lists(windowS)
            of tuples (start- and end times)
        :type windows: dict
        """
        window_group_manager = self.get(window_set_name)
        window_group_manager.write_windows_bulk(event_name, windows)

    def read_all_windows(self, event: str, window_set_name: str):
        """
        Return a flat dictionary with all windows for a specific event.
        This should always be
        fairly small.

        :param event: Name of event
        :type event: str
        :param window_set_name: The name of the window set.
        :type window_set_name: str
        """
        window_group_manager = self.get(window_set_name)
        return window_group_manager.get_all_windows_for_event(event_name=event)

    def get_window_statistics(self, window_set_name: str, events: List[str]):
        """
        Get a dictionary with window statistics for an iteration per event.
        Depending on the size of your inversion and chosen iteration,
        this might take a while...

        :param window_set_name: The window_set_name.
        :type window_set_name: str
        :param events: List of event(s)
        :type events: List[str]
        """
        statistics = {}

        for _i, event in enumerate(events):
            print(
                "Collecting statistics for event %i of %i ..."
                % (_i + 1, len(events))
            )

            wm = self.read_all_windows(
                event=event, window_set_name=window_set_name
            )

            # wm is dict with stations/channels/list of start_end tuples
            station_details = self.comm.query.get_all_stations_for_event(event)

            component_window_count = {"E": 0, "N": 0, "Z": 0}
            component_length_sum = {"E": 0, "N": 0, "Z": 0}
            stations_with_windows_count = 0
            stations_without_windows_count = 0
            for station in station_details.keys():
                wins = wm[station] if station in wm.keys() else {}
                has_windows = False
                for channel, windows in wins.items():
                    component = channel[-1].upper()

                    total_length = 0.0
                    for window in windows:
                        total_length += window[1] - window[0]
                    if not total_length > 0.0:
                        continue
                    has_windows = True
                    component_window_count[component] += 1
                    component_length_sum[component] += total_length
                if has_windows:
                    stations_with_windows_count += 1
                else:
                    stations_without_windows_count += 1

            statistics[event] = {
                "total_station_count": len(station_details.keys()),
                "stations_with_windows": stations_with_windows_count,
                "stations_without_windows": stations_without_windows_count,
                "stations_with_vertical_windows": component_window_count["Z"],
                "stations_with_north_windows": component_window_count["N"],
                "stations_with_east_windows": component_window_count["E"],
                "total_window_length": sum(component_length_sum.values()),
                "window_length_vertical_components": component_length_sum["Z"],
                "window_length_north_components": component_length_sum["N"],
                "window_length_east_components": component_length_sum["E"],
            }

        return statistics

    def select_windows(
        self, event: str, iteration_name: str, window_set_name: str, **kwargs
    ):
        """
        Automatically select the windows for the given event and iteration.

        Function must be called with MPI.

        :param event: The event.
        :type event: str
        :param iteration_name: The iteration.
        :type iteration_name: str
        :param window_set_name: The name of the window set to pick into
        :type window_set_name: str
        """
        from lasif.utils import select_component_from_stream

        # from mpi4py import MPI
        import pyasdf

        event = self.comm.events.get(event)

        # Get the ASDF filenames.
        processed_filename = self.comm.waveforms.get_asdf_filename(
            event_name=event["event_name"],
            data_type="processed",
            tag_or_iteration=self.comm.waveforms.preprocessing_tag,
        )
        synthetic_filename = self.comm.waveforms.get_asdf_filename(
            event_name=event["event_name"],
            data_type="synthetic",
            tag_or_iteration=iteration_name,
        )

        if not os.path.exists(processed_filename):
            msg = "File '%s' does not exists." % processed_filename
            raise LASIFNotFoundError(msg)

        if not os.path.exists(synthetic_filename):
            msg = "File '%s' does not exists." % synthetic_filename
            raise LASIFNotFoundError(msg)

        # Load project specific window selection function.
        select_windows = self.comm.project.get_project_function(
            "window_picking_function"
        )

        # Get source time function
        stf_fct = self.comm.project.get_project_function(
            "source_time_function"
        )
        delta = self.comm.project.simulation_settings["time_step_in_s"]
        npts = self.comm.project.simulation_settings["number_of_time_steps"]
        freqmax = (
            1.0 / self.comm.project.simulation_settings["minimum_period_in_s"]
        )
        freqmin = (
            1.0 / self.comm.project.simulation_settings["maximum_period_in_s"]
        )
        stf_trace = stf_fct(
            npts=npts, delta=delta, freqmin=freqmin, freqmax=freqmax
        )

        process_params = self.comm.project.simulation_settings
        minimum_period = process_params["minimum_period_in_s"]
        maximum_period = process_params["maximum_period_in_s"]

        def process(observed_station, synthetic_station):
            obs_tag = observed_station.get_waveform_tags()
            syn_tag = synthetic_station.get_waveform_tags()

            # Make sure both have length 1.
            assert (
                len(obs_tag) == 1
            ), "Station: %s - Requires 1 observed waveform tag. Has %i." % (
                observed_station._station_name,
                len(obs_tag),
            )
            assert (
                len(syn_tag) == 1
            ), "Station: %s - Requires 1 synthetic waveform tag. Has %i." % (
                observed_station._station_name,
                len(syn_tag),
            )

            obs_tag = obs_tag[0]
            syn_tag = syn_tag[0]

            # Finally get the data.
            st_obs = observed_station[obs_tag]
            st_syn = synthetic_station[syn_tag]

            # Extract coordinates once.
            coordinates = observed_station.coordinates

            # Process the synthetics.
            st_syn = self.comm.waveforms.process_synthetics(
                st=st_syn.copy(),
                event_name=event["event_name"],
                iteration=iteration_name,
            )

            all_windows = {}

            for component in ["E", "N", "Z"]:
                try:
                    data_tr = select_component_from_stream(st_obs, component)
                    synth_tr = select_component_from_stream(st_syn, component)

                    if self.comm.project.simulation_settings[
                        "scale_data_to_synthetics"
                    ]:
                        scaling_factor = (
                            synth_tr.data.ptp() / data_tr.data.ptp()
                        )
                        # Store and apply the scaling.
                        data_tr.stats.scaling_factor = scaling_factor
                        data_tr.data *= scaling_factor

                except LASIFNotFoundError:
                    continue

                windows = None
                try:
                    windows = select_windows(
                        data_tr,
                        synth_tr,
                        stf_trace,
                        event["latitude"],
                        event["longitude"],
                        event["depth_in_km"],
                        coordinates["latitude"],
                        coordinates["longitude"],
                        minimum_period=minimum_period,
                        maximum_period=maximum_period,
                        iteration=iteration_name,
                        **kwargs,
                    )
                except Exception as e:
                    print(e)

                if not windows:
                    continue
                all_windows[data_tr.id] = windows

            if all_windows:
                return all_windows

        ds = pyasdf.ASDFDataSet(processed_filename, mode="r", mpi=False)
        ds_synth = pyasdf.ASDFDataSet(synthetic_filename, mode="r", mpi=False)

        results = process_two_files_without_parallel_output(
            ds, ds_synth, process
        )
        MPI.COMM_WORLD.Barrier()
        # Write files on rank 0.
        if MPI.COMM_WORLD.rank == 0:
            print("Finished window selection", flush=True)
        size = MPI.COMM_WORLD.size
        MPI.COMM_WORLD.Barrier()
        for thread in range(size):
            rank = MPI.COMM_WORLD.rank
            if rank == thread:
                print(
                    f"Writing windows for rank: {rank+1} " f"out of {size}",
                    flush=True,
                )
                self.comm.windows.write_windows_to_sql(
                    event_name=event["event_name"],
                    windows=results,
                    window_set_name=window_set_name,
                )
            MPI.COMM_WORLD.Barrier()

    def select_windows_multiprocessing(
        self,
        event: str,
        iteration_name: str,
        window_set_name: str,
        num_processes: int = 16,
        **kwargs,
    ):
        """
        Automatically select the windows for the given event and iteration.
        Uses Python's multiprocessing for parallelization.

        :param event: The event.
        :type event: str
        :param iteration_name: The iteration.
        :type iteration_name: str
        :param window_set_name: The name of the window set to pick into
        :type window_set_name: str
        :param num_processes: The number of processes used in multiprocessing
        :type num_processes: int
        """
        from lasif.utils import select_component_from_stream
        from tqdm import tqdm
        import multiprocessing
        import warnings
        import pyasdf

        warnings.filterwarnings("ignore")

        global _window_select

        event = self.comm.events.get(event)

        # Get the ASDF filenames.
        processed_filename = self.comm.waveforms.get_asdf_filename(
            event_name=event["event_name"],
            data_type="processed",
            tag_or_iteration=self.comm.waveforms.preprocessing_tag,
        )
        synthetic_filename = self.comm.waveforms.get_asdf_filename(
            event_name=event["event_name"],
            data_type="synthetic",
            tag_or_iteration=iteration_name,
        )

        if not os.path.exists(processed_filename):
            msg = "File '%s' does not exists." % processed_filename
            raise LASIFNotFoundError(msg)

        if not os.path.exists(synthetic_filename):
            msg = "File '%s' does not exists." % synthetic_filename
            raise LASIFNotFoundError(msg)

        # Load project specific window selection function.
        select_windows = self.comm.project.get_project_function(
            "window_picking_function"
        )

        # Get source time function
        stf_fct = self.comm.project.get_project_function(
            "source_time_function"
        )
        delta = self.comm.project.simulation_settings["time_step_in_s"]
        npts = self.comm.project.simulation_settings["number_of_time_steps"]
        freqmax = (
            1.0 / self.comm.project.simulation_settings["minimum_period_in_s"]
        )
        freqmin = (
            1.0 / self.comm.project.simulation_settings["maximum_period_in_s"]
        )
        stf_trace = stf_fct(
            npts=npts, delta=delta, freqmin=freqmin, freqmax=freqmax
        )

        process_params = self.comm.project.simulation_settings
        minimum_period = process_params["minimum_period_in_s"]
        maximum_period = process_params["maximum_period_in_s"]

        def _window_select(station):
            ds = pyasdf.ASDFDataSet(processed_filename, mode="r", mpi=False)
            ds_synth = pyasdf.ASDFDataSet(
                synthetic_filename, mode="r", mpi=False
            )
            observed_station = ds.waveforms[station]
            synthetic_station = ds_synth.waveforms[station]

            obs_tag = observed_station.get_waveform_tags()
            syn_tag = synthetic_station.get_waveform_tags()

            try:
                # Make sure both have length 1.
                assert len(obs_tag) == 1, (
                    "Station: %s - Requires 1 observed waveform tag. Has %i."
                    % (observed_station._station_name, len(obs_tag))
                )
                assert len(syn_tag) == 1, (
                    "Station: %s - Requires 1 synthetic waveform tag. Has %i."
                    % (observed_station._station_name, len(syn_tag))
                )
            except AssertionError:
                return {station: None}

            obs_tag = obs_tag[0]
            syn_tag = syn_tag[0]

            # Finally get the data.
            st_obs = observed_station[obs_tag]
            st_syn = synthetic_station[syn_tag]

            # Extract coordinates once.
            coordinates = observed_station.coordinates

            # Process the synthetics.
            st_syn = self.comm.waveforms.process_synthetics(
                st=st_syn.copy(),
                event_name=event["event_name"],
                iteration=iteration_name,
            )

            all_windows = {}
            for component in ["E", "N", "Z"]:
                try:
                    data_tr = select_component_from_stream(st_obs, component)
                    synth_tr = select_component_from_stream(st_syn, component)

                    if self.comm.project.simulation_settings[
                        "scale_data_to_synthetics"
                    ]:
                        scaling_factor = (
                            synth_tr.data.ptp() / data_tr.data.ptp()
                        )
                        # Store and apply the scaling.
                        data_tr.stats.scaling_factor = scaling_factor
                        data_tr.data *= scaling_factor

                except LASIFNotFoundError:
                    continue

                windows = None
                try:
                    windows = select_windows(
                        data_tr,
                        synth_tr,
                        stf_trace,
                        event["latitude"],
                        event["longitude"],
                        event["depth_in_km"],
                        coordinates["latitude"],
                        coordinates["longitude"],
                        minimum_period=minimum_period,
                        maximum_period=maximum_period,
                        iteration=iteration_name,
                        **kwargs,
                    )
                except Exception as e:
                    print(e)

                if not windows:
                    continue
                all_windows[data_tr.id] = windows

            if all_windows:
                return {station: all_windows}
            else:
                return {station: None}

        # Generate task list
        with pyasdf.ASDFDataSet(processed_filename, mode="r", mpi=False) as ds:
            task_list = ds.waveforms.list()

        # Use at most num_processes workers
        number_processes = min(num_processes, multiprocessing.cpu_count())

        # Open Pool of workers
        with multiprocessing.Pool(number_processes) as pool:
            results = {}
            with tqdm(total=len(task_list)) as pbar:
                for i, r in enumerate(
                    pool.imap_unordered(_window_select, task_list)
                ):
                    pbar.update()
                    k, v = r.popitem()
                    results[k] = v

            pool.close()
            pool.join()

        # Write files with a single worker
        print("Finished window selection", flush=True)
        num_sta_with_windows = sum(v is not None for k, v in results.items())
        print(
            f"Writing windows for {num_sta_with_windows} out of "
            f"{len(task_list)} stations."
        )
        self.comm.windows.write_windows_to_sql(
            event_name=event["event_name"],
            windows=results,
            window_set_name=window_set_name,
        )

    def select_windows_for_station(
        self,
        event: str,
        iteration: str,
        station: str,
        window_set_name: str,
        **kwargs,
    ):
        """
        Selects windows for the given event, iteration, and station. Will
        delete any previously existing windows for that station if any.

        :param event: The event.
        :type event: str
        :param iteration: The iteration name.
        :type iteration: str
        :param station: The station id in the form NET.STA.
        :type station: str
        :param window_set_name: Name of window set
        :type window_set_name: str
        """
        from lasif.utils import select_component_from_stream

        # Load project specific window selection function.
        select_windows = self.comm.project.get_project_function(
            "window_picking_function"
        )

        event = self.comm.events.get(event)
        data = self.comm.query.get_matching_waveforms(
            event["event_name"], iteration, station
        )

        # Get source time function
        stf_fct = self.comm.project.get_project_function(
            "source_time_function"
        )
        delta = self.comm.project.simulation_settings["time_step_in_s"]
        npts = self.comm.project.simulation_settings["number_of_time_steps"]
        freqmax = (
            1.0 / self.comm.project.simulation_settings["minimum_period_in_s"]
        )
        freqmin = (
            1.0 / self.comm.project.simulation_settings["maximum_period_in_s"]
        )
        stf_trace = stf_fct(
            npts=npts, delta=delta, freqmin=freqmin, freqmax=freqmax
        )

        process_params = self.comm.project.simulation_settings
        minimum_period = process_params["minimum_period_in_s"]
        maximum_period = process_params["maximum_period_in_s"]

        window_group_manager = self.comm.windows.get(window_set_name)

        found_something = False
        for component in ["E", "N", "Z"]:
            try:
                data_tr = select_component_from_stream(data.data, component)
                synth_tr = select_component_from_stream(
                    data.synthetics, component
                )
                # delete preexisting windows
                window_group_manager.del_all_windows_from_event_channel(
                    event["event_name"], data_tr.id
                )
            except LASIFNotFoundError:
                continue
            found_something = True

            windows = select_windows(
                data_tr,
                synth_tr,
                stf_trace,
                event["latitude"],
                event["longitude"],
                event["depth_in_km"],
                data.coordinates["latitude"],
                data.coordinates["longitude"],
                minimum_period=minimum_period,
                maximum_period=maximum_period,
                iteration=iteration,
                **kwargs,
            )
            if not windows:
                continue

            for starttime, endtime, b_wave in windows:
                window_group_manager.add_window_to_event_channel(
                    event_name=event["event_name"],
                    channel_name=data_tr.id,
                    start_time=starttime,
                    end_time=endtime,
                )

        if found_something is False:
            raise LASIFNotFoundError(
                "No matching data found for event '%s', iteration '%s', and "
                "station '%s'."
                % (event["event_name"], iteration.name, station)
            )
