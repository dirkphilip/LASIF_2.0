#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import

import pyasdf
import obspy
import os
import numpy as np
from lasif.utils import process_two_files_without_parallel_output
import toml

from lasif.exceptions import LASIFNotFoundError, LASIFError
from .component import Component
from lasif.tools.adjoint.adjoint_source import calculate_adjoint_source
from lasif.utils import select_component_from_stream

TAUPY_MODEL_CACHE = {}


class AdjointSourcesComponent(Component):
    """
    Component dealing with the adjoint sources.

    :param folder: The folder where the files are stored.
    :param communicator: The communicator instance.
    :param component_name: The name of this component for the communicator.
    """

    def __init__(self, folder, communicator, component_name):
        self._folder = folder
        super(AdjointSourcesComponent, self).__init__(
            communicator, component_name
        )

    def get_filename(self, event: str, iteration: str):
        """
        Gets the filename for the adjoint source.

        :param event: The event.
        :type event: str
        :param iteration: The iteration name.
        :type iteration: str
        """
        iteration_long_name = self.comm.iterations.get_long_iteration_name(
            iteration
        )

        folder = os.path.join(self._folder, iteration_long_name, event)
        if not os.path.exists(folder):
            os.makedirs(folder)

        return os.path.join(folder, "adjoint_source_auxiliary.h5")

    def get_misfit_file(self, iteration: str):
        """
        Get path to the iteration misfit file

        :param iteration: Name of iteration
        :type iteration: str
        """
        iteration_name = self.comm.iterations.get_long_iteration_name(
            iteration
        )
        file = (
            self.comm.project.paths["iterations"]
            / iteration_name
            / "misfits.toml"
        )
        if not os.path.exists(file):
            raise LASIFNotFoundError(f"File {file} does not exist")
        return file

    def get_misfit_for_event(
        self,
        event: str,
        iteration: str,
        weight_set_name: str = None,
        include_station_misfit: bool = False,
    ):
        """
        This function returns the total misfit for an event.

        :param event: name of the event
        :type event: str
        :param iteration: iteration for which to get the misfit
        :type iteration: str
        :param weight_set_name: Name of station weights, defaults to None
        :type weight_set_name: str, optional
        :param include_station_misfit: Whether individual station misfits
            should be written down or not, defaults to False
        :type include_station_misfit: bool, optional
        """
        misfit_file = self.get_misfit_file(iteration)
        misfits = toml.load(misfit_file)
        if event not in misfits.keys():
            raise LASIFError(
                f"Misfit has not been computed for event {event}, "
                f"iteration: {iteration}. "
            )
        event_misfit = misfits[event]["event_misfit"]
        if include_station_misfit:
            return misfits[event]
        else:
            return event_misfit

    def calculate_validation_misfits(self,
                                     event: str,
                                     iteration: str):
        """

        This fuction computed the L2 weighted waveform misfit over
        a whole trace. It is meant to provide misfits for validation
        purposes. E.g. to steer regularization parameters.

        :param event: name of the event
        :type event: str
        :param iteration: iteration for which to get the misfit
        :type iteration: str
        """
        from scipy.integrate import simps
        from obspy import geodetics
        from lasif.utils import progress
        min_sn_ratio = 0.05
        event = self.comm.events.get(event)

        # Fill cache if necessary.
        if not TAUPY_MODEL_CACHE:
            from obspy.taup import TauPyModel  # NOQA

            TAUPY_MODEL_CACHE["model"] = TauPyModel("AK135")
        model = TAUPY_MODEL_CACHE["model"]

        # Get the ASDF filenames.
        processed_filename = self.comm.waveforms.get_asdf_filename(
            event_name=event["event_name"],
            data_type="processed",
            tag_or_iteration=self.comm.waveforms.preprocessing_tag,
        )
        synthetic_filename = self.comm.waveforms.get_asdf_filename(
            event_name=event["event_name"],
            data_type="synthetic",
            tag_or_iteration=iteration,
        )

        dt = self.comm.project.simulation_settings["time_step_in_s"]

        ds_syn = pyasdf.ASDFDataSet(synthetic_filename, mode="r", mpi=False)
        ds_obs = pyasdf.ASDFDataSet(processed_filename, mode="r", mpi=False)

        event_latitude = event["latitude"]
        event_longitude = event["longitude"]
        event_depth_in_km = event["depth_in_km"]

        minimum_period = self.comm.project.simulation_settings[
            "minimum_period_in_s"]

        misfit = 0.0
        for i, station in enumerate(ds_obs.waveforms.list()):

            if i % 30 == 0:
                progress(i+1, len(ds_obs.waveforms.list()),
                         status="Computing misfits")
            observed_station = ds_obs.waveforms[station]
            synthetic_station = ds_syn.waveforms[station]

            obs_tag = observed_station.get_waveform_tags()
            syn_tag = synthetic_station.get_waveform_tags()

            try:
                # Make sure both have length 1.
                assert len(obs_tag) == 1, (
                        "Station: %s - Requires 1 observed waveform tag."
                        " Has %i."
                        % (observed_station._station_name, len(obs_tag))
                )
            except AssertionError:
                continue

            assert len(syn_tag) == 1, (
                    "Station: %s - Requires 1 synthetic waveform tag. "
                    "Has %i."
                    % (observed_station._station_name, len(syn_tag))
            )

            obs_tag = obs_tag[0]
            syn_tag = syn_tag[0]

            station_latitude = observed_station.coordinates["latitude"]
            station_longitude = observed_station.coordinates["longitude"]

            st_obs = observed_station[obs_tag]
            st_syn = synthetic_station[syn_tag]

            # Sample points down to 10 points per minimum_period
            # len_s = st_obs[0].stats.endtime - st_obs[0].stats.starttime
            # num_samples_wavelength = 10.0
            # new_sampling_rate = num_samples_wavelength * \
            #                     minimum_period / len_s
            # st_obs = st_obs.resample(new_sampling_rate)
            # st_syn = st_syn.resample(new_sampling_rate)
            # dt = 1.0/new_sampling_rate

            dist_in_deg = geodetics.locations2degrees(
                    station_latitude, station_longitude, event_latitude,
                    event_longitude
                )

            # Get only a couple of P phases which should be the
            # first arrival
            # for every epicentral distance. Its quite a bit faster
            # than calculating
            # the arrival times for every phase.
            # Assumes the first sample is the centroid time of the event.
            ttp = model.get_travel_times(
                    source_depth_in_km=event_depth_in_km,
                    distance_in_degree=dist_in_deg,
                    phase_list=["ttp"],
                )
            # Sort just as a safety measure.
            ttp = sorted(ttp, key=lambda x: x.time)
            first_tt_arrival = ttp[0].time

            # Estimate noise level from waveforms prior to the
            # first arrival.
            idx_end = int(np.ceil((first_tt_arrival - 0.5 * minimum_period) / dt))
            idx_end = max(10, idx_end)
            idx_start = int(
                    np.ceil((first_tt_arrival - 2.5 * minimum_period) / dt))
            idx_start = max(10, idx_start)

            if idx_start >= idx_end:
                idx_start = max(0, idx_end - 10)

            for component in ["E", "N", "Z"]:
                try:
                    data_tr = select_component_from_stream(st_obs, component)
                    synth_tr = select_component_from_stream(st_syn, component)
                except LASIFNotFoundError:
                    continue
                # Scale data to synthetics
                scaling_factor = (synth_tr.data.ptp() / data_tr.data.ptp())
                if np.isinf(scaling_factor):
                    continue

                # Store and apply the scaling.
                data_tr.stats.scaling_factor = scaling_factor
                data_tr.data *= scaling_factor

                data = data_tr.data
                abs_data = np.abs(data)
                noise_absolute = abs_data[idx_start:idx_end].max()
                noise_relative = noise_absolute / abs_data.max()

                if noise_relative > min_sn_ratio:
                    continue

                # normalize the trace to [-1,1], reduce source effects
                # and balance amplitudes
                norm_scaling_fac = 1.0 / np.max(np.abs(synth_tr.data))
                data_tr.data *= norm_scaling_fac
                synth_tr.data *= norm_scaling_fac
                # envelope = obspy.signal.filter.envelope(data_tr.data)

                # scale up to around 1, also never divide by 0
                # by adding regularization term, dependent on noise level
                # env_weighting = 1.0 / (
                #            envelope + np.max(envelope) * 0.2)
                # data_tr.data *= env_weighting
                # synth_tr.data *= env_weighting

                diff = data_tr.data - synth_tr.data
                misfit += 0.5 * simps(y=diff ** 2, dx=data_tr.stats.delta)

        print("\nTotal event misfit: ", misfit)
        return misfit


    def calculate_validation_misfits_multiprocssing(
            self,
            event: str,
            iteration: str, num_processes: int = 12,
            min_sn_ratio: float = 0.1):
        """

        This fuction computed the L2 weighted waveform misfit over
        a whole trace. It is meant to provide misfits for validation
        purposes. E.g. to steer regularization parameters.

        This function implements the same as the above with a few
        changes. We don't use taupy anymore, but instead use the synthetics
        to estimate expected arrival times.

        In addition, we use multiprocessing here

        :param event: name of the event
        :type event: str
        :param iteration: iteration for which to get the misfit
        :type iteration: str
        :param num_processes: Number of threads to use
        :type num_processes: int
        :param min_sn_ratio: Minimum signal to noise ratio
        :type min_sn_ratio: float

        """
        from scipy.integrate import simps
        from lasif.utils import select_component_from_stream
        from tqdm import tqdm
        import multiprocessing
        import warnings
        warnings.filterwarnings("ignore")
        event = self.comm.events.get(event)
        global _process

        # Get the ASDF filenames.
        processed_filename = self.comm.waveforms.get_asdf_filename(
            event_name=event["event_name"],
            data_type="processed",
            tag_or_iteration=self.comm.waveforms.preprocessing_tag,
        )
        synthetic_filename = self.comm.waveforms.get_asdf_filename(
            event_name=event["event_name"],
            data_type="synthetic",
            tag_or_iteration=iteration,
        )

        def _process(station):
            ds = pyasdf.ASDFDataSet(processed_filename, mode="r", mpi=False)
            ds_synth = pyasdf.ASDFDataSet(synthetic_filename, mode="r",
                                          mpi=False)
            observed_station = ds.waveforms[station]
            synthetic_station = ds_synth.waveforms[station]

            # print(observed_station, synthetic_station)
            obs_tag = observed_station.get_waveform_tags()
            syn_tag = synthetic_station.get_waveform_tags()

            misfits = {}
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
                return {station: misfits}

            obs_tag = obs_tag[0]
            syn_tag = syn_tag[0]

            # Finally get the data.
            st_obs = observed_station[obs_tag]
            st_syn = synthetic_station[syn_tag]


            # Process the synthetics.
            st_syn = self.comm.waveforms.process_synthetics(
                st=st_syn.copy(),
                event_name=event["event_name"],
                iteration=iteration,
            )

            for component in ["E", "N", "Z"]:
                try:
                    data_tr = select_component_from_stream(st_obs, component)
                    synth_tr = select_component_from_stream(st_syn, component)

                except LASIFNotFoundError:
                    continue

                # Skip to skip case, where this explodes
                if data_tr.data.ptp() == 0 or synth_tr.data.ptp() == 0:
                    continue

                # Normalize amplitudes to avoid strong influences by amplitudes
                # due to source or propagation distance.
                data_tr.data /= data_tr.data.ptp()
                synth_tr.data /= synth_tr.data.ptp()

                first_tt_arrival = \
                np.where(np.abs(synth_tr.data) > 5e-3 * np.max(np.abs(synth_tr.data)))[0][0]
                idx_end = int(0.8 * first_tt_arrival)
                idx_end = max(idx_end,
                              1)  # ensure at least 1 sample is available
                idx_start = 0

                if idx_start >= idx_end:
                    idx_start = max(0, idx_end - 10)

                abs_data = np.abs(data_tr.data)

                # Return empty window when no data is available
                if np.max(abs_data) == 0.0 or np.max(np.abs(synth_tr.data)) == 0.0:
                    continue

                noise_absolute = abs_data[idx_start:idx_end].max()
                noise_relative = noise_absolute / abs_data.max()

                if noise_relative > min_sn_ratio:
                    continue

                diff = data_tr.data - synth_tr.data
                misfit = 0.5 * simps(y=diff ** 2, dx=data_tr.stats.delta)
                misfits[data_tr.id] = {
                    "misfit": misfit,
                }

            misfit_dict = {station: misfits}
            return misfit_dict

        ds = pyasdf.ASDFDataSet(processed_filename, mode="r", mpi=False)

        # Generate task list
        task_list = ds.waveforms.list()

        # Use at most num_processes
        number_processes = min(num_processes, multiprocessing.cpu_count())

        with multiprocessing.Pool(number_processes) as pool:
            results = {}
            with tqdm(total=len(task_list)) as pbar:
                for i, r in enumerate(pool.imap_unordered(_process, task_list)):
                    pbar.update()
                    k, v = r.popitem()
                    results[k] = v

            pool.close()

        misfit = 0.0
        for station in results.keys():
            for trace in results[station].keys():
                misfit += results[station][trace]["misfit"]

        print("\nTotal event misfit: ", misfit)
        return misfit

    def calculate_adjoint_sources_multiprocessing(
            self,
            event: str,
            iteration: str,
            window_set_name: str,
            num_processes: int,
            plot: bool = False,
            **kwargs,
    ):
        """
        Calculate adjoint sources based on the type of misfit defined in
        the lasif config file.
        The computed misfit for each station is also written down into
        a misfit toml file.
        This function uses multiprocessing for parallelization

        :param event: Name of event
        :type event: str
        :param iteration: Name of iteration
        :type iteration: str
        :param window_set_name: Name of window set
        :type window_set_name: str
        :param num_processes: The number of processes used in multiprocessing
        :type num_processes: int
        :param plot: Should the adjoint source be plotted?, defaults to False
        :type plot: bool, optional
        """
        from lasif.utils import select_component_from_stream
        from tqdm import tqdm
        import multiprocessing
        import warnings
        warnings.filterwarnings("ignore")

        # Globally define the processing function. This is required to enable
        # pickling of a function within a function. Alternatively, a solution
        # can be found that does not utilize a function within a function.
        global _process
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
            tag_or_iteration=iteration,
        )

        if not os.path.exists(processed_filename):
            msg = "File '%s' does not exists." % processed_filename
            raise LASIFNotFoundError(msg)

        if not os.path.exists(synthetic_filename):
            msg = "File '%s' does not exists." % synthetic_filename
            raise LASIFNotFoundError(msg)

        all_windows = self.comm.windows.read_all_windows(
            event=event["event_name"], window_set_name=window_set_name
        )

        if len(all_windows.keys()) == 0:
            raise Exception(f"No windows where found for event "
                            f"{event['event_name']} in window set "
                            f"{window_set_name}.")

        process_params = self.comm.project.simulation_settings

        def _process(station):
            ds = pyasdf.ASDFDataSet(processed_filename, mode="r", mpi=False)
            ds_synth = pyasdf.ASDFDataSet(synthetic_filename, mode="r",
                                          mpi=False)
            observed_station = ds.waveforms[station]
            synthetic_station = ds_synth.waveforms[station]

            # print(observed_station, synthetic_station)
            obs_tag = observed_station.get_waveform_tags()
            syn_tag = synthetic_station.get_waveform_tags()

            adjoint_sources = {}
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
                return {station: adjoint_sources}

            obs_tag = obs_tag[0]
            syn_tag = syn_tag[0]

            # Finally get the data.
            st_obs = observed_station[obs_tag]
            st_syn = synthetic_station[syn_tag]

            # Process the synthetics.
            st_syn = self.comm.waveforms.process_synthetics(
                st=st_syn.copy(),
                event_name=event["event_name"],
                iteration=iteration,
            )

            ad_src_type = self.comm.project.optimization_settings[
                "misfit_type"
            ]
            if ad_src_type == "weighted_waveform_misfit":
                env_scaling = True
                ad_src_type = "waveform_misfit"
            else:
                env_scaling = False

            for component in ["E", "N", "Z"]:
                try:
                    data_tr = select_component_from_stream(st_obs, component)
                    synth_tr = select_component_from_stream(st_syn, component)
                except LASIFNotFoundError:
                    continue

                if self.comm.project.simulation_settings[
                        "scale_data_to_synthetics"]:
                    if (not self.comm.project.optimization_settings[
                                "misfit_type"] == "envelope_misfit"):
                        scaling_factor = (
                                synth_tr.data.ptp() / data_tr.data.ptp()
                        )
                        # Store and apply the scaling.
                        data_tr.stats.scaling_factor = scaling_factor
                        data_tr.data *= scaling_factor

                net, sta, cha = data_tr.id.split(".", 2)
                station = net + "." + sta

                if station not in all_windows:
                    continue
                if data_tr.id not in all_windows[station]:
                    continue
                # Collect all.
                windows = all_windows[station][data_tr.id]
                try:
                    # for window in windows:
                    asrc = calculate_adjoint_source(
                        observed=data_tr,
                        synthetic=synth_tr,
                        window=windows,
                        min_period=process_params["minimum_period_in_s"],
                        max_period=process_params["maximum_period_in_s"],
                        adj_src_type=ad_src_type,
                        window_set=window_set_name,
                        taper_ratio=0.15,
                        taper_type="cosine",
                        plot=plot,
                        envelope_scaling=env_scaling,
                    )
                except:
                    # Either pass or fail for the whole component.
                    continue

                if not asrc:
                    continue
                # Sum up both misfit, and adjoint source.
                misfit = asrc.misfit
                adj_source = asrc.adjoint_source.data

                adjoint_sources[data_tr.id] = {
                    "misfit": misfit,
                    "adj_source": adj_source,
                }
            adj_dict = {station: adjoint_sources}
            return adj_dict

        ds = pyasdf.ASDFDataSet(processed_filename, mode="r", mpi=False)

        # Generate task list
        task_list = ds.waveforms.list()

        # Use at most num_processes
        number_processes = min(num_processes, multiprocessing.cpu_count())

        with multiprocessing.Pool(number_processes) as pool:
            results = {}
            with tqdm(total=len(task_list)) as pbar:
                for i, r in enumerate(pool.imap_unordered(_process, task_list)):
                    pbar.update()
                    k, v = r.popitem()
                    results[k] = v

            pool.close()
            pool.join()

        # Write adjoint sources
        filename = self.get_filename(
            event=event["event_name"], iteration=iteration
        )
        long_iter_name = self.comm.iterations.get_long_iteration_name(
            iteration
        )
        misfit_toml = self.comm.project.paths["iterations"]
        toml_filename = misfit_toml / long_iter_name / "misfits.toml"

        ad_src_counter = 0
        if os.path.exists(toml_filename):
            iteration_misfits = toml.load(toml_filename)
            if event["event_name"] in iteration_misfits.keys():
                iteration_misfits[event["event_name"]][
                    "event_misfit"
                ] = 0.0
            with open(toml_filename, "w") as fh:
                toml.dump(iteration_misfits, fh)

        print("Writing adjoint sources...")
        with pyasdf.ASDFDataSet(filename=filename, mpi=False, mode="a") as bs:
            if toml_filename.exists():
                iteration_misfits = toml.load(toml_filename)
                if event["event_name"] in iteration_misfits.keys():
                    total_misfit = iteration_misfits[
                        event["event_name"]
                    ]["event_misfit"]
                else:
                    iteration_misfits[event["event_name"]] = {}
                    iteration_misfits[event["event_name"]][
                        "stations"
                    ] = {}
                    total_misfit = 0.0
            else:
                iteration_misfits = {}
                iteration_misfits[event["event_name"]] = {}
                iteration_misfits[event["event_name"]]["stations"] = {}
                total_misfit = 0.0
            for value in results.values():
                if not value:
                    continue
                station_misfit = 0.0
                for c_id, adj_source in value.items():
                    net, sta, loc, cha = c_id.split(".")

                    bs.add_auxiliary_data(
                        data=adj_source["adj_source"],
                        data_type="AdjointSources",
                        path="%s_%s/Channel_%s_%s"
                             % (net, sta, loc, cha),
                        parameters={"misfit": adj_source["misfit"]},
                    )
                    station_misfit += adj_source["misfit"]
                    station_name = f"{net}.{sta}"
                iteration_misfits[event["event_name"]]["stations"][
                    station_name
                ] = float(station_misfit)
                ad_src_counter += 1
                total_misfit += station_misfit
            iteration_misfits[event["event_name"]][
                "event_misfit"
            ] = float(total_misfit)
            with open(toml_filename, "w") as fh:
                toml.dump(iteration_misfits, fh)

        with pyasdf.ASDFDataSet(
                filename=filename, mpi=False, mode="a"
        ) as ds:
            length = len(ds.auxiliary_data.AdjointSources.list())
        print(f"{length} Adjoint sources are in your file.")

    def calculate_adjoint_sources(
        self,
        event: str,
        iteration: str,
        window_set_name: str,
        plot: bool = False,
        **kwargs,
    ):
        """
        Calculate adjoint sources based on the type of misfit defined in
        the lasif config file.
        The computed misfit for each station is also written down into
        a misfit toml file.

        :param event: Name of event
        :type event: str
        :param iteration: Name of iteration
        :type iteration: str
        :param window_set_name: Name of window set
        :type window_set_name: str
        :param plot: Should the adjoint source be plotted?, defaults to False
        :type plot: bool, optional
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
            tag_or_iteration=iteration,
        )

        if not os.path.exists(processed_filename):
            msg = "File '%s' does not exists." % processed_filename
            raise LASIFNotFoundError(msg)

        if not os.path.exists(synthetic_filename):
            msg = "File '%s' does not exists." % synthetic_filename
            raise LASIFNotFoundError(msg)

        # Read all windows on rank 0 and broadcast.
        if MPI.COMM_WORLD.rank == 0:
            all_windows = self.comm.windows.read_all_windows(
                event=event["event_name"], window_set_name=window_set_name
            )
        else:
            all_windows = {}
        all_windows = MPI.COMM_WORLD.bcast(all_windows, root=0)

        process_params = self.comm.project.simulation_settings

        def process(observed_station, synthetic_station):
            obs_tag = observed_station.get_waveform_tags()
            syn_tag = synthetic_station.get_waveform_tags()

            # Make sure both have length 1.
            assert len(obs_tag) == 1, (
                "Station: %s - Requires 1 observed waveform tag. Has %i."
                % (observed_station._station_name, len(obs_tag))
            )
            assert len(syn_tag) == 1, (
                "Station: %s - Requires 1 synthetic waveform tag. Has %i."
                % (observed_station._station_name, len(syn_tag))
            )

            obs_tag = obs_tag[0]
            syn_tag = syn_tag[0]

            # Finally get the data.
            st_obs = observed_station[obs_tag]
            st_syn = synthetic_station[syn_tag]

            # Process the synthetics.
            st_syn = self.comm.waveforms.process_synthetics(
                st=st_syn.copy(),
                event_name=event["event_name"],
                iteration=iteration,
            )

            adjoint_sources = {}
            ad_src_type = self.comm.project.optimization_settings[
                "misfit_type"
            ]
            if ad_src_type == "weighted_waveform_misfit":
                env_scaling = True
                ad_src_type = "waveform_misfit"
            else:
                env_scaling = False

            for component in ["E", "N", "Z"]:
                try:
                    data_tr = select_component_from_stream(st_obs, component)
                    synth_tr = select_component_from_stream(st_syn, component)
                except LASIFNotFoundError:
                    continue

                if self.comm.project.simulation_settings[
                    "scale_data_to_synthetics"
                ]:
                    if (
                        not self.comm.project.optimization_settings[
                            "misfit_type"
                        ]
                        == "envelope_misfit"
                    ):
                        scaling_factor = (
                            synth_tr.data.ptp() / data_tr.data.ptp()
                        )
                        # Store and apply the scaling.
                        data_tr.stats.scaling_factor = scaling_factor
                        data_tr.data *= scaling_factor

                net, sta, cha = data_tr.id.split(".", 2)
                station = net + "." + sta

                if station not in all_windows:
                    continue
                if data_tr.id not in all_windows[station]:
                    continue
                # Collect all.
                windows = all_windows[station][data_tr.id]
                try:
                    # for window in windows:
                    asrc = calculate_adjoint_source(
                        observed=data_tr,
                        synthetic=synth_tr,
                        window=windows,
                        min_period=process_params["minimum_period_in_s"],
                        max_period=process_params["maximum_period_in_s"],
                        adj_src_type=ad_src_type,
                        window_set=window_set_name,
                        taper_ratio=0.15,
                        taper_type="cosine",
                        plot=plot,
                        envelope_scaling=env_scaling,
                    )
                except:
                    # Either pass or fail for the whole component.
                    continue

                if not asrc:
                    continue
                # Sum up both misfit, and adjoint source.
                misfit = asrc.misfit
                adj_source = asrc.adjoint_source.data

                adjoint_sources[data_tr.id] = {
                    "misfit": misfit,
                    "adj_source": adj_source,
                }

            return adjoint_sources

        ds = pyasdf.ASDFDataSet(processed_filename, mode="r", mpi=False)
        ds_synth = pyasdf.ASDFDataSet(synthetic_filename, mode="r", mpi=False)

        # Launch the processing. This will be executed in parallel across
        # ranks.
        results = process_two_files_without_parallel_output(
            ds, ds_synth, process
        )
        # Write files on all ranks.
        filename = self.get_filename(
            event=event["event_name"], iteration=iteration
        )
        long_iter_name = self.comm.iterations.get_long_iteration_name(
            iteration
        )
        misfit_toml = self.comm.project.paths["iterations"]
        toml_filename = misfit_toml / long_iter_name / "misfits.toml"

        ad_src_counter = 0
        size = MPI.COMM_WORLD.size
        if MPI.COMM_WORLD.rank == 0:
            if os.path.exists(toml_filename):
                iteration_misfits = toml.load(toml_filename)
                if event["event_name"] in iteration_misfits.keys():
                    iteration_misfits[event["event_name"]][
                        "event_misfit"
                    ] = 0.0
                with open(toml_filename, "w") as fh:
                    toml.dump(iteration_misfits, fh)
        MPI.COMM_WORLD.Barrier()
        for thread in range(size):
            rank = MPI.COMM_WORLD.rank
            if rank == thread:
                print(
                    f"Writing adjoint sources for rank: {rank+1} "
                    f"out of {size}",
                    flush=True,
                )
                with pyasdf.ASDFDataSet(
                    filename=filename, mpi=False, mode="a"
                ) as bs:
                    if toml_filename.exists():
                        iteration_misfits = toml.load(toml_filename)
                        if event["event_name"] in iteration_misfits.keys():
                            total_misfit = iteration_misfits[
                                event["event_name"]
                            ]["event_misfit"]
                        else:
                            iteration_misfits[event["event_name"]] = {}
                            iteration_misfits[event["event_name"]][
                                "stations"
                            ] = {}
                            total_misfit = 0.0
                    else:
                        iteration_misfits = {}
                        iteration_misfits[event["event_name"]] = {}
                        iteration_misfits[event["event_name"]]["stations"] = {}
                        total_misfit = 0.0
                    for value in results.values():
                        if not value:
                            continue
                        station_misfit = 0.0
                        for c_id, adj_source in value.items():
                            net, sta, loc, cha = c_id.split(".")

                            bs.add_auxiliary_data(
                                data=adj_source["adj_source"],
                                data_type="AdjointSources",
                                path="%s_%s/Channel_%s_%s"
                                % (net, sta, loc, cha),
                                parameters={"misfit": adj_source["misfit"]},
                            )
                            station_misfit += adj_source["misfit"]
                            station_name = f"{net}.{sta}"
                        iteration_misfits[event["event_name"]]["stations"][
                            station_name
                        ] = float(station_misfit)
                        ad_src_counter += 1
                        total_misfit += station_misfit
                    iteration_misfits[event["event_name"]][
                        "event_misfit"
                    ] = float(total_misfit)
                    with open(toml_filename, "w") as fh:
                        toml.dump(iteration_misfits, fh)

            MPI.COMM_WORLD.barrier()
        if MPI.COMM_WORLD.rank == 0:
            with pyasdf.ASDFDataSet(
                filename=filename, mpi=False, mode="a"
            ) as ds:
                length = len(ds.auxiliary_data.AdjointSources.list())
            print(f"{length} Adjoint sources are in your file.")

    def finalize_adjoint_sources(
        self, iteration_name: str, event_name: str, weight_set_name: str = None
    ):
        """
        Work with adjoint source in a way that it is written down properly
        into an hdf5 file and prepared for being used as a source time
        function.
        The misfit values and adjoint sources are multiplied by the
        weight of the event and the station.

        :param iteration_name: Name of iteration
        :type iteration_name: str
        :param event_name: Name of event
        :type event_name: str
        :param weight_set_name: Name of station weights, defaults to None
        :type weight_set_name: str, optional
        """
        import pyasdf
        import h5py
        print("Finalizing adjoint sources...")
        # This will do stuff for each event and a single iteration
        # Step one, read adj_src file that should have been created already
        iteration = self.comm.iterations.get_long_iteration_name(
            iteration_name
        )

        adj_src_file = self.get_filename(event_name, iteration)

        ds = pyasdf.ASDFDataSet(adj_src_file, mpi=False)
        adj_srcs = ds.auxiliary_data["AdjointSources"]

        input_files_dir = self.comm.project.paths["adjoint_sources"]

        receivers = self.comm.query.get_all_stations_for_event(event_name)

        output_dir = os.path.join(input_files_dir, iteration, event_name)
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        adjoint_source_file_name = os.path.join(output_dir, "stf.h5")

        f = h5py.File(adjoint_source_file_name, "w")

        event_weight = 1.0
        if weight_set_name is not None:
            ws = self.comm.weights.get(weight_set_name)
            event_weight = ws.events[event_name]["event_weight"]
            station_weights = ws.events[event_name]["stations"]
            computed_misfits = toml.load(self.get_misfit_file(iteration))

        for adj_src in adj_srcs:
            station_name = adj_src.auxiliary_data_type.split("/")[1]
            channels = adj_src.list()

            e_comp = np.zeros_like(adj_src[channels[0]].data[()])
            n_comp = np.zeros_like(adj_src[channels[0]].data[()])
            z_comp = np.zeros_like(adj_src[channels[0]].data[()])

            for channel in channels:
                # check channel and set component
                if channel[-1] == "E":
                    e_comp = adj_src[channel].data[()]
                elif channel[-1] == "N":
                    n_comp = adj_src[channel].data[()]
                elif channel[-1] == "Z":
                    z_comp = adj_src[channel].data[()]
            zne = np.array((z_comp, n_comp, e_comp))
            for receiver in receivers.keys():
                station = receiver.replace(".", "_")
                # station = receiver["network"] + "_" + receiver["station"]

                if station == station_name:
                    # transform_mat = np.array(receiver["transform_matrix"])
                    # xyz = np.dot(transform_mat.T, zne).T

                    # net_dot_sta = \
                    #    receiver["network"] + "." + receiver["station"]
                    if weight_set_name is not None:
                        weight = (
                            station_weights[receiver]["station_weight"]
                            * event_weight
                        )
                        zne *= weight
                        computed_misfits[event_name]["stations"][
                            receiver
                        ] *= weight

                    source = f.create_dataset(station, data=zne.T)
                    source.attrs["dt"] = self.comm.project.simulation_settings[
                        "time_step_in_s"
                    ]
                    source.attrs["sampling_rate_in_hertz"] = (
                        1 / source.attrs["dt"]
                    )

                    # source.attrs['location'] = np.array(
                    #    [receivers[receiver]["s"]])
                    source.attrs["spatial-type"] = np.string_("vector")
                    # Start time in nanoseconds
                    source.attrs[
                        "start_time_in_seconds"
                    ] = self.comm.project.simulation_settings[
                        "start_time_in_s"
                    ]

                    # toml_string += f"[[source]]\n" \
                    #               f"name = \"{station}\"\n" \
                    #               f"dataset_name = \"/{station}\"\n\n"
        if weight_set_name is not None:
            computed_misfits[event_name]["event_misfit"] = np.sum(
                np.array(
                    list(computed_misfits[event_name]["stations"].values())
                )
            )
            with open(self.get_misfit_file(iteration), "w") as fh:
                toml.dump(computed_misfits, fh)

        f.close()

    @staticmethod
    def _validate_return_value(adsrc):
        if not isinstance(adsrc, dict):
            return False
        elif sorted(adsrc.keys()) != [
            "adjoint_source",
            "details",
            "misfit_value",
        ]:
            return False
        elif not isinstance(adsrc["adjoint_source"], np.ndarray):
            return False
        elif not isinstance(adsrc["misfit_value"], float):
            return False
        elif not isinstance(adsrc["details"], dict):
            return False
        return True
