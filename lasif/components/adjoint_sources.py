#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import

import pyasdf
import os
import numpy as np
from lasif.utils import process_two_files_without_parallel_output

from lasif import LASIFNotFoundError
from .component import Component
from lasif.tools.adjoint.adjoint_source import calculate_adjoint_source

# Map the adjoint source type names to functions implementing them.
# MISFIT_MAPPING = {
#     "TimeFrequencyPhaseMisfitFichtner2008": adsrc_tf_phase_misfit,
#     "L2Norm": adsrc_l2_norm_misfit,
#     "CCTimeShift": adsrc_cc_time_shift,
#     "DoubleDifference": double_difference_adjoint,
#     "L2NormWeighted": adsrc_l2_norm_weighted
# }


class AdjointSourcesComponent(Component):
    """
    Component dealing with the windows and adjoint sources.

    :param folder: The folder where the files are stored.
    :param communicator: The communicator instance.
    :param component_name: The name of this component for the communicator.
    """

    def __init__(self, folder, communicator, component_name):
        self._folder = folder
        super(AdjointSourcesComponent, self).__init__(
            communicator, component_name)

    def get_filename(self, event, iteration):
        """
        Gets the filename for the adjoint source.

        :param event: The event.
        :param iteration: The iteration.
        """
        iteration_long_name = self.comm.iterations.get_long_iteration_name(
            iteration)

        folder = os.path.join(self._folder, iteration_long_name, event)
        if not os.path.exists(folder):
            os.makedirs(folder)

        return os.path.join(
            folder, "adjoint_source_auxiliary.h5")

    def get_misfit_for_event(self, event, iteration, weight_set_name=None):
        """
        This function returns the total misfit for an event.
        :param event: name of the event
        :param iteration: teration for which to get the misfit
        :return: t
        """
        filename = self.get_filename(event=event, iteration=iteration)

        event_weight = 1.0
        if weight_set_name:
            ws = self.comm.weights.get(weight_set_name)
            event_weight = ws.events[event]["event_weight"]
            station_weights = ws.events[event]["stations"]

        if not os.path.exists(filename):
            raise LASIFNotFoundError(f"Could not find {filename}")

        with pyasdf.ASDFDataSet(filename, mode="r") as ds:
            adj_src_data = ds.auxiliary_data["AdjointSources"]
            stations = ds.auxiliary_data["AdjointSources"].list()

            total_misfit = 0.0
            for station in stations:
                channels = adj_src_data[station].list()
                for channel in channels:
                    if weight_set_name:
                        station_weight = \
                            station_weights[".".join(
                                station.split("_"))]["station_weight"]
                        misfit = \
                            adj_src_data[station][channel].parameters[
                                "misfit"] * station_weight
                    else:
                        misfit = \
                            adj_src_data[station][channel].parameters["misfit"]
                    total_misfit += misfit
        return total_misfit * event_weight

    def write_adjoint_sources(self, event, iteration, adj_sources):
        """
        Write an ASDF file
        """
        filename = self.get_filename(event=event, iteration=iteration)

        print("\nStarting to write adjoint sources to ASDF file ...")

        adj_src_counter = 0
        # print(adj_sources)

        # DANGERZONE: manually disable the MPIfile driver for pyasdf as
        # we are already in MPI but only rank 0 will enter here and that
        # will confuse pyasdf otherwise.
        with pyasdf.ASDFDataSet(filename, mpi=False) as ds:
            for value in adj_sources.values():
                if not value:
                    continue
                for c_id, adj_source in value.items():
                    net, sta, loc, cha = c_id.split(".")
                    ds.add_auxiliary_data(
                        data=adj_source["adj_source"],
                        data_type="AdjointSources",
                        path="%s_%s/Channel_%s_%s" % (net, sta, loc, cha),
                        parameters={"misfit": adj_source["misfit"]})
                    adj_src_counter += 1
        print("Wrote %i adjoint_sources to the ASDF file." % adj_src_counter)

    def calculate_adjoint_sources(self, event, iteration, window_set_name,
                                  plot=False, **kwargs):
        from lasif.utils import select_component_from_stream

        from mpi4py import MPI
        import pyasdf

        event = self.comm.events.get(event)

        # Get the ASDF filenames.
        processed_filename = self.comm.waveforms.get_asdf_filename(
            event_name=event["event_name"],
            data_type="processed",
            tag_or_iteration=self.comm.waveforms.preprocessing_tag)
        synthetic_filename = self.comm.waveforms.get_asdf_filename(
            event_name=event["event_name"],
            data_type="synthetic",
            tag_or_iteration=iteration)

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

        process_params = self.comm.project.processing_params

        def process(observed_station, synthetic_station):
            obs_tag = observed_station.get_waveform_tags()
            syn_tag = synthetic_station.get_waveform_tags()

            # Make sure both have length 1.
            assert len(obs_tag) == 1, (
                "Station: %s - Requires 1 observed waveform tag. Has %i." % (
                    observed_station._station_name, len(obs_tag)))
            assert len(syn_tag) == 1, (
                "Station: %s - Requires 1 synthetic waveform tag. Has %i." % (
                    observed_station._station_name, len(syn_tag)))

            obs_tag = obs_tag[0]
            syn_tag = syn_tag[0]

            # Finally get the data.
            st_obs = observed_station[obs_tag]
            st_syn = synthetic_station[syn_tag]

            # Process the synthetics.
            st_syn = self.comm.waveforms.process_synthetics(
                st=st_syn.copy(), event_name=event["event_name"],
                iteration=iteration)

            adjoint_sources = {}
            ad_src_type = self.comm.project.config["misfit_type"]
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

                if self.comm.project.processing_params["scale_data_"
                                                       "to_synthetics"]:
                    if not self.comm.project.config["misfit_type"] == \
                            "L2NormWeighted":
                        scaling_factor = \
                            synth_tr.data.ptp() / data_tr.data.ptp()
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
                        observed=data_tr, synthetic=synth_tr,
                        window=windows,
                        min_period=process_params["highpass_period"],
                        max_period=process_params["lowpass_period"],
                        adj_src_type=ad_src_type,
                        window_set=window_set_name,
                        taper_ratio=0.15, taper_type='cosine',
                        plot=plot, envelope_scaling=env_scaling)
                except:
                    # Either pass or fail for the whole component.
                    continue

                if not asrc:
                    continue
                # Sum up both misfit, and adjoint source.
                misfit = asrc.misfit
                adj_source = asrc.adjoint_source
                # Time reversal is currently needed in Salvus but that will
                # change later and this can be removed
                adj_source = adj_source[::-1]

                adjoint_sources[data_tr.id] = {
                    "misfit": misfit,
                    "adj_source": adj_source
                }

            return adjoint_sources

        ds = pyasdf.ASDFDataSet(processed_filename, mode="r", mpi=False)
        ds_synth = pyasdf.ASDFDataSet(synthetic_filename, mode="r", mpi=False)

        # Launch the processing. This will be executed in parallel across
        # ranks.
        results = process_two_files_without_parallel_output(ds, ds_synth,
                                                            process)
        # Write files on all ranks.
        filename = self.get_filename(
            event=event["event_name"], iteration=iteration)
        ad_src_counter = 0
        size = MPI.COMM_WORLD.size
        MPI.COMM_WORLD.Barrier()
        for thread in range(size):
            rank = MPI.COMM_WORLD.rank
            if rank == thread:
                print(
                    f"Writing adjoint sources for rank: {rank+1} "
                    f"out of {size}", flush=True)
                with pyasdf.ASDFDataSet(filename=filename, mpi=False,
                                        mode="a") as bs:
                    for value in results.values():
                        if not value:
                            continue
                        for c_id, adj_source in value.items():
                            net, sta, loc, cha = c_id.split(".")
                            bs.add_auxiliary_data(
                                data=adj_source["adj_source"],
                                data_type="AdjointSources",
                                path="%s_%s/Channel_%s_%s" % (net, sta,
                                                              loc, cha),
                                parameters={"misfit": adj_source["misfit"]})
                        ad_src_counter += 1

            MPI.COMM_WORLD.barrier()
        if MPI.COMM_WORLD.rank == 0:
            with pyasdf.ASDFDataSet(filename=filename, mpi=False,
                                    mode="a")as ds:
                length = len(ds.auxiliary_data.AdjointSources.list())
            print(f"{length} Adjoint sources are in your file.")

    def finalize_adjoint_sources(self, iteration_name, event_name,
                                 weight_set_name=None):
        """
        Finalizes the adjoint sources.
        """
        import pyasdf
        import h5py

        # This will do stuff for each event and a single iteration
        # Step one, read adj_src file that should have been created already
        print(iteration_name)
        print(event_name)
        print(weight_set_name)
        iteration = self.comm.iterations.\
            get_long_iteration_name(iteration_name)

        adj_src_file = self.\
            get_filename(event_name, iteration)

        ds = pyasdf.ASDFDataSet(adj_src_file, mpi=False)
        adj_srcs = ds.auxiliary_data["AdjointSources"]

        input_files_dir = self.comm.project.paths['adjoint_sources']

        receivers = self.comm.query.get_all_stations_for_event(event_name)

        output_dir = os.path.join(input_files_dir, iteration,
                                  event_name)
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        adjoint_source_file_name = os.path.join(
            output_dir, "stf.h5")

        f = h5py.File(adjoint_source_file_name, 'w')

        event_weight = 1.0
        if weight_set_name:
            ws = self.comm.weights.get(weight_set_name)
            event_weight = ws.events[event_name]["event_weight"]
            station_weights = ws.events[event_name]["stations"]

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
                    if weight_set_name:
                        weight = \
                            station_weights[receiver]["station_weight"] * \
                            event_weight
                        zne *= weight

                    source = f.create_dataset(station, data=zne.T)
                    source.attrs["dt"] = self.comm.project. \
                        solver_settings["time_increment"]
                    source.attrs["sampling_rate_in_hertz"] = 1 / \
                        source.attrs["dt"]
                    # source.attrs['location'] = np.array(
                    #    [receivers[receiver]["s"]])
                    source.attrs['spatial-type'] = np.string_("vector")
                    # Start time in nanoseconds
                    source.attrs['start_time_in_seconds'] = self.comm.\
                        project.solver_settings["start_time"]

                    # toml_string += f"[[source]]\n" \
                    #               f"name = \"{station}\"\n" \
                    #               f"dataset_name = \"/{station}\"\n\n"

        f.close()

    @staticmethod
    def _validate_return_value(adsrc):
        if not isinstance(adsrc, dict):
            return False
        elif sorted(adsrc.keys()) != ["adjoint_source", "details",
                                      "misfit_value"]:
            return False
        elif not isinstance(adsrc["adjoint_source"], np.ndarray):
            return False
        elif not isinstance(adsrc["misfit_value"], float):
            return False
        elif not isinstance(adsrc["details"], dict):
            return False
        return True
