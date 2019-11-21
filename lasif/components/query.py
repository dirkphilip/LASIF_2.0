#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import

import collections
import warnings

from lasif import LASIFError, LASIFNotFoundError, LASIFWarning
import pyasdf

from .component import Component

DataTuple = collections.namedtuple(
    "DataTuple", ["data", "synthetics", "coordinates"]
)


class QueryComponent(Component):
    """
    This component is responsible for making queries across the different
    components and integrating them in a meaningful way.

    :param communicator: The communicator instance.
    :param component_name: The name of this component for the communicator.

    It should thus be initialized fairly late as it needs access to a number
    of other components via the communicator.
    """

    def get_all_stations_for_event(
        self, event_name, list_only=False, event_names_intersection=None
    ):
        """
        Returns a dictionary of all stations for one event and their
        coordinates.

        A station is considered to be available for an event if at least one
        channel has raw data and an associated station file. Furthermore it
        must be possible to derive coordinates for the station.

        :type event_name: str
        :param event_name: Name of the event.
        :param event_names_intersection: Name of events which need to
        intersect receivers with. These will be selected based on equality of
        station code and receiver coordinates.
        """
        waveform_file = self.comm.waveforms.get_asdf_filename(
            event_name=event_name, data_type="raw"
        )

        if event_names_intersection is None:
            # If we don't care about the other events, just proceed
            if list_only:
                with pyasdf.ASDFDataSet(
                    waveform_file, mode="r", mpi=False
                ) as ds:
                    return ds.waveforms.list()

            with pyasdf.ASDFDataSet(waveform_file, mode="r", mpi=False) as ds:
                return ds.get_all_coordinates()
        else:
            # Else, we care about having the same receivers for all events in
            # the same place
            
            # Remove 'original' event from the intersection list to prevent
            # extra work
            event_names_intersection.remove(event_name)

            # Get stations with coordinates
            with pyasdf.ASDFDataSet(waveform_file, mode="r", mpi=False) as ds:
                coordinate_dictionary_event = ds.get_all_coordinates()

            station_codes_set = set(coordinate_dictionary_event.keys())

            # Get filenames of all other events
            intsec_waveform_filenames = [
                self.comm.waveforms.get_asdf_filename(
                    event_name=name, data_type="raw"
                )
                for name in event_names_intersection
            ]

            # Open the datasets of all other events, until we have everything
            # we need ... [*]
            intsec_datasets = [
                pyasdf.ASDFDataSet(
                    intsec_waveform_filename, mode="r", mpi=False
                )
                for intsec_waveform_filename in intsec_waveform_filenames
            ]

            # Extract stations and coordinates
            intsec_coordinate_dictionaries = [
                intsec_dataset.get_all_coordinates()
                for intsec_dataset in intsec_datasets
            ]

            # [*] ... and close datasets
            # [ds.close() for ds in intsec_datasets]

            # Create sets of station codes (without information on the
            # location)
            intsec_station_sets = [
                set(intsec_cor_dict.keys())
                for intsec_cor_dict in intsec_coordinate_dictionaries
            ]

            # Interesect the set of the station codes for the current event
            # with the rest
            intersection_stations_codes = station_codes_set.intersection(
                *intsec_station_sets
            )

            # Convert the intersection to a list
            list_intsec_station_codes = sorted(intersection_stations_codes)

            # Create a copy which we can edit to contain only those stations
            # with the correct coordinates
            updated_list_intsec_station_codes = list_intsec_station_codes

            # Iterate over stations and check the equality of coordinates
            # during all events (some stations might be moved around)
            for station_code_in_intersection in list_intsec_station_codes:

                # Get coordinates of the original event in a dictionary
                coordinates_in_origi_event = coordinate_dictionary_event[
                    station_code_in_intersection
                ]

                # Get coordinates of the other events in a list of dictionaries
                coordinates_in_other_events = [
                    oth_evt_dict[station_code_in_intersection]
                    for oth_evt_dict in intsec_coordinate_dictionaries
                ]

                # Check if the dictionaries are all equal
                if (
                    len(coordinates_in_other_events)
                    == coordinates_in_other_events.count(
                        coordinates_in_other_events[0]
                    )
                    and coordinates_in_other_events[0]
                    == coordinates_in_origi_event
                ):
                    # Coordinates are equivalent for the same station, nothing
                    # to worry about
                    pass
                else:
                    # Coordinates are not equivalent for the same station,
                    # removing the station
                    updated_list_intsec_station_codes.remove(
                        station_code_in_intersection
                    )

            if list_only:
                return updated_list_intsec_station_codes
            else:
                # Filter the existing coordiante dictionary with the
                # intersection and equivalent coordinate stations.
                filtered_coordinate_dictionary_event = {
                    k: v
                    for k, v in coordinate_dictionary_event.items()
                    if k in updated_list_intsec_station_codes
                }
                return filtered_coordinate_dictionary_event

    def get_coordinates_for_station(self, event_name, station_id):
        """
        Get the coordinates for one station.

        Must be in sync with :meth:`~.get_all_stations_for_event`.
        """
        waveform_file = self.comm.waveforms.get_asdf_filename(
            event_name=event_name, data_type="raw"
        )

        with pyasdf.ASDFDataSet(waveform_file, mode="r") as ds:
            return ds.waveforms[station_id].coordinates

    def get_stations_for_all_events(self, intersects=False):
        """
        Returns a dictionary with a list of stations per event.
        """
        events = {}

        if not intersects:
            for event in self.comm.events.list():
                try:
                    data = self.get_all_stations_for_event(
                        event, list_only=True
                    )
                except LASIFNotFoundError:
                    continue
                events[event] = data
        else:
            # If we only want to get the stations that are the same for all
            # events, we can just query the stations once.
            data = self.get_all_stations_for_event(
                self.comm.events.list()[0],
                list_only=True,
                event_names_intersection=self.comm.events.list(),
            )
            # And add them to every event equally
            for event in self.comm.events.list():
                events[event] = data
        return events

    def get_matching_waveforms(self, event, iteration, station_or_channel_id):
        seed_id = station_or_channel_id.split(".")
        if len(seed_id) == 2:
            channel = None
            station_id = station_or_channel_id
        elif len(seed_id) == 4:
            network, station, _, channel = seed_id
            station_id = ".".join((network, station))
        else:
            raise ValueError(
                "'station_or_channel_id' must either have " "2 or 4 parts."
            )

        iteration_long_name = self.comm.iterations.get_long_iteration_name(
            iteration
        )
        event = self.comm.events.get(event)

        # Get the metadata for the processed and synthetics for this
        # particular station.
        data = self.comm.waveforms.get_waveforms_processed(
            event["event_name"],
            station_id,
            tag=self.comm.waveforms.preprocessing_tag,
        )
        # data_fly = self.comm.waveforms.get_waveforms_processed_on_the_fly(
        #     event["event_name"], station_id)

        synthetics = self.comm.waveforms.get_waveforms_synthetic(
            event["event_name"],
            station_id,
            long_iteration_name=iteration_long_name,
        )
        coordinates = self.comm.query.get_coordinates_for_station(
            event["event_name"], station_id
        )

        # Clear data and synthetics!
        for _st, name in ((data, "observed"), (synthetics, "synthetic")):
            # Get all components and loop over all components.
            _comps = set(tr.stats.channel[-1].upper() for tr in _st)
            for _c in _comps:
                traces = [
                    _i for _i in _st if _i.stats.channel[-1].upper() == _c
                ]
                if len(traces) == 1:
                    continue
                elif len(traces) > 1:
                    traces = sorted(traces, key=lambda x: x.id)
                    warnings.warn(
                        "%s data for event '%s', iteration '%s', "
                        "station '%s', and component '%s' has %i traces: "
                        "%s. LASIF will select the first one, but please "
                        "clean up your data."
                        % (
                            name.capitalize(),
                            event["event_name"],
                            iteration,
                            station_id,
                            _c,
                            len(traces),
                            ", ".join(tr.id for tr in traces),
                        ),
                        LASIFWarning,
                    )
                    for tr in traces[1:]:
                        _st.remove(tr)
                else:
                    # Should not happen.
                    raise NotImplementedError

        # Make sure all data has the corresponding synthetics. It should not
        # happen that one has three channels of data but only two channels
        # of synthetics...in that case, discard the additional data and
        # raise a warning.
        temp_data = []
        for data_tr in data:
            component = data_tr.stats.channel[-1].upper()
            synthetic_tr = [
                tr
                for tr in synthetics
                if tr.stats.channel[-1].upper() == component
            ]
            if not synthetic_tr:
                warnings.warn(
                    "Station '%s' has observed data for component '%s' but no "
                    "matching synthetics." % (station_id, component),
                    LASIFWarning,
                )
                continue
            temp_data.append(data_tr)
        data.traces = temp_data

        if len(data) == 0:
            raise LASIFError(
                "No data remaining for station '%s'." % station_id
            )

        # Scale the data if required.
        if self.comm.project.processing_params["scale_data_to_synthetics"]:
            for data_tr in data:
                synthetic_tr = [
                    tr
                    for tr in synthetics
                    if tr.stats.channel[-1].lower()
                    == data_tr.stats.channel[-1].lower()
                ][0]
                scaling_factor = synthetic_tr.data.ptp() / data_tr.data.ptp()
                # Store and apply the scaling.
                data_tr.stats.scaling_factor = scaling_factor
                data_tr.data *= scaling_factor

        data.sort()
        synthetics.sort()

        # Select component if necessary.
        if channel and channel is not None:
            # Only use the last letter of the channel for the selection.
            # Different solvers have different conventions for the location
            # and channel codes.
            component = channel[-1].upper()
            data.traces = [
                i
                for i in data.traces
                if i.stats.channel[-1].upper() == component
            ]
            synthetics.traces = [
                i
                for i in synthetics.traces
                if i.stats.channel[-1].upper() == component
            ]

        return DataTuple(
            data=data, synthetics=synthetics, coordinates=coordinates
        )

    def point_in_domain(self, latitude, longitude, depth):
        """
        Tests if the point is in the domain. Returns True/False

        :param latitude: The latitude of the point.
        :param longitude: The longitude of the point.
        :param depth: The depth of the point
        """
        domain = self.comm.project.domain
        return domain.point_in_domain(
            longitude=longitude, latitude=latitude, depth=depth
        )
