#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import

import copy
import warnings

import os
import glob

import obspy

from .component import Component
from lasif.exceptions import LASIFNotFoundError, LASIFWarning
from obspy.geodetics import FlinnEngdahl
import pyasdf


class EventsComponent(Component):
    """
    Component managing a folder of QuakeML files.

    Each file must adhere to the scheme ``*.xml``.

    :param folder: Folder with QuakeML files.
    :param communicator: The communicator instance.
    :param component_name: The name of this component for the
        communicator.
    """

    def __init__(self, folder, communicator, component_name):
        super(EventsComponent, self).__init__(communicator, component_name)
        self.__event_info_cache = {}
        self.folder = folder

        self.index_values = [
            ("filename"),
            ("event_name"),
            ("latitude"),
            ("longitude"),
            ("depth_in_km"),
            ("origin_time"),
            ("m_rr"),
            ("m_pp"),
            ("m_tt"),
            ("m_rp"),
            ("m_rt"),
            ("m_tp"),
            ("magnitude"),
            ("magnitude_type"),
            ("region"),
        ]

        self.all_events = {}
        self.fill_all_events()

    def fill_all_events(self):
        files = glob.glob(os.path.join(self.folder, "*.h5"))
        for file in files:
            event_name = os.path.splitext(os.path.basename(file))[0]
            self.all_events[event_name] = file

    def _update_cache(self):
        files = glob.glob(os.path.join(self.folder, "*.h5"))
        for filename in files:
            event_name = os.path.splitext(os.path.basename(filename))[0]
            self.get(event_name)

    @staticmethod
    def _extract_index_values_quakeml(filename):
        """
        Reads QuakeML files and extracts some keys per channel. Only one
        event per file is allows.
        """
        with pyasdf.ASDFDataSet(filename, mode="r", mpi=False) as ds:
            event = ds.events[0]

            # Extract information.
            mag = event.preferred_magnitude() or event.magnitudes[0]
            org = event.preferred_origin() or event.origins[0]
            if org.depth is None:
                warnings.warn(
                    "Origin contains no depth. Will be assumed to be 0",
                    LASIFWarning,
                )
                org.depth = 0.0
            if mag.magnitude_type is None:
                warnings.warn(
                    "Magnitude has no specified type. Will be assumed "
                    "to be Mw",
                    LASIFWarning,
                )
                mag.magnitude_type = "Mw"

            # Get the moment tensor.
            fm = event.preferred_focal_mechanism() or event.focal_mechanisms[0]
            mt = fm.moment_tensor.tensor

            event_name = os.path.splitext(os.path.basename(filename))[0]

            return [
                str(filename),
                str(event_name),
                float(org.latitude),
                float(org.longitude),
                float(org.depth / 1000.0),
                float(org.time.timestamp),
                float(mt.m_rr),
                float(mt.m_pp),
                float(mt.m_tt),
                float(mt.m_rp),
                float(mt.m_rt),
                float(mt.m_tp),
                float(mag.mag),
                str(mag.magnitude_type),
                str(FlinnEngdahl().get_region(org.longitude, org.latitude)),
            ]

    def list(self, iteration=None):
        """
        List of all events.
        >>> comm = getfixture('events_comm')
        >>> comm.events.list() #  doctest: +NORMALIZE_WHITESPACE
        ['GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11',
         'GCMT_event_TURKEY_Mag_5.9_2011-5-19-20-15']
        """
        self._update_cache()
        if iteration is not None:
            import toml

            iter_name = self.comm.iterations.get_long_iteration_name(iteration)
            path = os.path.join(
                self.comm.project.paths["iterations"],
                iter_name,
                "events_used.toml",
            )
            if not os.path.exists(path):
                print(
                    "You do not have any iteration toml. "
                    "Will give all events"
                )
                return sorted(self.__event_info_cache.keys())
            iter_events = toml.load(path)
            return sorted(iter_events["events"]["events_used"])
        else:
            return sorted(self.__event_info_cache.keys())

    def count(self, iteration=None):
        """
        Get the number of events managed by this component.
        If iteration given, return number of events in iteration

        >>> comm = getfixture('events_comm')
        >>> comm.events.count()
        2
        """
        if iteration is not None:
            import toml

            iter_name = self.comm.iterations.get_long_iteration_name(iteration)
            path = os.path.join(
                self.comm.project.paths["iterations"],
                iter_name,
                "events_used.toml",
            )
            if not os.path.exists(path):
                print(
                    "You do not have any iteration toml. "
                    "Will give all events"
                )
                return len(self.all_events)
            iter_events = toml.load(path)
            return len(iter_events["events"]["events_used"])
        else:
            return len(self.all_events)

    def has_event(self, event_name):
        """
        Test for existence of an event.
        :type event_name: str
        :param event_name: The name of the event.
        """
        # Make sure  it also works with existing event dictionaries. This
        # has the potential to simplify lots of code.
        self.fill_all_events()
        try:
            event_name = event_name["event_name"]
        except (KeyError, TypeError):
            pass
        return event_name in self.all_events

    def get_all_events(self, iteration=None):
        """
        Returns a dictionary with the key being the event names and the
        values the information about each event, as would be returned by the
        :meth:`~lasif.components.events.EventsComponent.get` method.
        """
        # make sure cache is filled
        self._update_cache()
        if iteration:
            return copy.deepcopy(
                {
                    event: self.__event_info_cache[event]
                    for event in self.list(iteration)
                }
            )
        else:
            return copy.deepcopy(self.__event_info_cache)

    def get(self, event_name):
        """
        Get information about one event.
        This function uses multiple cache layers and is thus very cheap to
        call.
        :type event_name: str
        :param event_name: The name of the event.
        :rtype: dict
        """
        try:
            event_name = event_name["event_name"]
        except (KeyError, TypeError):
            pass

        if event_name not in self.all_events:
            raise LASIFNotFoundError(
                "Event '%s' not known to LASIF." % event_name
            )

        if event_name not in self.__event_info_cache:
            values = dict(
                zip(
                    self.index_values,
                    self._extract_index_values_quakeml(
                        self.all_events[event_name]
                    ),
                )
            )
            values["origin_time"] = obspy.UTCDateTime(values["origin_time"])
            self.__event_info_cache[event_name] = values
        return self.__event_info_cache[event_name]
