#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import

import glob
import os
import numpy as np


from lasif.exceptions import LASIFNotFoundError, LASIFError
from .component import Component


class WeightsComponent(Component):
    """
    Component dealing with station and event weights.

    :param weights_folder: The folder with the weight toml files.
    :type weights_folder: pathlib.Path
    :param communicator: The communicator instance.
    :param component_name: The name of this component for the communicator.
    """

    def __init__(self, weights_folder, communicator, component_name):
        self.__cached_weights = {}
        self._folder = weights_folder
        super(WeightsComponent, self).__init__(communicator, component_name)

    def get_filename_for_weight_set(self, weight_set: str):
        """
        Helper function returning the filename of a weight set.

        :param weight_set: Name of weight set
        :type weight_set: str
        """
        long_weight_set_name = self.get_long_weight_set_name(weight_set)
        folder = self._get_folder_for_weight_set(weight_set)
        return os.path.join(
            folder, long_weight_set_name + os.path.extsep + "toml"
        )

    def _get_folder_for_weight_set(self, weight_set_name: str):
        """
        Helper function returning the path of a weights folder.

        :param weight_set_name: Name of weight set
        :type weight_set_name: str
        """
        long_weight_set_name = self.get_long_weight_set_name(weight_set_name)
        folder = os.path.join(
            self.comm.project.paths["weights"], long_weight_set_name
        )
        return folder

    def create_folder_for_weight_set(self, weight_set_name: str):
        """
        Create the folder needed for weight set

        :param weight_set_name: name of weight set
        :type weight_set_name: str
        """
        long_weight_set_name = self.get_long_weight_set_name(weight_set_name)
        folder = os.path.join(
            self.comm.project.paths["weights"], long_weight_set_name
        )
        if not os.path.exists(folder):
            os.makedirs(folder)

    def get_weight_set_dict(self):
        """
        Returns a dictionary with the keys being the weight_set names and the
        values the weight_set filenames.
        """
        files = [
            os.path.abspath(_i)
            for _i in glob.iglob(
                os.path.join(
                    self.comm.project.paths["weights"],
                    "WEIGHTS_*/WEIGHTS_*%stoml" % os.extsep,
                )
            )
        ]
        weight_dict = {
            os.path.splitext(os.path.basename(_i))[0][8:]: _i for _i in files
        }
        return weight_dict

    def get_long_weight_set_name(self, weight_set_name: str):
        """
        Returns the long form of a weight set from its short name.

        :param weight_set_name: Name of weight set
        :type weight_set_name: str
        """
        return "WEIGHTS_%s" % weight_set_name

    def list(self):
        """
        Get a list of all weight sets managed by this component.
        """
        return sorted(self.get_weight_set_dict().keys())

    def count(self):
        """
        Get the number of weight sets managed by this component.
        """
        return len(self.get_weight_set_dict())

    def has_weight_set(self, weight_set_name: str):
        """
        Test for existence of a weight_set.

        :param weight_set_name: The name of the weight_set.
        :type weight_set_name: str
        """
        # Make it work with both the long and short version of the iteration
        # name, and existing iteration object.
        try:
            weight_set_name = weight_set_name.weight_set_name
        except AttributeError:
            pass
        weight_set_name = weight_set_name.replace("WEIGHTS_", "")

        return weight_set_name in self.get_weight_set_dict()

    def create_new_weight_set(self, weight_set_name: str, events_dict: dict):
        """
        Creates a new weight set.

        :param weight_set_name: The name of the weight set.
        :type weight_set_name: str
        :param events_dict: A dictionary specifying the used events.
        :type events_dict: dict
        """
        weight_set_name = str(weight_set_name)
        if weight_set_name in self.get_weight_set_dict():
            msg = "Weight set %s already exists." % weight_set_name
            raise LASIFError(msg)

        self.create_folder_for_weight_set(weight_set_name)

        from lasif.weights_toml import create_weight_set_toml_string

        with open(
            self.get_filename_for_weight_set(weight_set_name), "wt"
        ) as fh:
            fh.write(
                create_weight_set_toml_string(weight_set_name, events_dict)
            )

    def change_weight_set(
        self, weight_set_name: str, weight_set: dict, events_dict: dict
    ):
        """
        Changes an existing weight set. Writes into a tempfile and if
        successful it will replace the old file with the tempfile.

        :param weight_set_name: The name of the weight set.
        :type weigth_set_name: str
        :param weight_set: The actual weight set
        :type weight_set: dict
        :param events_dict: A dictionary specifying the used events.
        :type events_dict: dict
        """
        weight_set_name = str(weight_set_name)

        from lasif.weights_toml import replace_weight_set_toml_string

        temp = self.get_filename_for_weight_set(weight_set_name)
        temp = temp + "_tmp"
        with open(temp, "w+") as fh:
            fh.write(
                replace_weight_set_toml_string(
                    weight_set_name, events_dict, weight_set
                )
            )
        os.remove(self.get_filename_for_weight_set(weight_set_name))
        os.rename(temp, self.get_filename_for_weight_set(weight_set_name))

    def calculate_station_weight(
        self, lat_1: float, lon_1: float, locations: np.ndarray
    ):
        """
        Calculates the weight set for a set of stations for one event

        :param lat_1: latitude of station
        :type lat_1: float
        :param lon_1: longitude of station
        :type lon_1: float
        :param locations: array of latitudes and longitudes of other stations
        :type locations: numpy.ndarray
        :return: weight. weight for this specific station
        :rtype: float
        """
        from obspy.geodetics import locations2degrees

        distance = np.zeros_like(locations[1, :])

        distance = 1.0 / (
            1.0
            + locations2degrees(lat_1, lon_1, locations[0, :], locations[1, :])
        )
        factor = np.sum(distance) - 1.0
        weight = 1.0 / factor

        assert np.all(weight >= 0.0)

        return weight

    def get(self, weight_set_name: str):
        """
        Returns a weight_set object.

        :param iteration_name: The name of the iteration to retrieve.
        :type iteration_name: str
        """
        # Make it work with both the long and short version of the iteration
        # name, and existing iteration object.
        try:
            weight_set_name = str(weight_set_name.weight_set_name)
        except AttributeError:
            weight_set_name = str(weight_set_name)
            weight_set_name = weight_set_name.replace("WEIGHTS_", "")

        # Access cache.
        if weight_set_name in self.__cached_weights:
            return self.__cached_weights[weight_set_name]

        weights_dict = self.get_weight_set_dict()
        if weight_set_name not in weights_dict:
            msg = "Weights '%s' not found." % weight_set_name
            raise LASIFNotFoundError(msg)

        from lasif.weights_toml import WeightSet

        weight_set = WeightSet(weights_dict[weight_set_name])

        # Store in cache.
        self.__cached_weights[weight_set_name] = weight_set

        return weight_set
