import inspect
import numpy as np
import obspy
import os
import pkgutil
import warnings

import matplotlib.pyplot as plt
from obspy.core import Stream
import obspy.signal.filter
from lasif.tools.adjoint.utils import window_trace, generic_adjoint_source_plot


from lasif import LASIFError, LASIFWarning


class AdjointSource(object):
    # Dictionary of available adjoint source. The key is the name, the value
    # a tuple of function, verbose name, and description.
    _ad_srcs = {}

    def __init__(self, adj_src_type, misfit, window_misfits,
                 adjoint_source=None, individual_ad_sources=None):
        """
        Class representing an already calculated adjoint source.

        :param adj_src_type: The type of adjoint source.
        :type adj_src_type:  str
        :param misfit: The misfit value for the entire trace..
        :type misfit: float
        :param window_misfits: Misfit for individual windows
        :type window_misfits: numpy array
        :param adjoint_source: The actual adjoint source from all windows.
        :type adjoint_source: Obspy trace
        :param individual_ad_sources: Adjoint source for each imported window
        :type individual_ad_sources: Obspy stream
        """
        if adj_src_type not in self._ad_srcs:
            raise ValueError("Unknown adjoint source type '%s'." %
                             adj_src_type)
        self.adj_src_type = adj_src_type
        self.adj_src_name = self._ad_srcs[adj_src_type][1]
        self.misfit = misfit
        self.individual_misfits = window_misfits

        self.adjoint_source = adjoint_source
        self.individual_ad_sources = individual_ad_sources

    def __str__(self):
        if self.adjoint_source.stats.network and \
                self.adjoint_source.stats.station:
            station = " at station %s.%s" % (self.adjoint_source.stats.network,
                                             self.adjoint_source.stats.station)
        else:
            station = ""

        if self.adjoint_source is not None:
            adj_src_status = "available with %i samples" % (len(
                self.adjoint_source))
        else:
            adj_src_status = "has not been calculated"

        return (
            "{name} Adjoint Source for component {component}{station}\n"
            "    Misfit: {misfit:.4g}\n"
            "    Adjoint source {adj_src_status}"
        ).format(
            name=self.adj_src_name,
            component=self.adjoint_source.stats.channel[-1],
            station=station,
            misfit=self.misfit,
            adj_src_status=adj_src_status
        )


def calculate_adjoint_source(adj_src_type, observed, synthetic,
                             window, min_period=None, max_period=None,
                             taper=True, taper_type='cosine',
                             adjoint_src=True, plot=False,
                             plot_filename=None, **kwargs):
    """
    Central function of SalvusMisfit used to calculate adjoint sources and
    misfit.

    This function uses the notion of observed and synthetic data to offer a
    nomenclature most users are familiar with. Please note that it is
    nonetheless independent of what the two data arrays actually represent.

    The function tapers the data from ``left_window_border`` to
    ``right_window_border``, both in seconds since the first sample in the
    data arrays.

    :param adj_src_type: The type of adjoint source to calculate.
    :type adj_src_type: str
    :param observed: The observed data.
    :type observed: :class:`obspy.core.trace.Trace`
    :param synthetic: The synthetic data.
    :type synthetic: :class:`obspy.core.trace.Trace`
    :param min_period: The minimum period of the spectral content of the data.
    :type min_period: float
    :param window: starttime and endtime of window(s) potentially including
        weighting for each window.
    :type window: list of tuples
    :param adjoint_src: Only calculate the misfit or also derive
        the adjoint source.
    :type adjoint_src: bool
    :param plot: Also produce a plot of the adjoint source. This will force
        the adjoint source to be calculated regardless of the value of
        ``adjoint_src``.
    :type plot: bool or empty :class:`matplotlib.figure.Figure` instance
    :param plot_filename: If given, the plot of the adjoint source will be
        saved there. Only used if ``plot`` is ``True``.
    :type plot_filename: str
    """
    observed, synthetic = _sanity_checks(observed, synthetic)
    # Keep these as they will need to be imported later

    if adj_src_type not in AdjointSource._ad_srcs:
        raise LASIFError(
            "Adjoint Source type '%s' is unknown. Available types: %s" % (
                adj_src_type, ", ".join(
                    sorted(AdjointSource._ad_srcs.keys()))))

    # window variable should be a list of windows, if it is not make it into
    # a list.
    if not isinstance(window, list):
        window = [window]

    fct = AdjointSource._ad_srcs[adj_src_type][0]

    if plot:
        if len(window) > 1:
            raise LASIFError("Currently plotting is only implemented"
                             "for a single window.")
        adjoint_src = True

    full_ad_src = None
    trace_misfit = 0.0
    window_misfit = []
    individual_adj_srcs = Stream()
    s = 0

    original_observed = observed.copy()
    original_synthetic = synthetic.copy()

    if "envelope_scaling" in kwargs and kwargs["envelope_scaling"]:
        # normalize the trace to [-1,1], reduce source effects
        norm_scaling_fac = 1.0 / np.max(np.abs(synthetic.data))
        original_observed.data *= norm_scaling_fac
        original_synthetic.data *= norm_scaling_fac
        envelope = obspy.signal.filter.envelope(original_observed.data)
        # scale up to the noise, also never divide by 0
        env_weighting = 1.0 / (envelope + np.max(envelope) * 0.001)
        original_observed.data *= env_weighting
        original_synthetic.data *= env_weighting
    for win in window:
        taper_ratio = 0.5 * (min_period / (win[1] - win[0]))
        if taper_ratio > 0.5:
            s += 1
            station_name = observed.stats.network + '.' + \
                observed.stats.station
            msg = f"Window {win} at Station {station_name} might be to " \
                  f"short for your frequency content. Adjoint source " \
                  f"was not calculated because it could result in " \
                  f"high frequency artifacts and wacky misfit measurements."
            warnings.warn(msg)
            if len(window) == 1 or s == len(window):
                adjoint = {"adjoint_source": np.zeros_like(observed.data),
                           "misfit": 0.0}
            else:
                continue

        observed = original_observed.copy()
        synthetic = original_synthetic.copy()

        # The window trace function modifies the passed trace
        observed = window_trace(trace=observed,
                                window=win,
                                taper=taper,
                                taper_ratio=taper_ratio,
                                taper_type=taper_type)
        synthetic = window_trace(trace=synthetic,
                                 window=win,
                                 taper=taper,
                                 taper_ratio=taper_ratio,
                                 taper_type=taper_type)

        adjoint = fct(observed=observed, synthetic=synthetic,
                      window=win, min_period=min_period, max_period=max_period,
                      adjoint_src=adjoint_src, plot=plot,
                      taper=taper, taper_ratio=taper_ratio,
                      taper_type=taper_type)

        if adjoint_src:
            adjoint["adjoint_source"] = window_trace(
                trace=adjoint["adjoint_source"], window=win, taper=taper,
                taper_ratio=taper_ratio, taper_type=taper_type)
            if win == window[0]:
                full_ad_src = adjoint["adjoint_source"]
            else:
                full_ad_src.data += adjoint["adjoint_source"].data
            # individual_adj_srcs.append(adjoint["adjoint_source"])

        window_misfit.append((win[0], win[1], adjoint["misfit"]))
        trace_misfit += adjoint["misfit"]

    if plot:
        time = observed.times()
        generic_adjoint_source_plot(observed=observed.data,
                                    synthetic=synthetic.data,
                                    time=time,
                                    adjoint_source=adjoint["adjoint_source"],
                                    misfit=adjoint["misfit"],
                                    adjoint_source_name=adj_src_type)
    if plot_filename:
        plt.savefig(plot_filename)
    else:
        plt.show()

    if "envelope_scaling" in kwargs and kwargs["envelope_scaling"]:
        full_ad_src.data *= (env_weighting * norm_scaling_fac)

    return AdjointSource(adj_src_type, misfit=trace_misfit,
                         window_misfits=window_misfit,
                         adjoint_source=full_ad_src,
                         individual_ad_sources=individual_adj_srcs)


def _sanity_checks(observed, synthetic):
    """
    Perform a number of basic sanity checks to assure the data is valid
    in a certain sense.

    It checks the types of both, the start time, sampling rate, number of
    samples, ...

    :param observed: The observed data.
    :type observed: :class:`obspy.core.trace.Trace`
    :param synthetic: The synthetic data.
    :type synthetic: :class:`obspy.core.trace.Trace`

    :raises: :class:`~lasif.LASIFError`
    """
    if not isinstance(observed, obspy.Trace):
        # Also accept Stream objects.
        if isinstance(observed, obspy.Stream) and \
                len(observed) == 1:
            observed = observed[0]
        else:
            raise LASIFError(
                "Observed data must be an ObsPy Trace object., not {}"
                "".format(observed))
    if not isinstance(synthetic, obspy.Trace):
        if isinstance(synthetic, obspy.Stream) and \
                len(synthetic) == 1:
            synthetic = synthetic[0]
        else:
            raise LASIFError(
                "Synthetic data must be an ObsPy Trace object.")

    if observed.stats.npts != synthetic.stats.npts:
        raise LASIFError("Observed and synthetic data must have the "
                         "same number of samples.")

    sr1 = observed.stats.sampling_rate
    sr2 = synthetic.stats.sampling_rate

    if abs(sr1 - sr2) / sr1 >= 1E-5:
        raise LASIFError("Observed and synthetic data must have the "
                         "same sampling rate.")

    # Make sure data and synthetics start within half a sample interval.
    if abs(observed.stats.starttime - synthetic.stats.starttime) > \
            observed.stats.delta * 0.5:
        raise LASIFError("Observed and synthetic data must have the "
                         "same starttime.")

    ptp = sorted([observed.data.ptp(), synthetic.data.ptp()])
    if ptp[1] / ptp[0] >= 5:
        warnings.warn("The amplitude difference between data and "
                      "synthetic is fairly large.", LASIFWarning)

    # Also check the components of the data to avoid silly mistakes of
    # users.
    if len(set([observed.stats.channel[-1].upper(),
                synthetic.stats.channel[-1].upper()])) != 1:
        warnings.warn("The orientation code of synthetic and observed "
                      "data is not equal.")

    observed = observed.copy()
    synthetic = synthetic.copy()
    observed.data = np.require(observed.data, dtype=np.float64,
                               requirements=["C"])
    synthetic.data = np.require(synthetic.data, dtype=np.float64,
                                requirements=["C"])

    return observed, synthetic


def _discover_adjoint_sources():
    """
    Discovers the available adjoint sources. This should work no matter if
    lasif is checked out from git, packaged as .egg or for any other
    possibility.
    """
    from lasif.tools.adjoint import adjoint_source_types

    AdjointSource._ad_srcs = {}

    FCT_NAME = "calculate_adjoint_source"
    NAME_ATTR = "VERBOSE_NAME"
    DESC_ATTR = "DESCRIPTION"
    ADD_ATTR = "ADDITIONAL_PARAMETERS"

    path = os.path.join(
        os.path.dirname(inspect.getfile(inspect.currentframe())),
        "adjoint_source_types")
    for importer, modname, _ in pkgutil.iter_modules(
            [path], prefix=adjoint_source_types.__name__ + "."):
        m = importer.find_module(modname).load_module(modname)
        if not hasattr(m, FCT_NAME):
            continue
        fct = getattr(m, FCT_NAME)
        if not callable(fct):
            continue

        name = modname.split('.')[-1]

        if not hasattr(m, NAME_ATTR):
            raise LASIFError(
                "Adjoint source '%s' does not have a variable named %s." %
                (name, NAME_ATTR))

        if not hasattr(m, DESC_ATTR):
            raise LASIFError(
                "Adjoint source '%s' does not have a variable named %s." %
                (name, DESC_ATTR))

        # Add tuple of name, verbose name, and description.
        AdjointSource._ad_srcs[name] = (
            fct,
            getattr(m, NAME_ATTR),
            getattr(m, DESC_ATTR),
            getattr(m, ADD_ATTR) if hasattr(m, ADD_ATTR) else None)


_discover_adjoint_sources()
