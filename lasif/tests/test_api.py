"""
Test cases for the API interface.

Many of these are similar to the CLI testing case but both need to be
tested if both are to be maintained.
"""

# import pytest
from unittest import mock
import lasif
import os
import toml
import pytest
from lasif.tests.testing_helpers import communicator as comm
from lasif.exceptions import LASIFError


def test_plot_domain(comm):
    vs = "lasif.components.visualizations.VisualizationsComponent."

    with mock.patch(vs + "plot_domain") as patch:
        lasif.api.plot_domain(comm, save=False, inner_boundary=False)
    assert patch.call_count == 1


def test_plot_event(comm):
    """
    Tests if the correct plotting functions are called.
    """
    vs = "lasif.components.visualizations.VisualizationsComponent."

    with mock.patch(vs + "plot_event") as patch:
        lasif.api.plot_event(comm, "event_name")
    patch.assert_called_once_with(
        "event_name", None, intersection_override=None, inner_boundary=False,
    )
    assert patch.call_count == 1

    with mock.patch(vs + "plot_event") as patch:
        lasif.api.plot_event(comm, "event_name", "A")
    patch.assert_called_once_with(
        "event_name", "A", intersection_override=None, inner_boundary=False,
    )
    assert patch.call_count == 1

    with mock.patch(vs + "plot_event") as patch:
        event = lasif.api.list_events(comm, output=True)[0]
        lasif.api.plot_event(comm, event, save=True)
        folder = comm.project.get_output_folder(
            type="event_plots", tag="event", timestamp=False
        )
        file = os.path.join(folder, f"{event}.png")
    patch.assert_called_once_with(
        event, None, intersection_override=None, inner_boundary=False,
    )
    assert patch.call_count == 1
    assert os.path.exists(file)


def test_plot_events(comm):
    # Test the different variations of the plot_events function.
    vs = "lasif.components.visualizations.VisualizationsComponent."

    it = lasif.api.list_iterations(comm, output=True)[0]
    with mock.patch(vs + "plot_events") as patch:
        lasif.api.plot_events(comm, inner_boundary=True)
    patch.assert_called_once_with(
        "map", iteration=None, inner_boundary=True,
    )
    assert patch.call_count == 1

    with mock.patch(vs + "plot_events") as patch:
        lasif.api.plot_events(comm, "time")
    patch.assert_called_once_with(
        "time", iteration=None, inner_boundary=False,
    )
    assert patch.call_count == 1

    with mock.patch(vs + "plot_events") as patch:
        lasif.api.plot_events(comm, "map", it)
    patch.assert_called_once_with(
        "map", iteration=it, inner_boundary=False,
    )
    assert patch.call_count == 1

    # with mock.patch(vs + "plot_events") as patch:
    #     lasif.api.plot_events(comm, "map", save=True)
    #     folder = comm.project.get_output_folder(
    #         type="event_plots", tag="events", timestamp=True
    #     )
    #     file = os.path.join(folder, f"events.png")
    # patch.assert_called_once_with("map", iteration=None)
    # assert patch.call_count == 1
    # assert os.path.exists(file)

    with mock.patch(vs + "plot_events") as patch:
        lasif.api.plot_events(comm, "map", iteration=it, save=True)
        folder = comm.project.get_output_folder(
            type="event_plots", tag="events", timestamp=False
        )
        file = os.path.join(folder, f"events_{it}.png")
    patch.assert_called_once_with("map", iteration=it, inner_boundary=False)
    assert patch.call_count == 1
    assert os.path.exists(file)


def test_plot_station_misfits(comm):
    vs = "lasif.components.visualizations.VisualizationsComponent."
    it = lasif.api.list_iterations(comm, output=True)[0]
    event = lasif.api.list_events(comm, output=True)[0]
    with mock.patch(vs + "plot_station_misfits") as patch:
        lasif.api.plot_station_misfits(comm, "event_name", "iteration")
    patch.assert_called_once_with(
        event_name="event_name", iteration="iteration",
    )
    assert patch.call_count == 1

    with mock.patch(vs + "plot_station_misfits") as patch:
        lasif.api.plot_station_misfits(
            comm, event=event, iteration=it, save=True
        )
        folder = comm.project.get_output_folder(
            type="event_plots", tag="events", timestamp=False
        )
        file = os.path.join(folder, f"misfit_{event}_{it}.png")
    patch.assert_called_once_with(
        event_name=event, iteration=it,
    )
    assert patch.call_count == 1
    assert os.path.exists(file)


def test_plot_raydensity(comm):
    # Misc plotting functionality.
    vs = "lasif.components.visualizations.VisualizationsComponent."
    it = lasif.api.list_iterations(comm, output=True)[0]
    with mock.patch(vs + "plot_raydensity") as patch:
        lasif.api.plot_raydensity(comm, plot_stations=False)
    patch.assert_called_once_with(
        iteration=None,
        plot_stations=False,
        save_plot=True,
        intersection_override=None,
    )
    assert patch.call_count == 1

    with mock.patch(vs + "plot_raydensity") as patch:
        lasif.api.plot_raydensity(comm, plot_stations=True)
    patch.assert_called_once_with(
        iteration=None,
        plot_stations=True,
        save_plot=True,
        intersection_override=None,
    )
    assert patch.call_count == 1

    with mock.patch(vs + "plot_raydensity") as patch:
        lasif.api.plot_raydensity(comm, plot_stations=True, iteration=it)
    patch.assert_called_once_with(
        iteration=it,
        plot_stations=True,
        save_plot=True,
        intersection_override=None,
    )
    assert patch.call_count == 1


def test_plot_all_rays(comm):
    vs = "lasif.components.visualizations.VisualizationsComponent."
    with mock.patch(vs + "plot_all_rays") as patch:
        lasif.api.plot_all_rays(comm, True)
    patch.assert_called_once_with(
        plot_stations=True,
        iteration=None,
        save_plot=True,
        intersection_override=None,
    )
    assert patch.call_count == 1


def test_add_gcmt_events(comm):
    gcmt = "lasif.tools.query_gcmt_catalog."
    with mock.patch(gcmt + "add_new_events") as patch:
        lasif.api.add_gcmt_events(comm, 4, 4.5, 5.5, 100.0)
    patch.assert_called_once_with(
        comm=comm,
        count=4,
        min_magnitude=4.5,
        max_magnitude=5.5,
        min_year=None,
        max_year=None,
        threshold_distance_in_km=100.0,
    )
    assert patch.call_count == 1


def test_download_data(comm):
    down = "lasif.components.downloads.DownloadsComponent."
    num_events = len(lasif.api.list_events(comm, output=True))
    with mock.patch(down + "download_data") as patch:
        lasif.api.download_data(comm)
    assert patch.call_count == num_events


def test_list_events(comm):
    events = lasif.api.list_events(comm, output=True)
    should_be = [
        "GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11",
        "GCMT_event_TURKEY_Mag_5.9_2011-5-19-20-15",
    ]
    assert events == should_be
    it = lasif.api.list_iterations(comm, output=True)[1]
    events = lasif.api.list_events(comm, iteration=it, output=True)
    assert events == ["GCMT_event_TURKEY_Mag_5.9_2011-5-19-20-15"]


def test_calculate_adjoint_sources(comm):
    adjoints = "lasif.components.adjoint_sources.AdjointSourcesComponent."
    it = lasif.api.list_iterations(comm, output=True)[0]
    events = lasif.api.list_events(comm, output=True)
    with mock.patch(adjoints + "calculate_adjoint_sources") as patch:
        with mock.patch(adjoints + "finalize_adjoint_sources") as patch_2:
            lasif.api.calculate_adjoint_sources(comm, it, "A")
    patch.assure_called_once_with(events[0], it, "A")
    patch.assure_called_once_with(events[1], it, "A")
    patch_2.assure_called_once_with(it, events[0], "A")
    patch_2.assure_called_once_with(it, events[1], "A")
    assert patch.call_count == len(events)
    assert patch_2.call_count == len(events)


def test_select_windows(comm):
    window = "lasif.components.windows.WindowsComponent."
    events = lasif.api.list_events(comm, output=True)
    it = "1"
    with mock.patch(window + "select_windows") as patch:
        lasif.api.select_windows(comm, it, "A")
    patch.assure_called_once_with(events[0], it, "A")
    patch.assure_called_once_with(events[1], it, "A")
    assert patch.call_count == 2

    it = "2"
    with mock.patch(window + "select_windows") as patch_2:
        lasif.api.select_windows(comm, it, "A")
    patch_2.assure_called_once_with(events[1], it, "A")
    assert patch_2.call_count == 1


def test_open_gui(comm):
    gugu = "lasif.misfit_gui.misfit_gui."
    with mock.patch(gugu + "launch") as patch:
        lasif.api.open_gui(comm)
    patch.assure_called_once_with(comm)


def test_compute_station_weights(comm):
    weight_set = "jonathan"
    lasif.api.compute_station_weights(
        comm, weight_set=weight_set, iteration="2"
    )
    file = (
        comm.project.paths["weights"]
        / f"WEIGHTS_{weight_set}"
        / f"WEIGHTS_{weight_set}.toml"
    )
    computed_weights = toml.load(file)
    assert (
        computed_weights["event"][0]["name"]
        == "GCMT_event_TURKEY_Mag_5.9_2011-5-19-20-15"
    )
    weights = [x["weight"] for x in computed_weights["event"][0]["station"]]
    should_be = [
        0.6752027078342712,
        1.3870174094097534,
        1.2337253102802304,
        0.7040545724757455,
    ]
    assert set(weights) == set(should_be)


def test_set_up_iteration(comm):
    events = lasif.api.list_events(comm, output=True)
    lasif.api.set_up_iteration(comm, "vote_for_pedro")

    assert comm.iterations.has_iteration("vote_for_pedro")
    lasif.api.set_up_iteration(comm, "vote_for_pedro", remove_dirs=True)
    assert not comm.iterations.has_iteration("vote_for_pedro")

    lasif.api.set_up_iteration(
        comm, "make_america_great_again", events=events[0]
    )
    assert comm.iterations.has_iteration("make_america_great_again")
    assert (
        len(
            [
                lasif.api.list_events(
                    comm, iteration="make_america_great_again", output=True
                )
            ]
        )
        == 1
    )
    lasif.api.set_up_iteration(
        comm, "make_america_great_again", remove_dirs=True
    )
    assert not comm.iterations.has_iteration("make_america_great_again")


def test_list_iterations(comm):
    it = lasif.api.list_iterations(comm, output=True)
    assert it == ["1", "2"]


def test_process_data(comm):
    process = "lasif.components.waveforms.WaveformsComponent."
    events = lasif.api.list_events(comm, output=True)
    with mock.patch(process + "process_data") as patch:
        lasif.api.process_data(comm)
        lasif.api.process_data(comm, iteration="2")
    patch.assure_called_once_with(events)
    patch.assure_called_once_with(events[1])


def test_find_event_mesh(comm):
    exists, mesh = lasif.api.find_event_mesh(comm, "event_name")
    file = (
        comm.project.paths["models"]
        / "EVENT_MESHES"
        / "event_name"
        / "mesh.h5"
    )
    assert not exists
    assert mesh == str(file)


def test_get_simulation_mesh(comm):
    event = lasif.api.list_events(comm, output=True)[1]
    mesh = lasif.api.get_simulation_mesh(comm, event, "2")
    file = (
        comm.project.paths["models"] / "ITERATION_2" / f"{event}" / "mesh.h5"
    )
    assert mesh == str(file)


def test_get_receivers(comm):
    uts = "lasif.utils."
    event = lasif.api.list_events(comm, output=True)[1]
    with mock.patch(uts + "place_receivers") as patch:
        _ = lasif.api.get_receivers(comm, event)
    patch.assure_called_once_with(event, comm)


def test_get_source(comm):
    uts = "lasif.utils."
    event = lasif.api.list_events(comm, output=True)[1]
    with mock.patch(uts + "prepare_source") as patch:
        _ = lasif.api.get_source(comm, event, "1")
    patch.assure_called_once_with(comm, event, "1")


def test_compare_misfits(comm, capsys):
    iterations = lasif.api.list_iterations(comm, output=True)
    with pytest.raises(LASIFError) as excinfo:
        lasif.api.compare_misfits(comm, iterations[0], iterations[1])
    assert "Misfit has not been computed" in str(excinfo.value)
    event = lasif.api.list_events(comm, output=True)[1]
    lasif.api.compare_misfits(comm, iterations[0], iterations[1], event)
    captured = capsys.readouterr()
    assert "18.06%" in captured.out
    assert "Misfit per event" in captured.out
    lasif.api.compare_misfits(comm, iterations[1], iterations[0], event)
    captured = capsys.readouterr()
    assert "22.04%" in captured.out


def test_create_weight_set(comm):
    with mock.patch(
        "lasif.components.weights.WeightsComponent.create_new_weight_set"
    ) as patch:
        lasif.api.create_weight_set(comm, "R")
    assert patch.call_count == 1


def test_list_weight_sets(comm, capsys):
    lasif.api.list_weight_sets(comm)
    captured = capsys.readouterr()
    assert "sw_1" in captured.out


def test_plot_window_statistics(comm):
    stuff = "lasif.components.visualizations.VisualizationsComponent."
    events = lasif.api.list_events(comm, output=True)
    with mock.patch(stuff + "plot_window_statistics") as patch:
        lasif.api.plot_window_statistics(comm, "A")
    patch.assure_called_once_with("A", events, None, False)


def test_plot_windows(comm):
    stuff = "lasif.components.visualizations.VisualizationsComponent."
    event = lasif.api.list_events(comm, output=True)[0]
    with mock.patch(stuff + "plot_windows") as patch:
        lasif.api.plot_windows(comm, event_name=event, window_set="A")
    patch.assure_called_once_with(
        event=event, window_set_name="A", ax=None, distance_bins=500, show=True
    )


def test_get_subset(comm):
    stuff = "lasif.tools.query_gcmt_catalog."
    with mock.patch(stuff + "get_subset_of_events") as patch:
        lasif.api.get_subset(comm, events=["A"], count=1)
    patch.assure_called_once_with(comm, 1, ["A"], None)


def test_create_salvus_simulation(comm):
    event = lasif.api.list_events(comm, output=True)[0]
    iteration = "1"
    adjoint = "lasif.salvus_utils.create_salvus_adjoint_simulation"
    forward = "lasif.salvus_utils.create_salvus_forward_simulation"

    with mock.patch(adjoint) as patch:
        lasif.api.create_salvus_simulation(
            comm,
            event=event,
            iteration=iteration,
            type_of_simulation="adjoint",
        )
    patch.assure_called_once_with(
        comm=comm, event=event, iteration=iteration, mesh=None
    )

    with mock.patch(forward) as patch:
        lasif.api.create_salvus_simulation(
            comm, event=event, iteration=iteration,
        )
    patch.assure_called_once_with(
        comm=comm, event=event, iteration=iteration, mesh=None, side_set=None
    )


def test_submit_salvus_simulation(comm):
    stuff = "lasif.salvus_utils.submit_salvus_simulation"

    with mock.patch(stuff) as patch:
        lasif.api.submit_salvus_simulation(comm, ["hh", "bb"])
    patch.assure_called_once_with(comm=comm, simulations=["hh", "bb"])


def test_validate_data(comm):
    stuff = "lasif.components.validator.ValidatorComponent."
    with mock.patch(stuff + "validate_data") as patch:
        lasif.api.validate_data(comm)
    patch.assure_called_once_with(False, False)
    with mock.patch(stuff + "validate_data") as patch:
        lasif.api.validate_data(comm, False, True)
    patch.assure_called_once_with(False, True)
    with mock.patch(stuff + "validate_data") as patch:
        lasif.api.validate_data(comm, False, False, True)
    patch.assure_called_once_with(True, True)
