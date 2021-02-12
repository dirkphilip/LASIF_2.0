from lasif.utils import place_receivers, prepare_source
from typing import Union, List, Dict
import os
from lasif.exceptions import LASIFError
import toml


def create_salvus_forward_simulation(
    comm: object, event: str, iteration: str, mesh=None, side_set: str = None,
):
    """
    Create a Salvus simulation object based on simulation and salvus
    specific parameters specified in config file.

    :param comm: The lasif communicator object
    :type comm: object
    :param event: Name of event
    :type event: str
    :param iteration: Name of iteration
    :type iteration: str
    :param mesh: Path to mesh or Salvus mesh object, if None it will use
        the domain file from config file, defaults to None
    :type mesh: Union[str, salvus.mesh.unstructured_mesh.UnstructuredMesh],
        optional
    :param side_set: Name of side set on mesh to place receivers,
        defaults to None.
    :type side_set: str, optional
    """
    import salvus.flow.simple_config as sc
    from salvus.flow.simple_config import stf

    source_info = prepare_source(comm=comm, event=event, iteration=iteration)
    iteration = comm.iterations.get_long_iteration_name(iteration)
    receivers = place_receivers(comm=comm, event=event)
    stf_path = os.path.join(
        comm.project.paths["salvus_files"], iteration, "stf.h5"
    )

    if mesh is None:
        mesh = comm.project.lasif_config["domain_settings"]["domain_file"]

    if side_set is None:
        recs = [
            sc.receiver.seismology.Point3D(
                latitude=rec["latitude"],
                longitude=rec["longitude"],
                network_code=rec["network-code"],
                station_code=rec["station-code"],
                fields=["displacement"],
            )
            for rec in receivers
        ]
    else:
        recs = [
            sc.receiver.seismology.SideSetPoint3D(
                latitude=rec["latitude"],
                longitude=rec["longitude"],
                network_code=rec["network-code"],
                station_code=rec["station-code"],
                side_set_name=side_set,
                fields=["displacement"],
            )
            for rec in receivers
        ]

    sources = [
        sc.source.seismology.MomentTensorPoint3D(
            latitude=src["latitude"],
            longitude=src["longitude"],
            depth_in_m=src["depth_in_m"],
            mrr=src["mrr"],
            mtt=src["mtt"],
            mpp=src["mpp"],
            mtp=src["mtp"],
            mrp=src["mrp"],
            mrt=src["mrt"],
            source_time_function=stf.Custom(
                filename=stf_path, dataset_name="/source"
            ),
        )
        for src in source_info
    ]

    w = sc.simulation.Waveform(mesh=mesh, sources=sources, receivers=recs,)
    sim_set = comm.project.simulation_settings
    sal_set = comm.project.salvus_settings
    w.physics.wave_equation.end_time_in_seconds = sim_set["end_time_in_s"]
    w.physics.wave_equation.time_step_in_seconds = sim_set["time_step_in_s"]
    w.physics.wave_equation.start_time_in_seconds = sim_set["start_time_in_s"]
    w.physics.wave_equation.attenuation = sal_set["attenuation"]

    import lasif.domain

    domain = lasif.domain.HDF5Domain(
        mesh, sal_set["absorbing_boundaries_in_km"]
    )
    boundaries = []
    if not domain.is_global_domain():
        if (
                "inner_boundary" in comm.project.domain.get_side_set_names()
        ):
            side_sets = ["inner_boundary"]
        else:
            side_sets = [
                "r0",
                "t0",
                "t1",
                "p0",
                "p1",
            ]

        absorbing = sc.boundary.Absorbing(
            width_in_meters=comm.project.salvus_settings[
                "absorbing_boundaries_in_km"
            ]
            * 1000.0,
            side_sets=side_sets,
            taper_amplitude=1.0
            / comm.project.simulation_settings["minimum_period_in_s"],
        )
        boundaries.append(absorbing)

    if comm.project.ocean_loading:
        ocean_loading = sc.boundary.OceanLoading(side_sets=["r1_ol"])
        boundaries.append(ocean_loading)

    w.physics.wave_equation.boundaries = boundaries

    # w.output.memory_per_rank_in_MB = 4000.0
    w.output.volume_data.format = "hdf5"
    w.output.volume_data.filename = "output.h5"
    w.output.volume_data.fields = ["adjoint-checkpoint"]
    w.output.volume_data.sampling_interval_in_time_steps = (
        "auto-for-checkpointing"
    )

    w.validate()
    return w



def get_adjoint_source(
    comm: object, event: str, iteration: str
) -> List[object]:
    """
    Get a list of adjoint source objects

    :param comm: The lasif communicator object
    :type comm: object
    :param event: Name of event
    :type event: str
    :param iteration: Name of iteration
    :type iteration: str
    :return: Adjoint source objects
    :type return: List[object]
    """
    from salvus.flow.simple_config import source, stf
    import h5py

    receivers = place_receivers(comm=comm, event=event)
    iteration_name = comm.iterations.get_long_iteration_name(iteration)
    adjoint_filename = str(
        comm.project.paths["adjoint_sources"]
        / iteration_name
        / event
        / "stf.h5"
    )

    p = h5py.File(adjoint_filename, "r")
    adjoint_recs = list(p.keys())
    adjoint_sources = []
    for rec in receivers:
        if rec["network-code"] + "_" + rec["station-code"] in adjoint_recs:
            adjoint_sources.append(rec)
    p.close()

    adj_src = [
        source.seismology.VectorPoint3DZNE(
            latitude=rec["latitude"],
            longitude=rec["longitude"],
            fz=1.0,
            fn=1.0,
            fe=1.0,
            source_time_function=stf.Custom(
                filename=adjoint_filename,
                dataset_name=f"/{rec['network-code']}_{rec['station-code']}",
            ),
        )
        for rec in adjoint_sources
    ]
    return adj_src


def create_salvus_adjoint_simulation(
    comm: object, event: str, iteration: str, mesh=None,
) -> object:
    """
    Create a Salvus simulation object based on simulation and salvus
    specific parameters specified in config file.

    :param comm: The lasif communicator object
    :type comm: object
    :param event: Name of event
    :type event: str
    :param iteration: Name of iteration
    :type iteration: str
    :param mesh: Path to mesh or Salvus mesh object, if None it will use
        the domain file from config file, defaults to None
    :type mesh: Union(str, salvus.mesh.unstructured_mesh.UnstructuredMesh),
        optional
    """
    import salvus.flow.api
    from salvus.flow.simple_config import simulation

    site_name = comm.project.salvus_settings["site_name"]
    forward_job_dict = _get_job_dict(
        comm=comm, iteration=iteration, sim_type="forward",
    )
    if "array_name" in forward_job_dict.keys():
        fwd_job_array = salvus.flow.api.get_job_array(
            site_name=site_name, job_array_name=forward_job_dict["array_name"],
        )
        fwd_job_names = [j.job_name for j in fwd_job_array.jobs]
        fwd_job_name = forward_job_dict[event]
        if fwd_job_name not in fwd_job_names:
            raise LASIFError(f"{fwd_job_name} not in job_array names")
        fwd_job_array_index = fwd_job_names.index(fwd_job_name)
        fwd_job_path = fwd_job_array.jobs[fwd_job_array_index].output_path
    else:
        fwd_job_path = salvus.flow.api.get_job(
            site_name=site_name, job_name=forward_job_dict[event]
        ).output_path

    meta = fwd_job_path / "meta.json"

    if mesh is None:
        mesh = comm.project.lasif_config["domain_settings"]["domain_file"]

    w = simulation.Waveform(mesh=mesh)
    w.adjoint.forward_meta_json_filename = f"REMOTE:{meta}"
    w.adjoint.gradient.parameterization = comm.project.salvus_settings[
        "gradient_parameterization"
    ]
    w.adjoint.gradient.output_filename = "gradient.h5"

    adj_src = get_adjoint_source(comm=comm, event=event, iteration=iteration)
    w.adjoint.point_source = adj_src

    w.validate()

    return w


def submit_salvus_simulation(
    comm: object,
    simulations: Union[List[object], object],
    events: Union[List[str], str],
    iteration: str,
    sim_type: str,
    verbosity: int = 1,
) -> object:
    """
    Submit a Salvus simulation to the machine defined in config file
    with details specified in config file

    :param comm: The Lasif communicator object
    :type comm: object
    :param simulations: Simulation object
    :type simulations: Union[List[object], object]
    :param events: We need names of events for the corresponding simulations
        in order to keep tabs on which simulation object corresponds to which
        event.
    :type events: Union[List[str], str]
    :param iteration: Name of iteration, this is needed to know where to
        download files to when jobs are done.
    :type iteration: str
    :param sim_type: can be either forward or adjoint.
    :type sim_type: str
    :return: SalvusJob object or an array of them
    :rtype: object
    """
    from salvus.flow.api import run_async, run_many_async

    if sim_type not in ["forward", "adjoint"]:
        raise LASIFError("sim_type needs to be forward or adjoint")

    array = False
    if isinstance(simulations, list):
        array = True
        if not isinstance(events, list):
            raise LASIFError(
                "If simulations are a list, events need to be a list as well,"
                " with the corresponding events in the same order"
            )
    else:
        if isinstance(events, list):
            raise LASIFError(
                "If there is only one simulation object, "
                "there should be only one event"
            )

    iteration = comm.iterations.get_long_iteration_name(iteration)

    if sim_type == "forward":
        toml_file = (
            comm.project.paths["salvus_files"]
            / iteration
            / "forward_jobs.toml"
        )
    elif sim_type == "adjoint":
        toml_file = (
            comm.project.paths["salvus_files"]
            / iteration
            / "adjoint_jobs.toml"
        )

    if os.path.exists(toml_file):
        jobs = toml.load(toml_file)
    else:
        jobs = {}

    site_name = comm.project.salvus_settings["site_name"]
    ranks = comm.project.salvus_settings["ranks"]
    wall_time = comm.project.salvus_settings["wall_time_in_s"]

    if array:
        job = run_many_async(
            site_name=site_name,
            input_files=simulations,
            ranks_per_job=ranks,
            wall_time_in_seconds_per_job=wall_time,
            verbosity=verbosity,
        )
        jobs["array_name"] = job.job_array_name
        for _i, j in enumerate(job.jobs):
            jobs[events[_i]] = j.job_name
    else:
        job = run_async(
            site_name=site_name,
            input_file=simulations,
            ranks=ranks,
            wall_time_in_seconds=wall_time,
            verbosity=verbosity,
        )
        jobs[events] = job.job_name

    with open(toml_file, mode="w") as fh:
        toml.dump(jobs, fh)
        print(f"Wrote job information into {toml_file}")
    return job


def _get_job_dict(comm: object, iteration: str, sim_type: str) -> dict:
    """
    Get dictionary with the job names
    """
    if sim_type not in ["forward", "adjoint"]:
        raise LASIFError("sim_type can only be forward or adjoint")
    iteration = comm.iterations.get_long_iteration_name(iteration)
    toml_file = (
        comm.project.paths["salvus_files"]
        / iteration
        / f"{sim_type}_jobs.toml"
    )

    if not os.path.exists(toml_file):
        raise LASIFError(f"Path {toml_file} does not exist")

    job_dict = toml.load(toml_file)

    return job_dict


def check_job_status(
    comm: object, events: Union[List[str], str], iteration: str, sim_type: str
) -> Dict[str, str]:
    """
    Check on the statuses of jobs which have been submitted before.

    :param comm: The Lasif communicator object
    :type comm: object
    :param events: We need names of events for the corresponding simulations
        in order to keep tabs on which simulation object corresponds to which
        event.
    :type events: Union[List[str], str]
    :param iteration: Name of iteration, this is needed to know where to
        download files to when jobs are done.
    :type iteration: str
    :param sim_type: can be either forward or adjoint.
    :type sim_type: str
    :return: Statuses of jobs
    :return type: Dict[str]
    """
    import salvus.flow.api

    job_dict = _get_job_dict(comm=comm, iteration=iteration, sim_type=sim_type)

    if not isinstance(events, list):
        events = [events]

    site_name = comm.project.salvus_settings["site_name"]
    statuses = {}

    if "array_name" in job_dict.keys():
        jobs = salvus.flow.api.get_job_array(
            job_array_name=job_dict["array_name"], site_name=site_name
        )
        jobs.update_status(force_update=True)
        job_names = [j.job_name for j in jobs.jobs]

        for event in events:
            job_name = job_dict[event]
            if job_name not in job_names:
                print(
                    f"{job_name} not in array {job_dict['array_name']}. "
                    f"Will check to see if job was posted individually"
                )
                job = salvus.flow.api.get_job(
                    job_name=job_name, site_name=site_name,
                )
                job_updated = job.update_status(force_update=True)
                statuses[event] = job_updated
                continue
                raise LASIFError(f"{job_name} not in List of job names")
            event_job_index = job_names.index(job_name)
            event_status = jobs.jobs[event_job_index].get_status_from_db()
            statuses[event] = event_status

    else:
        for event in events:
            job_name = job_dict[event]
            job = salvus.flow.api.get_job(
                job_name=job_name, site_name=site_name
            )
            job_updated = job.update_status(force_update=True)
            statuses[event] = job_updated

    return statuses


def download_output(comm: object, event: str, iteration: str, sim_type: str):
    """
    Download output

    :param comm: The Lasif communicator object
    :type comm: object
    :param event: Name of event to download output from
    :type event: str
    :param iteration: Name of iteration, this is needed to know where to
        download files to.
    :type iteration: str
    :param sim_type: can be either forward or adjoint.
    :type sim_type: str
    """
    import salvus.flow.api

    job_dict = _get_job_dict(comm=comm, iteration=iteration, sim_type=sim_type)
    site_name = comm.project.salvus_settings["site_name"]
    job_name = job_dict[event]
    iteration = comm.iterations.get_long_iteration_name(iteration)

    if sim_type == "forward":
        destination_folder = (
            comm.project.paths["synthetics"]
            / "EARTHQUAKES"
            / iteration
            / event
        )
    elif sim_type == "adjoint":
        destination_folder = (
            comm.project.paths["gradients"] / iteration / event
        )

    if "array_name" in job_dict.keys():
        job_array = salvus.flow.api.get_job_array(
            job_array_name=job_dict["array_name"], site_name=site_name
        )
        job_names = [j.job_name for j in job_array.jobs]
        if job_name not in job_names:
            job = salvus.flow.api.get_job(
                job_name=job_dict[event], site_name=site_name
            )
            job.copy_output(
                destination=destination_folder,
                get_all=False,
                allow_existing_destination_folder=True,
            )
            return
        event_job_index = job_names.index(job_name)
        job_array.jobs[event_job_index].copy_output(
            destination=destination_folder,
            get_all=False,
            allow_existing_destination_folder=True,
        )
    else:
        job = salvus.flow.api.get_job(
            job_name=job_dict[event], site_name=site_name
        )
        job.copy_output(
            destination=destination_folder,
            get_all=False,
            allow_existing_destination_folder=True,
        )


def retrieve_salvus_simulations(
    comm: object, events: Union[List[str], str], iteration: str, sim_type: str
):
    """
    Retrieve Salvus simulations based on job names currently in job_toml file

    :param comm: The Lasif communicator object
    :type comm: object
    :param events: We need names of events for the corresponding simulations
        in order to keep tabs on which simulation object corresponds to which
        event.
    :type events: Union[List[str], str]
    :param iteration: Name of iteration, this is needed to know where to
        download files to when jobs are done.
    :type iteration: str
    :param sim_type: can be either forward or adjoint.
    :type sim_type: str
    """
    # We need to figure out the status of the jobs
    # if finished we can proceed with downloading, else we have to
    # output a message.
    # If only some are finished, we retrieve those and send an informative
    # message regarding the rest.
    status = check_job_status(
        comm=comm, events=events, iteration=iteration, sim_type=sim_type,
    )
    if not isinstance(events, list):
        events = [events]

    for event in events:
        if status[event].name == "finished":
            print(f"Retrieving simulation for event {event}")
            download_output(
                comm=comm, event=event, iteration=iteration, sim_type=sim_type,
            )
        else:
            print(
                f"Job status for event {event} is: {status[event].name}. "
                "Can not download output."
            )


def retrieve_salvus_simulations_blocking(
    comm: object,
    events: Union[List[str], str],
    iteration: str,
    sim_type: str,
    verbosity: int = 1,
):
    """
    Retrieve Salvus simulations based on job names currently in job_toml file

    :param comm: The Lasif communicator object
    :type comm: object
    :param events: We need names of events for the corresponding simulations
        in order to keep tabs on which simulation object corresponds to which
        event.
    :type events: Union[List[str], str]
    :param iteration: Name of iteration, this is needed to know where to
        download files to when jobs are done.
    :type iteration: str
    :param sim_type: can be either forward or adjoint.
    :type sim_type: str
    """
    # We need to figure out the status of the jobs
    # if finished we can proceed with downloading, else we have to
    # output a message.
    # If only some are finished, we retrieve those and send an informative
    # message regarding the rest.

    finished = False

    while not finished:
        # Get status
        status = check_job_status(
            comm=comm, events=events, iteration=iteration, sim_type=sim_type,
        )

        # Convert to list if a single entry
        if not isinstance(events, list):
            events = [events]

        status_per_event = []

        # Check all events
        for event in events:
            status_per_event.append(status[event].name)

        if verbosity > 0:
            print(status_per_event)

        # Check if all are finished
        if all(status == "finished" for status in status_per_event):
            finished = True

    # Download all once finished
    for event in events:
        download_output(
            comm=comm, event=event, iteration=iteration, sim_type=sim_type,
        )
