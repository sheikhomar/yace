from cmath import exp
import re, subprocess, os

from pathlib import Path
from typing import List

import click
import numba
import numpy as np
import pandas as pd
import scipy.sparse as sp_sparse

from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits

from yace.helpers.logger import get_logger
from yace.helpers.evaluation import DistortionCalculator
from yace.experiments.simple import SimpleInstanceExperiment
from yace.experiments.adv import AdvInstanceExperiment
from yace.experiments.rw import RealWorldDataExperiment
from yace.run_worker import JobInfo


logger = get_logger("distort")


def calc_distortion_for_simple_instance(job_info: JobInfo):
    working_dir = job_info.working_dir
    experiment = SimpleInstanceExperiment(
        experiment_params=job_info.experiment_params,
        working_dir=working_dir,
    )

    k = experiment._params.k
    experiment.set_random_seed()
    input_points = experiment.get_data_set()

    logger.debug("Loading coreset...")
    coreset_points = sp_sparse.load_npz(working_dir/"coreset-points.npz")
    coreset_weights = np.load(working_dir/"coreset-weights.npz", allow_pickle=True)["matrix"]

    calc = DistortionCalculator(
        working_dir=working_dir,
        k=k,
        input_points=input_points,
        coreset_points=coreset_points,
        coreset_weights=coreset_weights,
    )

    calc.on_random()
    calc.on_convex()


def calc_distortion_for_adv_instance(job_info: JobInfo):
    working_dir = job_info.working_dir
    experiment = AdvInstanceExperiment(
        experiment_params=job_info.experiment_params,
        working_dir=working_dir,
    )

    k = experiment._params.k
    rng = experiment.set_random_seed()
    input_points = experiment.get_data_set()
    
    logger.debug("Loading coreset...")
    coreset_points = np.load(working_dir/"coreset-points.npz", allow_pickle=True)["matrix"]
    coreset_weights = np.load(working_dir/"coreset-weights.npz", allow_pickle=True)["matrix"]

    DistortionCalculator(
        working_dir=working_dir,
        k=k,
        input_points=input_points,
        coreset_points=coreset_points,
        coreset_weights=coreset_weights,
    ).calc_distortions_for_adv_instance(rng)


def calc_distortion_for_real_world_data_sets(job_info: JobInfo):
    working_dir = job_info.working_dir
    experiment = RealWorldDataExperiment(
        experiment_params=job_info.experiment_params,
        working_dir=working_dir,
    )

    k = experiment._params.k
    experiment.set_random_seed()
    input_points = experiment.get_data_set()
    
    logger.debug("Loading coreset...")
    coreset_points = np.load(working_dir/"coreset-points.npz", allow_pickle=True)["matrix"]
    coreset_weights = np.load(working_dir/"coreset-weights.npz", allow_pickle=True)["matrix"]
    
    calc = DistortionCalculator(
        working_dir=working_dir,
        k=k,
        input_points=input_points,
        coreset_points=coreset_points,
        coreset_weights=coreset_weights,
    )

    calc.on_kmeans_plus_plus_coreset()
    calc.on_kmeans_plus_plus_input()


def calc_distortion_for_job(index: int, n_total: int, job_info: JobInfo, n_threads: int):
    experiment_dir = job_info.working_dir
    experiment_type = job_info.command_params["experiment-type"]

    logger.debug(f"[{index+1}/{n_total}] Experiment type: {experiment_type}. ")

    with threadpool_limits(limits=n_threads):
        if experiment_type == "simple":
            calc_distortion_for_simple_instance(job_info)
        elif experiment_type == "adv":
            calc_distortion_for_adv_instance(job_info)
        elif experiment_type == "rw":
            calc_distortion_for_real_world_data_sets(job_info)
        else:
            logger.debug(f"[{index+1}/{n_total}]: Skipping! {experiment_type} is unknown - for  {experiment_dir}")
        

def find_completed_jobs(results_dir: str) -> List[JobInfo]:
    search_dir = Path(results_dir)
    output_paths = list(search_dir.glob('**/coreset-points.npz'))
    return_list = []
    for file_path in output_paths:
        run_info = JobInfo.load_json(file_path.parent / "job-info.json")
        if run_info is not None:
            return_list.append(run_info)
    return return_list


def calc_distortions(results_dir: str, n_jobs: int, n_threads: int) -> None:
    completed_jobs = find_completed_jobs(results_dir)
    total_files = len(completed_jobs)
    logger.debug(f"Found {total_files} jobs to be processed...")
    Parallel(n_jobs=n_jobs)(
        delayed(calc_distortion_for_job)(
            index=index,
            n_total=total_files,
            job_info=job_info,
            n_threads=n_threads,
        )
        for index, job_info in enumerate(completed_jobs)
    )


@click.command(help="Evaluate algorithm on benchmark dataset.")
@click.option(
    "-r",
    "--results-dir",
    type=click.STRING,
    required=True,
)
@click.option(
    "-j",
    "--n-jobs",
    type=click.INT,
    required=False,
    default=1
)
@click.option(
    "-t",
    "--n-threads",
    type=click.INT,
    required=False,
    default=1
)
def main(results_dir: str, n_jobs: int, n_threads: int) -> None:
    calc_distortions(
        results_dir=results_dir,
        n_jobs=n_jobs,
        n_threads=n_threads,
    )


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
