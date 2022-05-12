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

import yace.helpers.evaluation as yace_eval

from yace.helpers.logger import get_logger
from yace.run_worker import JobInfo
from yace.experiments.simple import SimpleInstanceExperiment


logger = get_logger("distort")


def calc_distortion_on_random_solutions(k: int, input_points: np.ndarray, coreset_points: np.ndarray, coreset_weights: np.ndarray, n_repetitions: int) -> pd.DataFrame:
    results = []
    n_dim = input_points.shape[1]
    for iter in range(n_repetitions):
        solution = yace_eval.generate_random_solution(n_dim=n_dim, k=k)

        coreset_cost = yace_eval.compute_coreset_cost(
            coreset_points=coreset_points,
            coreset_weights=coreset_weights,
            candidate_solution=solution,
        )

        input_cost = yace_eval.compute_input_cost(
            input_points=input_points,
            candidate_solution=solution,
        )

        solution = None
        del solution

        distortion = max(float(input_cost/coreset_cost), float(coreset_cost/input_cost))

        results.append(dict(
            iteration=iter,
            solution_type="random",
            coreset_cost=coreset_cost,
            input_cost=input_cost,
            distortion=distortion,
        ))
    return pd.DataFrame(results)


def calc_distortion_for_simple_instance(job_info: JobInfo):
    working_dir = job_info.working_dir
    experiment = SimpleInstanceExperiment(
        experiment_params=job_info.experiment_params,
        working_dir=working_dir,
    )

    experiment.set_random_seed()
    input_points = experiment.get_data_set()
    
    logger.debug("Loading coreset...")
    coreset_points = sp_sparse.load_npz(working_dir/"coreset-points.npz")
    coreset_weights = np.load(working_dir/"coreset-weights.npz", allow_pickle=True)["matrix"]

    dist_random_path = working_dir/"distortions-random-solutions.feather"
    if not dist_random_path.exists():
        logger.debug("Calculating distortions on solutions generated uniformly at random")
        df_dist_random = calc_distortion_on_random_solutions(
            k=experiment._params.k,
            input_points=input_points,
            coreset_points=coreset_points,
            coreset_weights=coreset_weights,
            n_repetitions=50,
        )

        logger.debug("Storing distortions")
        df_dist_random.to_feather(dist_random_path)


def calc_distortion_for_job(index: int, n_total: int, job_info: JobInfo, n_threads: int):
    experiment_dir = job_info.working_dir
    experiment_type = job_info.command_params["experiment-type"]

    logger.debug(f"[{index+1}/{n_total}] Experiment type: {experiment_type}. ")

    with threadpool_limits(limits=n_threads):
        if experiment_type == "simple":
            calc_distortion_for_simple_instance(job_info)
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
