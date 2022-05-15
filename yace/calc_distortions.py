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
from yace.clustering.kmeans import kmeans_plusplus_with_weights

import yace.helpers.evaluation as yace_eval

from yace.helpers.logger import get_logger
from yace.run_worker import JobInfo
from yace.experiments.simple import SimpleInstanceExperiment
from yace.experiments.adv import AdvInstanceExperiment
from yace.experiments.rw import RealWorldDataExperiment


logger = get_logger("distort")


class DistortionCalculator:
    def __init__(self,
        working_dir: Path,
        k: int,
        input_points: np.ndarray,
        coreset_points: np.ndarray,
        coreset_weights: np.ndarray,
        ) -> None:
        self.working_dir = working_dir
        self.input_points = input_points
        self.coreset_points = coreset_points
        self.coreset_weights = coreset_weights
        self.k = k
    
    def calc_distortions(self, solution_generator, solution_type: str, n_repetitions: int) -> pd.DataFrame:
        results = []
        for iter in range(n_repetitions):
            solution = solution_generator()

            coreset_cost = yace_eval.compute_coreset_cost(
                coreset_points=self.coreset_points,
                coreset_weights=self.coreset_weights,
                candidate_solution=solution,
            )

            input_cost = yace_eval.compute_input_cost(
                input_points=self.input_points,
                candidate_solution=solution,
            )

            solution = None
            del solution

            distortion = max(float(input_cost/coreset_cost), float(coreset_cost/input_cost))

            results.append(dict(
                iteration=iter,
                solution_type=solution_type,
                coreset_cost=coreset_cost,
                input_cost=input_cost,
                distortion=distortion,
            ))
        return pd.DataFrame(results)

    def on_random(self):
        dist_random_path = self.working_dir/"distortions-random-solutions.feather"
        if not dist_random_path.exists():
            logger.debug("Calculating distortions on solutions generated uniformly at random")
            n_dim = self.input_points.shape[1]
            random_solution_generator = lambda: yace_eval.generate_random_solution(n_dim=n_dim, k=self.k)

            df_dist_random = self.calc_distortions(
                solution_generator=random_solution_generator,
                solution_type="random",
                n_repetitions=50,
            )

            logger.debug("Storing distortions")
            df_dist_random.to_feather(dist_random_path)
            df_dist_random.to_csv(str(dist_random_path).replace(".feather", ".csv"))

    def on_convex(self):
        dist_convex_path = self.working_dir/"distortions-convex-solutions.feather"
        if not dist_convex_path.exists():
            logger.debug("Calculating distortions on solutions by convex combinations of random points")

            convex_solution_generator = lambda: yace_eval.generate_random_points_within_convex_hull(
                data_matrix=self.input_points, k=self.k, n_samples=2,
            )
            df_dist_convex = self.calc_distortions(
                solution_generator=convex_solution_generator,
                solution_type="convex",
                n_repetitions=50,
            )

            logger.debug("Storing distortions")
            df_dist_convex.to_feather(dist_convex_path)
            df_dist_convex.to_csv(str(dist_convex_path).replace(".feather", ".csv"))

    def on_adv(self, ratio: float):
        sol_type = f"adv{ratio:0.2f}".replace(".", "_")
        output_path = self.working_dir/f"distortions-{sol_type}-solutions.feather"
        if not output_path.exists():
            sample_size = int(np.power(self.k, ratio))
            solution_generator = lambda: yace_eval.generate_candidate_solution_for_adv_instance(
                data_matrix=self.input_points, k=self.k, sample_size=sample_size,
            )
            df_distortions = self.calc_distortions(
                solution_generator=solution_generator,
                solution_type=sol_type,
                n_repetitions=50,
            )
            logger.debug("Storing distortions")
            df_distortions.to_feather(output_path)
            df_distortions.to_csv(str(output_path).replace(".feather", ".csv"))

    def on_kmeans_plus_plus(self):
        solution_type = "kmeans++-on-input"
        output_path = self.working_dir/f"distortions-{solution_type}-solutions.feather"
        if not output_path.exists():
            solution_generator = lambda: yace_eval.generate_candidate_solution_via_kmeans_plus_plus(
                data_matrix=self.input_points, k=self.k,
            )

            distortions_list = []
            for i in range(2):
                df_local_distortions = self.calc_distortions(
                    solution_generator=solution_generator,
                    solution_type=solution_type,
                    n_repetitions=5,
                )
                max_distortion_idx = df_local_distortions['distortion'].idxmax()
                max_distortion_row = df_local_distortions.iloc[max_distortion_idx].copy()
                distortions_list.append(max_distortion_row.to_dict())

            df_distortions = pd.DataFrame(distortions_list)

            # Reset iteration column
            df_distortions["iteration"] = np.arange(len(distortions_list))
            
            logger.debug(f"Storing distortions to {output_path}")
            df_distortions.to_feather(output_path)
            df_distortions.to_csv(str(output_path).replace(".feather", ".csv"))

    def on_kmeans_plus_plus_coreset(self):
        solution_type = "kmeans++-on-coreset"
        output_path = self.working_dir/f"distortions-{solution_type}-solutions.feather"
        if not output_path.exists():
            def solution_generator():
                center_indices = kmeans_plusplus_with_weights(
                    points=self.coreset_points, 
                    weights=self.coreset_weights,
                    n_clusters=self.k,
                )
                solution = self.input_points[center_indices]
                return solution

            distortions_list = []
            for i in range(2):
                df_local_distortions = self.calc_distortions(
                    solution_generator=solution_generator,
                    solution_type=solution_type,
                    n_repetitions=5,
                )
                max_distortion_idx = df_local_distortions['distortion'].idxmax()
                max_distortion_row = df_local_distortions.iloc[max_distortion_idx].copy()
                distortions_list.append(max_distortion_row.to_dict())

            df_distortions = pd.DataFrame(distortions_list)

            # Reset iteration column
            df_distortions["iteration"] = np.arange(len(distortions_list))
            
            logger.debug(f"Storing distortions to {output_path}")
            df_distortions.to_feather(output_path)
            df_distortions.to_csv(str(output_path).replace(".feather", ".csv"))


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

    calc.on_adv(ratio=1/3)
    calc.on_adv(ratio=1/2)
    calc.on_adv(ratio=2/3)


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
    calc.on_kmeans_plus_plus()
    calc.on_kmeans_plus_plus_coreset()


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
