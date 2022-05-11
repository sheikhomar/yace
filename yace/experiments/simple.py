import dataclasses, os

from pathlib import Path
from typing import Dict, List, Generator

import numpy as np
import pandas as pd

from dacite import from_dict
from sklearn.metrics.pairwise import _euclidean_distances
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize
from sklearn.utils.extmath import row_norms

from yace.coresets.sensitivity_sampling import SensitivitySampling
from yace.data.synthetic import generate_simplex
from yace.experiments import Experiment, ExperimentParams, make_experiment_generation_registry
from yace.helpers.logger import get_logger


@dataclasses.dataclass
class SimpleInstanceExperimentParams(ExperimentParams):
    k: int
    epsilon: float
    n_points: int
    coreset_size: int
    algorithm_name: str
    random_seed: int


def create_experiment_param(
    k=None,
    epsilon=None,
    n_points=None,
    coreset_size=None,
    algorithm_name=None,
    random_seed=None,
    ) -> SimpleInstanceExperimentParams:

    k = 10 if k is None else k
    epsilon = 0.1 if epsilon is None else epsilon
    n_points = int(np.power(k, 1.5) / np.power(epsilon, 2)) if n_points is None else n_points
    coreset_size = int(k / (10 * np.power(epsilon, 2))) if coreset_size is None else coreset_size
    algorithm_name = "ss" if algorithm_name is None else algorithm_name
    random_seed = int.from_bytes(os.urandom(3), "big") if random_seed is None else random_seed

    return SimpleInstanceExperimentParams(
        k=k, 
        epsilon=epsilon,
        n_points=n_points,
        coreset_size=coreset_size,
        algorithm_name=algorithm_name,
        random_seed=random_seed,
    )


logger = get_logger(name="sie")

class SimpleInstanceExperiment(Experiment):
    def __init__(self, experiment_params: Dict[str, object], working_dir: str) -> None:
        super().__init__()
        self._params: SimpleInstanceExperimentParams = from_dict(
            data_class=SimpleInstanceExperimentParams, 
            data=experiment_params
        )
        self._working_dir = working_dir

    def run(self) -> None:
        logger.debug(f"Running SimpleInstanceExperiment with params: \n{self._params}")

        np.random.default_rng(self._params.random_seed)
        np.random.seed(self._params.random_seed)

        output_dir = Path(self._working_dir)
    
        logger.debug("Generating data")
        X = generate_simplex(d=self._params.n_points, alpha=1)

        logger.debug(f"The shape of the generated data matrix is: {X.shape}")

        logger.debug(f"Running {self._params.algorithm_name}...")

        if self._params.algorithm_name == "ss":
            algorithm = SensitivitySampling(
                n_clusters=self._params.k,
                coreset_size=self._params.coreset_size,
            )
            coreset_points, coreset_weights = algorithm.run(X)
            np.savez_compressed(output_dir/"coreset-points.npz", matrix=coreset_points)
            np.savez_compressed(output_dir/"coreset-weights.npz", matrix=coreset_weights)

            df_distortions = self._evaluate_random_solutions(
                input_points=X,
                coreset_points=coreset_points,
                coreset_weights=coreset_weights,
                n_repetitions=50,
            )
            df_distortions.to_feather(output_dir/"distortions.feather")

        with open(output_dir / "done.out", "w") as f:
            f.write("done")

    def _evaluate_random_solutions(self, input_points: np.ndarray, coreset_points: np.ndarray, coreset_weights: np.ndarray, n_repetitions: int) -> pd.DataFrame:
        logger.debug("Evaluating solution...")

        n_dim = input_points.shape[1]
        k = self._params.k

        results = []
        for iter in range(n_repetitions):
            logger.debug(f"Solution {iter}")
            # Generate a solution that consists of a set of k d-dimensional vectors
            solution = np.random.normal(loc=0, scale=1, size=(k, n_dim))

            # Normalize the solution
            solution = normalize(solution)

            coreset_cost = self._compute_coreset_cost(
                coreset_points=coreset_points,
                coreset_weights=coreset_weights,
                candidate_solution=solution,
            )

            input_cost = self._compute_input_cost(
                input_points=input_points,
                candidate_solution=solution,
            )

            del solution

            distortion = max(float(input_cost/coreset_cost), float(coreset_cost/input_cost))

            results.append(dict(
                coreset_cost=coreset_cost,
                input_cost=input_cost,
                distortion=distortion,
            ))

        return pd.DataFrame(results)

    def _compute_coreset_cost(self, coreset_points: np.ndarray, coreset_weights: np.ndarray, candidate_solution: np.ndarray):
        # Distances between all corset points and candidate solution points
        dist_sq = _euclidean_distances(X=coreset_points, Y=candidate_solution, squared=True)

        # For each point (w, p) in S, find the distance to its closest center
        closest_dist_sq = np.min(dist_sq, axis=1)

        coreset_cost = np.sum(coreset_weights * closest_dist_sq)
        return coreset_cost

    def _compute_input_cost(self, input_points: np.ndarray, candidate_solution: np.ndarray) -> float:
        dist_sq = _euclidean_distances(X=input_points, Y=candidate_solution, squared=True)
        closest_dist_sq = np.min(dist_sq, axis=1)
        input_cost = np.sum(closest_dist_sq)
        return input_cost
       

experiment_generation = make_experiment_generation_registry()

initialize_experiment = SimpleInstanceExperiment


@experiment_generation
def initial_ss_01() -> Generator[object, None, None]:
    for k in [10, 20, 50, 70]:
        for epsilon in [0.20, 0.10, 0.05, 0.01]:
            yield create_experiment_param(
                k=k,
                epsilon=epsilon,
            )
