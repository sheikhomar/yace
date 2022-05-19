import dataclasses, os

from pathlib import Path
from typing import Dict, List, Generator

import numpy as np
import pandas as pd
import scipy.sparse as sp_sparse

from dacite import from_dict

from yace.coresets.sensitivity_sampling import SensitivitySampling
from yace.coresets.sensitivity_sampling_ex import SensitivitySamplingExtended
from yace.coresets.uniform_sampling import UniformSampling
from yace.data.synthetic import generate_adv_instance
from yace.experiments import Experiment, ExperimentParams, make_experiment_generation_registry
from yace.helpers.logger import get_logger


@dataclasses.dataclass
class AdvInstanceExperimentParams(ExperimentParams):
    k: int
    epsilon: float
    alpha: float
    simplex_size: int
    n_points_per_simplex: int
    simplex_points_variance: float
    coreset_size: int
    algorithm_name: str
    random_seed: int


def create_experiment_param(
    k=None,
    epsilon=None,
    alpha=None,
    simplex_size=None,
    n_points_per_simplex=None,
    simplex_points_variance=None,
    coreset_size=None,
    algorithm_name=None,
    random_seed=None,
    ) -> AdvInstanceExperimentParams:

    k = 10 if k is None else k
    epsilon = 0.1 if epsilon is None else epsilon
    alpha = np.sqrt( np.power(epsilon, -1) ) if alpha is None else alpha
    simplex_size = int(k / 2) if simplex_size is None else simplex_size
    n_points_per_simplex = int(2 * np.sqrt(k) / np.power(epsilon, 2)) if n_points_per_simplex is None else n_points_per_simplex
    simplex_points_variance = 1 if simplex_points_variance is None else simplex_points_variance
    coreset_size = int(k / (10 * np.power(epsilon, 2))) if coreset_size is None else coreset_size
    algorithm_name = "ss" if algorithm_name is None else algorithm_name
    random_seed = int.from_bytes(os.urandom(3), "big") if random_seed is None else random_seed

    # Sanity check: we want the total number of generated
    # points to be k^1.5 / epsilon^2
    n_points_target = int(np.power(k, 1.5) / np.power(epsilon, 2))
    n_gen_points = simplex_size * n_points_per_simplex
    n_points_off = int(np.abs(n_points_target - n_gen_points))
    assert n_points_off < 100, f"Number of generated points is off by {n_points_off}. Should be less than 100."

    return AdvInstanceExperimentParams(
        k=k, 
        epsilon=epsilon,
        alpha=alpha,
        simplex_size=simplex_size,
        n_points_per_simplex=n_points_per_simplex,
        simplex_points_variance=simplex_points_variance,
        coreset_size=coreset_size,
        algorithm_name=algorithm_name,
        random_seed=random_seed,
    )


logger = get_logger(name="sie")

class AdvInstanceExperiment(Experiment):
    def __init__(self, experiment_params: Dict[str, object], working_dir: str) -> None:
        super().__init__()
        self._params: AdvInstanceExperimentParams = from_dict(
            data_class=AdvInstanceExperimentParams, 
            data=experiment_params
        )
        self._working_dir = Path(working_dir)

    def run(self) -> None:
        logger.debug(f"Running AdvInstanceExperiment with params: \n{self._params}")

        self.set_random_seed()
        X = self.get_data_set()

        if self._params.algorithm_name in ["ss", "sensitivity", "sensitivity-sampling"]:
            logger.debug(f"Running Sensitivity Sampling to generate coreset...")
            algorithm = SensitivitySampling(
                n_clusters=2*self._params.k,
                coreset_size=self._params.coreset_size,
            )
        elif self._params.algorithm_name in ["ssx", "sensitivity-sampling-ex"]:
            logger.debug(f"Running Sensitivity Sampling Extended to generate coreset...")
            algorithm = SensitivitySamplingExtended(
                n_clusters=2*self._params.k,
                coreset_size=self._params.coreset_size,
            )
        elif self._params.algorithm_name in ["us", "uniform", "uniform-sampling"]:
            logger.debug(f"Running Uniform Sampling to generate coreset...")
            algorithm = UniformSampling(
                n_clusters=2*self._params.k,
                coreset_size=self._params.coreset_size,
            )
        else:
            raise ValueError(f"Unknown algorithm: {self._params.algorithm_name}")

        coreset_points, coreset_weights = algorithm.run(X)
        np.savez_compressed(self._working_dir/"coreset-points.npz", matrix=coreset_points)
        np.savez_compressed(self._working_dir/"coreset-weights.npz", matrix=coreset_weights)

        with open(self._working_dir / "done.out", "w") as f:
            f.write("done")
        logger.debug("Done")

    def set_random_seed(self):
        np.random.default_rng(self._params.random_seed)
        np.random.seed(self._params.random_seed)

    def get_data_set(self):
        logger.debug("Generating data")
        X = generate_adv_instance(
            simplex_size=self._params.simplex_size,
            alpha=self._params.alpha,
            n_points_per_simplex=self._params.n_points_per_simplex,
            simplex_points_variance=self._params.simplex_points_variance,
        )
        logger.debug(f"The shape of the generated data matrix is: {X.shape}")
        return X


experiment_generation = make_experiment_generation_registry()

initialize_experiment = AdvInstanceExperiment


@experiment_generation
def ss_us_01() -> Generator[object, None, None]:
    for algo in ["sensitivity-sampling", "uniform-sampling"]:
        for k in [10, 20, 50, 70, 100]:
            for epsilon in [0.20, 0.10, 0.05, 0.01]:
                yield create_experiment_param(
                    k=k,
                    epsilon=epsilon,
                    algorithm_name=algo
                )


@experiment_generation
def ss_us_02() -> Generator[object, None, None]:
    for algo in ["sensitivity-sampling", "uniform-sampling"]:
        for k in [10, 20, 50, 70, 100]:
            for epsilon in [0.20, 0.10, 0.05, 0.01]:
                yield create_experiment_param(
                    k=k,
                    epsilon=epsilon,
                    algorithm_name=algo,
                    coreset_size=int(np.power(k, 1.25) / (10 * np.power(epsilon, 2)))
                )


@experiment_generation
def ss_us_03() -> Generator[object, None, None]:
    for algo in ["sensitivity-sampling", "uniform-sampling"]:
        for k in [10, 20, 50, 70, 100]:
            for epsilon in [0.20, 0.10, 0.05, 0.01]:
                yield create_experiment_param(
                    k=k,
                    epsilon=epsilon,
                    algorithm_name=algo,
                    coreset_size=int(np.power(k, 1.5) / (10 * np.power(epsilon, 2)))
                )


@experiment_generation
def ssx_us_01() -> Generator[object, None, None]:
    for algo in ["sensitivity-sampling", "sensitivity-sampling-ex", "uniform-sampling"]:
        for k in [10, 20, 50, 70, 100]:
            for epsilon in [0.20, 0.10, 0.05, 0.01]:
                yield create_experiment_param(
                    k=k,
                    epsilon=epsilon,
                    algorithm_name=algo,
                    coreset_size=int(np.power(k, 1.0) / (10 * np.power(epsilon, 2))),
                )


@experiment_generation
def ss_us_alpha_01() -> Generator[object, None, None]:
    for algo in ["sensitivity-sampling", "uniform-sampling"]:
        for k in [10, 20, 50, 70, 100]:
            for epsilon in [0.20, 0.10, 0.05, 0.01]:
                yield create_experiment_param(
                    k=k,
                    epsilon=epsilon,
                    algorithm_name=algo,
                    coreset_size=int(np.power(k, 1.0) / (10 * np.power(epsilon, 2))),
                    alpha=np.power(epsilon, -1),
                )


@experiment_generation
def ss_us_alpha_02() -> Generator[object, None, None]:
    for algo in ["sensitivity-sampling", "uniform-sampling", "sensitivity-sampling-ex"]:
        for k in [10, 20, 50, 70, 100]:
            for epsilon in [0.20, 0.10, 0.05, 0.01]:
                yield create_experiment_param(
                    k=k,
                    epsilon=epsilon,
                    algorithm_name=algo,
                    coreset_size=int(np.power(k, 1.0) / (10 * np.power(epsilon, 2))),
                    alpha=2*np.power(epsilon, -1),
                )


@experiment_generation
def full_01() -> Generator[object, None, None]:
    for algo in ["sensitivity-sampling", "sensitivity-sampling-ex", "uniform-sampling"]:
        for k in [10, 20, 50, 70, 100]:
            for epsilon in [0.20, 0.10, 0.05, 0.01]:
                yield create_experiment_param(
                    k=k,
                    epsilon=epsilon,
                    algorithm_name=algo,
                    coreset_size=int(np.power(k, 1.0) / (10 * np.power(epsilon, 2))),
                )


@experiment_generation
def full_02() -> Generator[object, None, None]:
    for algo in ["sensitivity-sampling", "sensitivity-sampling-ex", "uniform-sampling"]:
        for k in [10, 20, 50, 70, 100]:
            for epsilon in [0.20, 0.10, 0.05, 0.01]:
                yield create_experiment_param(
                    k=k,
                    epsilon=epsilon,
                    algorithm_name=algo,
                    coreset_size=int(np.power(k, 1.0) / (10 * np.power(epsilon, 2))),
                )


@experiment_generation
def full_03() -> Generator[object, None, None]:
    for algo in ["sensitivity-sampling", "sensitivity-sampling-ex", "uniform-sampling"]:
        for k in [10, 20, 50, 70, 100]:
            for epsilon in [0.20, 0.10, 0.05, 0.01]:
                yield create_experiment_param(
                    k=k,
                    epsilon=epsilon,
                    algorithm_name=algo,
                    coreset_size=int(np.power(k, 1.0) / (10 * np.power(epsilon, 2))),
                )
