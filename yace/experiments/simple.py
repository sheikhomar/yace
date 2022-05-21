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
from yace.data.synthetic import generate_simplex
from yace.experiments import Experiment, ExperimentParams, make_experiment_generation_registry
from yace.helpers.logger import get_logger
from yace.helpers.evaluation import DistortionCalculator


@dataclasses.dataclass
class SimpleInstanceExperimentParams(ExperimentParams):
    k: int
    epsilon: float
    n_points: int
    coreset_size: int
    algorithm_name: str
    adjust_weights: bool
    random_seed: int


def create_experiment_param(
    k=None,
    epsilon=None,
    n_points=None,
    coreset_size=None,
    algorithm_name=None,
    adjust_weights=None,
    random_seed=None,
    ) -> SimpleInstanceExperimentParams:

    k = 10 if k is None else k
    epsilon = 0.1 if epsilon is None else epsilon
    n_points = int(np.power(k, 1.5) / np.power(epsilon, 2)) if n_points is None else n_points
    coreset_size = int(k / (10 * np.power(epsilon, 2))) if coreset_size is None else coreset_size
    algorithm_name = "ss" if algorithm_name is None else algorithm_name
    adjust_weights = True if adjust_weights is None else adjust_weights
    random_seed = int.from_bytes(os.urandom(3), "big") if random_seed is None else random_seed

    return SimpleInstanceExperimentParams(
        k=k, 
        epsilon=epsilon,
        n_points=n_points,
        coreset_size=coreset_size,
        algorithm_name=algorithm_name,
        adjust_weights=adjust_weights,
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
        self._working_dir = Path(working_dir)

    def run(self) -> None:
        logger.debug(f"Running SimpleInstanceExperiment with params: \n{self._params}")

        self.set_random_seed()
        X = self.get_data_set()

        if self._params.algorithm_name in ["ss", "sensitivity", "sensitivity-sampling"]:
            logger.debug(f"Running Sensitivity Sampling to generate coreset...")
            algorithm = SensitivitySampling(
                n_clusters=2*self._params.k,
                coreset_size=self._params.coreset_size,
                adjust_weights=self._params.adjust_weights,
            )
        elif self._params.algorithm_name in ["ssx", "sensitivity-sampling-ex"]:
            logger.debug(f"Running Sensitivity Sampling Extended to generate coreset...")
            algorithm = SensitivitySamplingExtended(
                n_clusters=2*self._params.k,
                coreset_size=self._params.coreset_size,
                adjust_weights=self._params.adjust_weights,
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
        sp_sparse.save_npz(self._working_dir/"coreset-points.npz", matrix=coreset_points, compressed=True)
        np.savez_compressed(self._working_dir/"coreset-weights.npz", matrix=coreset_weights)

        with open(self._working_dir / "coreset-done.out", "w") as f:
            f.write("done")

        logger.debug("Coreset constructed. Computing distortion...")
        DistortionCalculator(
            working_dir=self._working_dir,
            k=self._params.k,
            input_points=X,
            coreset_points=coreset_points,
            coreset_weights=coreset_weights,
        ).calc_distortions_for_simple_instance()

        with open(self._working_dir / "done.out", "w") as f:
            f.write("done")
        logger.debug("Done")

    def set_random_seed(self) -> np.random.Generator:
        np.random.seed(self._params.random_seed)
        return np.random.default_rng(self._params.random_seed)

    def get_data_set(self):
        logger.debug("Generating data")
        X = generate_simplex(d=self._params.n_points, alpha=1)
        logger.debug(f"The shape of the generated data matrix is: {X.shape}")
        return X


experiment_generation = make_experiment_generation_registry()

initialize_experiment = SimpleInstanceExperiment


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
                    coreset_size=int(k / (10 * np.power(epsilon, 2)))
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
                    coreset_size=int(k / (10 * np.power(epsilon, 2)))
                )


@experiment_generation
def full_adjusted_01() -> Generator[object, None, None]:
    for algo in ["sensitivity-sampling", "sensitivity-sampling-ex", "uniform-sampling"]:
        for k in [10, 20, 50, 70, 100]:
            for epsilon in [0.20, 0.10, 0.05, 0.01]:
                yield create_experiment_param(
                    k=k,
                    epsilon=epsilon,
                    algorithm_name=algo,
                    adjust_weights=True,
                    coreset_size=int(k / (10 * np.power(epsilon, 2)))
                )


@experiment_generation
def full_unadjusted_01() -> Generator[object, None, None]:
    for algo in ["sensitivity-sampling", "sensitivity-sampling-ex", "uniform-sampling"]:
        for k in [10, 20, 50, 70, 100]:
            for epsilon in [0.20, 0.10, 0.05, 0.01]:
                yield create_experiment_param(
                    k=k,
                    epsilon=epsilon,
                    algorithm_name=algo,
                    adjust_weights=False,
                    coreset_size=int(k / (10 * np.power(epsilon, 2)))
                )
