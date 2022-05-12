import dataclasses, os

from pathlib import Path
from typing import Dict, List, Generator

import numpy as np
import pandas as pd
import scipy.sparse as sp_sparse

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
        self._working_dir = Path(working_dir)

    def run(self) -> None:
        logger.debug(f"Running SimpleInstanceExperiment with params: \n{self._params}")

        self.set_random_seed()
        X = self.get_data_set()

        if self._params.algorithm_name == "ss":
            logger.debug(f"Running Sensitivity Sampling to generate coreset...")
            algorithm = SensitivitySampling(
                n_clusters=self._params.k,
                coreset_size=self._params.coreset_size,
            )
        else:
            raise ValueError(f"Unknown algorithm: {self._params.algorithm_name}")

        coreset_points, coreset_weights = algorithm.run(X)
        sp_sparse.save_npz(self._working_dir/"coreset-points.npz", matrix=coreset_points, compressed=True)
        np.savez_compressed(self._working_dir/"coreset-weights.npz", matrix=coreset_weights)

        with open(self._working_dir / "done.out", "w") as f:
            f.write("done")
        logger.debug("Done")

    def set_random_seed(self):
        np.random.default_rng(self._params.random_seed)
        np.random.seed(self._params.random_seed)

    def get_data_set(self):
        logger.debug("Generating data")
        X = generate_simplex(d=self._params.n_points, alpha=1)
        logger.debug(f"The shape of the generated data matrix is: {X.shape}")
        return X


experiment_generation = make_experiment_generation_registry()

initialize_experiment = SimpleInstanceExperiment


@experiment_generation
def initial_ss_01() -> Generator[object, None, None]:
    for k in [10, 20, 50, 70, 100]:
        for epsilon in [0.20, 0.10, 0.05, 0.01]:
            yield create_experiment_param(
                k=k,
                epsilon=epsilon,
            )
