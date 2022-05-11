import dataclasses, os
import json

from pathlib import Path
from textwrap import wrap
from typing import Dict, List, Generator

import numpy as np
import pandas as pd

from dacite import from_dict
from yace.coresets.sensitivity_sampling import SensitivitySampling

from yace.experiments import Experiment, ExperimentParams, make_experiment_generation_registry
from yace.helpers.logger import get_logger
from yace.data.synthetic import generate_simplex


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


class SimpleInstanceExperiment(Experiment):
    def __init__(self, experiment_params: Dict[str, object], working_dir: str) -> None:
        super().__init__()
        self._params: SimpleInstanceExperimentParams = from_dict(
            data_class=SimpleInstanceExperimentParams, 
            data=experiment_params
        )
        self._working_dir = working_dir

    def run(self) -> None:
        logger = get_logger(name="sie")

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
            coreset = algorithm.run(X)
            output_file_path = output_dir / "coreset.npz"
            np.savez_compressed(output_file_path, matrix=coreset)

        with open(output_dir / "done.out", "w") as f:
            f.write("done")


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
