import dataclasses, os

from pathlib import Path
from typing import Dict, List, Generator

import numpy as np

from dacite import from_dict

from yace.coresets.sensitivity_sampling import SensitivitySampling
from yace.coresets.uniform_sampling import UniformSampling
from yace.experiments import Experiment, ExperimentParams, make_experiment_generation_registry
from yace.helpers.logger import get_logger


logger = get_logger(name="rw")
experiment_generation = make_experiment_generation_registry()


@dataclasses.dataclass
class RealWorldDataExperimentParams(ExperimentParams):
    data_set: str
    k: int
    coreset_size: int
    algorithm_name: str
    random_seed: int


def create_experiment_param(
    data_set=None,
    k=None,
    coreset_size=None,
    algorithm_name=None,
    random_seed=None,
    ) -> RealWorldDataExperimentParams:

    data_set = "tower" if data_set is None else data_set
    k = 10 if k is None else k
    coreset_size = 200 * k
    algorithm_name = "ss" if algorithm_name is None else algorithm_name
    random_seed = int.from_bytes(os.urandom(3), "big") if random_seed is None else random_seed

    return RealWorldDataExperimentParams(
        data_set=data_set,
        k=k, 
        coreset_size=coreset_size,
        algorithm_name=algorithm_name,
        random_seed=random_seed,
    )


class RealWorldDataExperiment(Experiment):
    def __init__(self, experiment_params: Dict[str, object], working_dir: str) -> None:
        super().__init__()
        self._params: RealWorldDataExperimentParams = from_dict(
            data_class=RealWorldDataExperimentParams, 
            data=experiment_params
        )
        self._working_dir = Path(working_dir)

    def run(self) -> None:
        logger.debug(f"Running RealWorldDataExperiment with params: \n{self._params}")

        self.set_random_seed()
        X = self.get_data_set()

        if self._params.algorithm_name in ["ss", "sensitivity", "sensitivity-sampling"]:
            logger.debug(f"Running Sensitivity Sampling to generate coreset...")
            algorithm = SensitivitySampling(
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
        data_file_path = f"data/input/{self._params.data_set}.npz"
        if not os.path.exists(data_file_path):
            raise ValueError(f"Data file not found: {data_file_path}. Did you run `python -m data.prepare_real_world`?")
        logger.debug(f"Loading data from {data_file_path}")
        X = np.load(data_file_path)["data"]
        logger.debug(f"The shape of the generated data matrix is: {X.shape}")
        return X


initialize_experiment = RealWorldDataExperiment


@experiment_generation
def census_01() -> Generator[object, None, None]:
    for algo in ["sensitivity-sampling", "uniform-sampling"]:
        for k in [10, 20, 30, 40, 50]:
            yield create_experiment_param(
                data_set="census",
                k=k,
                coreset_size=200*k,
                algorithm_name=algo,
            )



@experiment_generation
def covertype_01() -> Generator[object, None, None]:
    for algo in ["sensitivity-sampling", "uniform-sampling"]:
        for k in [10, 20, 30, 40, 50]:
            yield create_experiment_param(
                data_set="census",
                k=k,
                coreset_size=200*k,
                algorithm_name=algo,
            )


@experiment_generation
def tower_01() -> Generator[object, None, None]:
    for algo in ["sensitivity-sampling", "uniform-sampling"]:
        for k in [20, 40, 60, 80, 100]:
            yield create_experiment_param(
                data_set="tower",
                k=k,
                coreset_size=200*k,
                algorithm_name=algo,
            )

