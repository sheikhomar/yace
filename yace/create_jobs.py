import os, uuid

from datetime import datetime
from importlib import import_module
from pathlib import Path

import click
import numpy as np

from yace.run_worker import JobInfo


def generate_id() -> str:
    return str(uuid.uuid4().hex)


class JobCreator:
    def __init__(
        self, queue_dir: str,
        experiment_type: str,
        experiment_name: str,
        n_repetitions: str,
        output_dir: str
    ) -> None:
        self._queue_dir = Path(queue_dir)
        self._experiment_type = experiment_type
        self._experiment_name = experiment_name
        self._n_repetitions = n_repetitions
        self._output_dir = Path(output_dir)

    def run(self) -> None:
        ready_dir = self._queue_dir / "ready"
        os.makedirs(ready_dir, exist_ok=True)

        generate_experiments = self._load_experiment_generator()
        gen_id = generate_id()

        for it in range(self._n_repetitions):
            for exp_params in generate_experiments():
                print(f"[{it+1:03d} / {self._n_repetitions:03d}] Creating a job with experiment params: {exp_params}")
                experiment_no = f"g{gen_id}-r{it}-x{generate_id()}"
                working_dir = self._output_dir / self._experiment_type / self._experiment_name / experiment_no
                os.makedirs(working_dir, exist_ok=True)

                params_path = working_dir / "experiment-params.json"
                exp_params.write_json(params_path)

                job_info = JobInfo()
                job_info.experiment_params = exp_params.to_dict()
                job_info.working_dir = working_dir
                job_info.command = f"poetry run python -m yace.run_experiment"
                job_info.command_params = {
                    "experiment-type": self._experiment_type,
                    "params-path": str(params_path),
                    "working-dir": str(working_dir),
                }

                job_name = f"{self._experiment_type}.{self._experiment_name}.{experiment_no}.json"
                job_info.write_json(ready_dir / job_name)

    def _load_experiment_generator(self):
        try:
            module_name = f"yace.experiments.{self._experiment_type}"
            experiment_module = import_module(module_name)
        except ModuleNotFoundError:
            raise ValueError(f"Experiment module '{module_name}' was not found.")
        registry = experiment_module.experiment_generation.registry # type: ignore
        print(registry)
        if self._experiment_name not in registry:
            raise ValueError(f"Experiment with the name '{self._experiment_name}' was not found in module '{module_name}'.")
        return registry[self._experiment_name] # type: ignore


@click.command(help="Create jobs.")
@click.option(
    "-q",
    "--queue-dir",
    type=click.STRING,
    required=True,
)
@click.option(
    "-t",
    "--experiment-type",
    type=click.STRING,
    required=True,
)
@click.option(
    "-n",
    "--experiment-name",
    type=click.STRING,
    required=True,
)
@click.option(
    "-r",
    "--n-repetitions",
    type=click.INT,
    default=1,
    required=False,
)
@click.option(
    "-o",
    "--output-dir",
    type=click.STRING,
    required=True,
)
def main(queue_dir: str, experiment_type: str, experiment_name: str, n_repetitions: int, output_dir: str):
    JobCreator(
        queue_dir=queue_dir,
        experiment_type=experiment_type,
        experiment_name=experiment_name,
        n_repetitions=n_repetitions,
        output_dir=output_dir,
    ).run()


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
