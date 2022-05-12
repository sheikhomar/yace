import abc
import dataclasses
import json


def make_experiment_generation_registry():
    registry = dict()
    def wrapper(func):
        registry[func.__name__] = func
        return func
    wrapper.registry = registry
    return wrapper


class Experiment:
    @abc.abstractmethod
    def run(self) -> None:
        raise NotImplementedError


class ExperimentParams:
    def write_json(self, file_path: str):
        with open(file_path, "w") as f:
            json.dump(dataclasses.asdict(self), f, indent=4, sort_keys=False)

    def __str__(self):
        return json.dumps(dataclasses.asdict(self), indent=4)
    
    def to_dict(self):
        return dataclasses.asdict(self)
