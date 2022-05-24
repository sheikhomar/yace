from typing import Dict

import numpy as np


class Coreset:
    def __init__(self, input_points: np.ndarray, cluster_labels: np.ndarray) -> None:
        self._point_map = Dict[int, float] = dict()
        self._cluster_map = Dict[int, float] = dict()
        self._input_points = input_points
        self._cluster_labels = cluster_labels

    def add_point(self, point_index: int, weight: float) -> None:
        if weight <= 0:
            raise ValueError("Weight cannot be negative or zero.")

        if point_index in self._point_map:
            self._point_map[point_index] += weight
        else:
            self._point_map[point_index]  = weight

    def add_cluster(self, cluster_index: int, weight: float) -> None:
        if weight <= 0:
            raise ValueError("Weight cannot be negative or zero.")

        if cluster_index in self._cluster_map:
            self._cluster_map[cluster_index] += weight
        else:
            self._cluster_map[cluster_index]  = weight
