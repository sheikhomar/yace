import dataclasses
from math import floor
from tokenize import group
from typing import Dict
import numpy as np
import scipy.sparse as sp_sparse

from scipy.sparse import issparse

from yace.coresets.sampling import SamplingBasedAlgorithm


@dataclasses.dataclass
class RinglessPoint:
    """Represents a point that is not captured by any ring. """
    point_index: int
    cluster_index: int
    point_cost: float
    cost_boundary: float
    is_overshot: bool


class ClusteredPoint:
    """Represents a reference to a point which has been assigned a cluster."""

    # The index of the point in the dataset.
    point_index: int

    # The index of the cluster for which this point is assigned.
    cluster_index: int

    # The cost of this point in its assigned cluster. 
    # It is the squared distance to the assigned cluster's center.
    cost: float


class Group:
    """Represents a group which is uniquely identified by its range value (j) and its ring range value (l) i.e., G_{j,l}."""

    def __init__(self, range_value: int, ring_range_value: int, lower_bound_cost: float, upper_bound_cost: float) -> None:
        self._range_value = range_value
        self._ring_range_value = ring_range_value
        self._lower_bound_cost = lower_bound_cost
        self._upper_bound_cost = upper_bound_cost
        self._clustered_points: Dict[int, ClusteredPoint] = dict()
        self._total_cost = 0.0

    def add_point(self, clustered_point: ClusteredPoint) -> None:
        key = clustered_point.cluster_index
        if not key in self._clustered_points:
            self._clustered_points[key] = []
        self._clustered_points[key].append(clustered_point)
        self._total_cost += clustered_point.cost

    @property
    def total_cost(self) -> float:
        return self._total_cost

    def count_points_in_cluster(self, cluster_index: int) -> int:
        """Counts the number of points in the given cluster."""
        return len(self._clustered_points[cluster_index])


class Ring:
    def __init__(self, cluster_index: int, range_value: int, average_cluster_cost: float) -> None:
        # The cluster for which this ring belongs to.
        self._cluster_index = cluster_index

        # The range value of this ring i.e., `l`.
        self._range_value = range_value
        
        # The average cost for the cluster associated with this ring.
        self._average_cluster_cost = average_cluster_cost

        #  Ring upper bound cost := Δ_c * 2^l
        self._lower_bound_cost = average_cluster_cost * np.power(2, range_value)

        # Ring upper bound cost := Δ_c * 2^(l+1)
        self._upper_bound_cost = average_cluster_cost * np.power(2, range_value + 1)

        self._total_cost = 0.0

    def try_add_point(self, point_index: int, point_cost: float) -> bool:
        # If cost(p, A) is between Δ_c*2^l and Δ_c*2^(l+1) ...
        if point_cost >= self._lower_bound_cost and point_cost < self._upper_bound_cost:
            pass

        return False

class RingSet:
    def __init__(self, range_start: int, range_end: int, n_clusters: int) -> None:
        self._range_start = range_start
        self._range_end = range_end
        self._n_clusters = n_clusters


class GroupSamplingSampling(SamplingBasedAlgorithm):
    def __init__(self, n_clusters: int, coreset_size: int, beta: int, group_range_size: int, min_group_sampling_size: int) -> None:
        super().__init__(n_clusters, coreset_size)
        self._beta = beta
        self._group_range_size = group_range_size
        self._min_group_sampling_size = min_group_sampling_size
    
    def run(self, A):
        D = self._compute_initial_solution_via_kmeans_plus_plus(A=A)
        
        cluster_labels = np.argmin(D, axis=1)
        point_costs = np.min(D, axis=1)

        self._make_rings(point_costs=point_costs)

    def _make_rings(self, point_costs: np.ndarray):
        ring_range_start = -int(np.floor(np.log10(self._beta)))
        ring_range_end = -ring_range_start

        n_points = point_costs.shape[0]
        pass