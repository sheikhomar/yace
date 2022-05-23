import dataclasses
from math import floor
from tokenize import group
from typing import Dict, List, Tuple
import numpy as np
import scipy.sparse as sp_sparse

from scipy.sparse import issparse
from sklearn import cluster

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


class GroupSet:
    def __init__(self, group_range_size: int) -> None:
        self._group_range_size = group_range_size
        self._groups: List[Group] = []

    def add(self, group: Group) -> None:
        self._groups.append(group)
    
    @property
    def size(self) -> int:
        return len(self._groups)

    def compute_normalized_costs(self) -> List[float]:
        costs = np.array([g.total_cost for g in self._groups])
        return costs / costs.sum()


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

        # The points assigned to this ring.
        self._points: List[ClusteredPoint] = []

        self._total_cost = 0.0

    def try_add_point(self, point_index: int, point_cost: float) -> bool:
        """Adds a point to the ring if its costs is within the bounds of this ring."""

        # Determine if point cost is within the bounds of this ring.
        # cost(p, A) is between Δ_c*2^l and Δ_c*2^(l+1) 
        if point_cost >= self._lower_bound_cost and point_cost < self._upper_bound_cost:
            self._points.append(ClusteredPoint(
                point_index=point_index,
                cluster_index=self._cluster_index,
                cost=point_cost,
            ))
            self._total_cost += point_cost
            return True
        return False

    @property
    def total_cost(self) -> float:
        return self._total_cost

    @property
    def range_value(self) -> int:
        return self._range_value

    @property
    def lower_bound_cost(self) -> float:
        return self._lower_bound_cost

    @property
    def upper_bound_cost(self) -> float:
        return self._upper_bound_cost
    
    def count_points(self) -> int:
        return len(self._points)


class RingSet:
    def __init__(self, range_start: int, range_end: int, n_clusters: int) -> None:
        self._range_start = range_start
        self._range_end = range_end
        self._n_clusters = n_clusters
        self._rings: Dict[Tuple[int, int], Ring] = []
        self._overshot_points: List[RinglessPoint] = []
        self._shortfall_points: List[RinglessPoint] = []

    def find_or_create(self, cluster_index: int, range_value: int, average_cluster_cost: float) -> Ring:
        key = tuple(cluster_index, range_value)
        if key not in self._rings:
            self._rings[key] = Ring(
                cluster_index=cluster_index,
                range_value=range_value,
                average_cluster_cost=average_cluster_cost
            )
        return self._rings[key]

    def add_overshot_point(self, point_index: int, cluster_index: int, point_cost: float, cost_boundary: float) -> None:
        self._overshot_points.append(RinglessPoint(
            point_index=point_index,
            cluster_index=cluster_index,
            point_cost=point_cost,
            cost_boundary=cost_boundary,
            is_overshot=True,
        ))

    def add_shortfall_point(self, point_index: int, cluster_index: int, point_cost: float, cost_boundary: float) -> None:
        self._shortfall_points.append(RinglessPoint(
            point_index=point_index,
            cluster_index=cluster_index,
            point_cost=point_cost,
            cost_boundary=cost_boundary,
            is_overshot=False,
        ))

    def calc_ring_cost(self, ring_range_value: int) -> float:
        """Sums the costs of all points in captured by all clusters for a given ring range i.e., cost(R_l) = sum_{p in R_l} cost(p, A)."""
        costs = [
            ring.total_cost
            for ring in self._rings.values()
            if ring.range_value == ring_range_value
        ]
        return np.sum(costs)

    def count_ring_points(self, ring_range_value: int) -> int:
        """Counts the number of points captured by a given ring range l i.e., |R_l|."""
        counts = [
            ring.count_points
            for ring in self._rings.values()
            if ring.range_value == ring_range_value
        ]
        return np.sum(counts)

    def get_number_of_shortfall_points(self, cluster_index: int) -> int:
        """Counts the number of shortfall points in a given cluster."""
        counts = [
            1
            for point in self._shortfall_points
            if point.cluster_index == cluster_index and point.is_overshot == False
        ]
        return np.sum(counts)

    def compute_cost_of_overshot_points(self, cluster_index: int=-1) -> float:
        """Computes the cost of overshot points. 
        If -1 then compute the costs for all points. 
        If non-negative, computes the cost for all points in the given cluster."""
        costs = [
            point.point_cost
            for point in self._overshot_points
            if cluster_index == -1 or (cluster_index >= 0 and point.cluster_index == cluster_index)
        ]
        return np.sum(costs)

    def get_overshot_points(self, cluster_index: int) -> List[RinglessPoint]:
        return [
            point
            for point in self._overshot_points
            if point.cluster_index == cluster_index
        ]


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