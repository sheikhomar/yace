import dataclasses

from typing import Dict, List, Optional, Tuple

import numpy as np
import numba
import pandas as pd
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


@dataclasses.dataclass
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
        self._total_cost: float = 0.0
        self._total_points: int = 0

    def add_point(self, clustered_point: ClusteredPoint) -> None:
        key = clustered_point.cluster_index
        if not key in self._clustered_points:
            self._clustered_points[key] = []
        self._clustered_points[key].append(clustered_point)
        self._total_cost += clustered_point.cost
        self._total_points += 1

    @property
    def total_cost(self) -> float:
        return self._total_cost
    
    @property
    def count_points(self) -> int:
        """Counts the number of points assigned to this group."""
        return self._total_points

    def count_points_in_cluster(self, cluster_index: int) -> int:
        """Counts the number of points in the given cluster."""
        return len(self._clustered_points[cluster_index])
    
    def get_point_indices(self) -> List[int]:
        return [
            cp.point_index
            for points in self._clustered_points.values()
            for cp in points
        ]


class GroupSet:
    def __init__(self, group_range_size: int) -> None:
        self._group_range_size = group_range_size
        self._groups: List[Group] = []

    @property
    def size(self) -> int:
        return len(self._groups)

    @property
    def group_range_size(self) -> int:
        return self._group_range_size

    def add(self, group: Group) -> None:
        self._groups.append(group)

    def compute_normalized_costs(self) -> List[float]:
        costs = np.array([g.total_cost for g in self._groups])
        return costs / costs.sum()

    def iterator(self):
        for group in self._groups:
            yield group

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

    @property
    def count_points(self) -> int:
        return len(self._points)
    
    def add_points_to_group(self, group: Group) -> None:
        """Adds points in this ring to the given group."""
        for point in self._points:
            group.add_point(ClusteredPoint(
                point_index=point.point_index,
                cluster_index=point.cluster_index,
                cost=point.cost,
            ))


class RingSet:
    def __init__(self, range_start: int, range_end: int, n_clusters: int) -> None:
        self._range_start = range_start
        self._range_end = range_end
        self._n_clusters = n_clusters
        self._rings: Dict[Tuple[int, int], Ring] = []
        self._overshot_points: List[RinglessPoint] = []
        self._shortfall_points: List[RinglessPoint] = []

    @property
    def range_start(self) -> int:
        return self._range_start

    @property
    def range_end(self) -> int:
        return self._range_end
    
    def find(self, cluster_index: int, range_value: int) -> Optional[Ring]:
        key = tuple(cluster_index, range_value)
        return self._rings.get(key, None)

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


def stochastic_rounding(value: float) -> int:
    value_high, value_low = np.floor(value), np.ceil(value)
    proba = (value - value_low) / (value_high - value_low)
    if np.random.rand() < proba:
        return int(value_high) # Round up
    return int(value_low) # Round down


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

        rings = self._make_rings(point_costs=point_costs, cluster_labels=cluster_labels)

    def _make_rings(self, point_costs: np.ndarray, cluster_labels: np.ndarray) -> RingSet:
        ring_range_start = -int(np.floor(np.log10(self._beta)))
        ring_range_end = -ring_range_start

        n_points = point_costs.shape[0]
        k = self._n_clusters

        rings = RingSet(range_start=ring_range_start, range_end=ring_range_end, n_clusters=k)

        average_cluster_costs = pd.DataFrame(dict(cost=point_costs, label=cluster_labels)).groupby("label").mean()["cost"].to_numpy()
        
        for point_index in range(n_points):
            cluster_index = cluster_labels[point_index]
            point_cost = point_costs[point_index]

            # The average cost of cluster `c`: Δ_c
            average_cluster_cost = average_cluster_costs[cluster_index]

            point_put_in_ring = False
            for l in range(ring_range_start, ring_range_end):
                ring = rings.find_or_create(cluster_index=cluster_index, range_value=l, average_cluster_cost=average_cluster_cost)

                # Add point if cost(p, A) is within bounds i.e. between Δ_c*2^l and Δ_c*2^(l+1)
                if ring.try_add_point(point_index=point_index, point_cost=point_cost):
                    point_put_in_ring = True
                    break
            
            if point_put_in_ring == False:
                inner_most_ring_cost = average_cluster_cost * np.power(2, ring_range_start)
                outer_most_ring_cost = average_cluster_cost * np.power(2, ring_range_end + 1)

                if point_cost < inner_most_ring_cost:
                    # Track shortfall points: below l's lower range i.e. l<log⁡(1/β)
                    rings.add_shortfall_point(
                        point_index=point_index,
                        cluster_index=cluster_index,
                        point_cost=point_cost,
                        cost_boundary=inner_most_ring_cost
                    )
                elif point_cost > outer_most_ring_cost:
                    # Track overshot points: above l's upper range i.e., l>log⁡(β)
                    rings.add_overshot_point(
                        point_index=point_index,
                        cluster_index=cluster_index,
                        point_cost=point_cost,
                        cost_boundary=inner_most_ring_cost
                    )
                else:
                    raise ValueError("Point should either belong to a ring or be ring less.")

        return rings

    def _add_shortfall_points_to_coreset(self, rings: RingSet):
        """
        Handle points whose costs are below the lowest ring range i.e. l < log(1/beta).
        These are called shortfall points because they fall short of being captured by the
        inner-most ring. These points are snapped to the center of the assigned cluster by
        adding the centers to the coreset weighted by the number of shortfall points of
        that cluster.
        """

        cluster_indices = []
        weights = []
        
        for cluster_index in range(self._n_clusters):
            n_shortfall_points = rings.get_number_of_shortfall_points(cluster_index=cluster_index)
            if n_shortfall_points == 0:
                # Not all clusters may have shortfall points so skip those.
                continue
            
            weights.append(n_shortfall_points)
            cluster_indices.append(cluster_indices)
        
        return cluster_indices, weights

    def _group_overshot_points(self, rings: RingSet, groups: GroupSet, n_groups: int = 5):
        k = self._n_clusters
        total_cost = rings.compute_cost_of_overshot_points()

        for cluster_index in range(self._n_clusters):
            cluster_cost = rings.compute_cost_of_overshot_points(cluster_index=cluster_index)
            points = rings.get_overshot_points(cluster_index=cluster_index)

            if len(points) == 0:
                # If there are no overshot points in the current cluster then go to next cluster.
                continue
            
            # Determine which points to include in the group
            for j in range(n_groups):
                lower_bound = 1 / k * np.power(2, -j) * total_cost
                upper_bound = 1 / k * np.power(2, -j + 1) * total_cost
                should_add_points_into_group = False

                if j == 0:
                    # Group 0 has no upper bound.
                    should_add_points_into_group = cluster_cost >= lower_bound
                elif j == (groups.group_range_size-1):
                    # Group j has no lower bound.
                    should_add_points_into_group = cluster_cost < upper_bound
                else:
                    should_add_points_into_group = cluster_cost >= lower_bound and cluster_cost < upper_bound
            
            # Create a group and add overshot points to it.
            if should_add_points_into_group:
                l = np.power(2, 31) # Simply assign a large value
                group = Group(
                    range_value=j,
                    ring_range_value=l,
                    lower_bound_cost=lower_bound,
                    upper_bound_cost=upper_bound,
                )
                groups.add(group=group)
                for point in points:
                    group.add_point(ClusteredPoint(
                        point_index=point.point_index,
                        cluster_index=point.cluster_index,
                        cost=point.point_cost,
                    ))

    def _group_ring_points(self, rings: RingSet, groups: GroupSet):
        k = self._n_clusters

        for l in range(rings.range_start, rings.range_end+1):
            ring_cost = rings.calc_ring_cost(ring_range_value=l)

            for cluster_index in self._n_clusters:
                ring = rings.find(cluster_index=cluster_index, range_value=l)

                if ring is None or ring.count_points == 0:
                    # If no ring exist for the given range value `l` and cluster index,
                    # no points are captured by the current ring, then continue to the next ring.
                    continue

                cluster_cost = ring.total_cost

                # Determine which points to include in the group
                for j in range(groups.group_range_size):
                    lower_bound = 1 / k * np.power(2, -j) * ring_cost
                    upper_bound = 1 / k * np.power(2, -j + 1) * ring_cost
                    should_add_points_into_group = False

                    if j == 0:
                        # Group 0 has no upper bound.
                        should_add_points_into_group = cluster_cost >= lower_bound
                    elif j == (groups.group_range_size-1):
                        # Group j has no lower bound.
                        should_add_points_into_group = cluster_cost < upper_bound
                    else:
                        should_add_points_into_group = cluster_cost >= lower_bound and cluster_cost < upper_bound

                # Create a group and add ring points to it.
                if should_add_points_into_group:
                    l = np.power(2, 31) # Simply assign a large value
                    group = Group(range_value=j, ring_range_value=l, lower_bound_cost=lower_bound, upper_bound_cost=upper_bound)
                    groups.add(group=group)
                    ring.add_points_to_group(group=group)

    def add_sampled_points_from_groups_to_coreset(self, point_costs: np.ndarray, groups: GroupSet, min_sampling_size: int = 1):
        T = self._coreset_size
        k = self._n_clusters

        cluster_indices = []
        cluster_weights = []
        coreset_indices = []
        coreset_weights = []

        total_cost = np.sum(point_costs)

        # The number of remaining points to reach the requested coreset size.
        n_remaining = T

        # Tracks the groups that we need to sample points from.
        groups_to_sample_from = []

        # Track the total cost of the groups that we need to sample points from.
        sampling_group_total_cost = 0.0

        # Determine which groups to sample points from
        for group in groups.iterator():
            group_cost = group.total_cost
            normalized_group_cost = group_cost / total_cost
            n_samples = T * normalized_group_cost

            if n_samples < min_sampling_size:
                # We will not sample from this group because T_m is below the minimum threshold
                # Instead, we simply add cluster centers to the coreset.
                for cluster_index in range(self._n_clusters):
                    weight = group.count_points_in_cluster(cluster_index=cluster_index)
                    if weight > 0:
                        cluster_indices.append(cluster_index)
                        cluster_weights.append(weight)
            elif n_samples >= group.count_points:
                # Will not sample because T_m >= |G_m|.
                group_point_indices = group.get_point_indices()
                coreset_indices += group_point_indices
                coreset_weights += [1.0 for i in range(len(group_point_indices))]
                n_remaining -= len(group_point_indices)
            else:
                groups_to_sample_from.append(group)
                sampling_group_total_cost += group_cost

        for group in groups_to_sample_from:
            group_cost = group.total_cost
            normalized_group_cost = group_cost / sampling_group_total_cost

            n_samples = n_remaining * normalized_group_cost
        