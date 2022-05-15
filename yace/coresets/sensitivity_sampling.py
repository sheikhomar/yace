import numpy as np

from yace.coresets.sampling import SamplingBasedAlgorithm


class SensitivitySampling(SamplingBasedAlgorithm):
    """Sensitivity sampling is non-uniform sampling coreset construction.

    First, it computes an initial solution using k-means++ for 2k centers.
    Then, it samples points proportionally to their distance to the initial solution,
    and adds them to the coreset. Each coreset point is assigned a weight that is
    inverse proportional to this distance.
    """

    def __init__(self, n_clusters: int, coreset_size: int) -> None:
        super().__init__(n_clusters, coreset_size)
    
    def run(self, A):
        # Compute squared Euclidean distances between the input points 
        # and points in the initial solution K: 
        #   D_{i,j} = ||p_i - K_j||^2_2
        D = self._compute_initial_solution_via_kmeans_plus_plus(A=A)

        sampled_indices, sampled_weights = self._non_uniform_sample_points_for_coreset(sq_pairwise_distances=D)

        # First add sampled points to the coreset
        coreset_points = A[sampled_indices]

        # Combine coreset weights
        coreset_weights = sampled_weights

        return coreset_points, coreset_weights

    def _non_uniform_sample_points_for_coreset(self, sq_pairwise_distances: np.ndarray):

        # For each input point p in A compute:
        #  cost(p, K) = min_{k in K} ||p-k||^2
        # where K is the initial solution
        point_costs = np.min(sq_pairwise_distances, axis=1)

        n_points = point_costs.shape[0]

        # Compute the total cost: cost_A(K) = sum_{p in A} cost_A(p, K)
        # where K is the initial solution obtained by running k-means++.
        total_cost = np.sum(point_costs)

        # Compute the sampling distribution: cost_A(p, K) / cost_A(K)
        sampling_distribution = point_costs / total_cost

        # Sample T points using the computed sampling distribution
        sampled_indices = np.random.choice(a=n_points, size=self._coreset_size, replace=True, p=sampling_distribution)

        # Compute the weights as 1 / sampling_distribution
        # sampled_proba = sampling_distribution[sampled_indices]
        # sampled_proba = sampled_proba / np.sum(sampled_proba)
        # weights = 1 / sampled_proba
        weights = np.sum(point_costs[sampled_indices]) / point_costs[sampled_indices]

        print(f"N={n_points}    T={self._coreset_size}    Sum of weights={weights.sum():0.0f}")

        return sampled_indices, weights

    def _non_uniform_sample_points_for_coreset_biased(self, sq_pairwise_distances: np.ndarray):
        # For each input point p in A compute:
        #  cost(p, K) = min_{k in K} ||p-k||^2
        # where K is the initial solution
        point_costs = np.min(sq_pairwise_distances, axis=1)

        n_points = point_costs.shape[0]

        # Compute the total cost: cost_A(K) = sum_{p in A} cost_A(p, K)
        # where K is the initial solution obtained by running k-means++.
        total_cost = np.sum(point_costs)

        # Compute the sampling distribution: cost_A(p, K) / cost_A(K)
        sampling_distribution = point_costs / total_cost

        # Sample T points using the computed sampling distribution
        sampled_indices = np.random.choice(a=n_points, size=self._coreset_size, replace=True, p=sampling_distribution)

        # Compute the weights
        average_cost_per_coreset_point = total_cost / self._coreset_size
        weights = average_cost_per_coreset_point / point_costs[sampled_indices]

        return sampled_indices, weights
