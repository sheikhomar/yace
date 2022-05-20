import numpy as np

from yace.coresets.sampling import SamplingBasedAlgorithm


class SensitivitySamplingExtended(SamplingBasedAlgorithm):
    """Sensitivity sampling is non-uniform sampling coreset construction.

    First, it computes an initial solution using k-means++ for 2k centers.
    Then, it samples points proportionally to their distance to the initial solution,
    and adds them to the coreset. Each coreset point is assigned a weight that is
    inverse proportional to this distance.
    """

    def __init__(self, n_clusters: int, coreset_size: int, adjust_weights: bool) -> None:
        super().__init__(n_clusters, coreset_size)
        self._adjust_weights = adjust_weights

    def run(self, A):
        # Compute an initial solution K. Then compute squared Euclidean distances
        # between the input points and points in the initial solution K. Build
        # a distance matrix D such that the (i,j)-th entry is:
        #   D_{i,j} = ||p_i - K_j||^2_2
        sq_pairwise_distances = self._compute_initial_solution_via_kmeans_plus_plus(A=A)

        # For each input point p in A compute:
        #  cost(p, K) = min_{c in K} || p - c||^2
        # where K is the initial solution
        point_costs = np.min(sq_pairwise_distances, axis=1)

        n_points = point_costs.shape[0]

        # Compute sens(p) which is the sensitivity of each point p :
        #  cost(p, K)/cost(A, K) + 1/|C_p|
        # where
        #  - K is the initial solution obtained by running k-means++.
        #  - C_p is the set of points belonging to same cluster as p
        #  - cost(p, K) = min_{c in K} ||p - c||^2
        #  - cost(A, K) = sum_{q in A} min_{c in K} ||q - c||^2
        sensitivities = point_costs.copy()
        cluster_labels = np.argmin(sq_pairwise_distances, axis=1)
        for cluster_index in np.unique(cluster_labels):
            cluster_member_idx = np.where(cluster_labels == cluster_index)[0]
            cluster_cost = np.sum(point_costs[cluster_member_idx])
            cluster_size = cluster_member_idx.shape[0]
            if cluster_cost > 0:
                sensitivities[cluster_member_idx] = point_costs[cluster_member_idx] / cluster_cost
            sensitivities[cluster_member_idx] += 1 / cluster_size

        # Sample T points proportionate to their sensitivities
        sampling_proba = sensitivities / sensitivities.sum()
        sampled_indices = np.random.choice(a=n_points, size=self._coreset_size, replace=True, p=sampling_proba)

        # The weights of the sampled points are 1/sampling_proba * 1/T
        weights = 1 / (self._coreset_size * sampling_proba[sampled_indices])

        if self._adjust_weights:
            # Due to rounding errors, the weight for each point will
            # be off by a small number. Add the deviation back.
            weight_deviation = (n_points - weights.sum()) / self._coreset_size
            weights += weight_deviation

        # Add sampled points to the coreset
        coreset_points = A[sampled_indices]

        return coreset_points, weights
