import numpy as np

from yace.coresets.sampling import SamplingBasedAlgorithm


class UniformSampling(SamplingBasedAlgorithm):
    def __init__(self, n_clusters: int, coreset_size: int) -> None:
        super().__init__(n_clusters, coreset_size)
    
    def run(self, X):
        D = self._compute_kmeans_cluster_distances(X=X)
        point_costs = np.min(D, axis=1)

        sampled_indices, sampled_weights = self._uniformly_sample_points_for_coreset(point_costs=point_costs)

        coreset_points = X[sampled_indices]
        coreset_weights = sampled_weights

        return coreset_points, coreset_weights
    
    def _uniformly_sample_points_for_coreset(self, point_costs: np.ndarray):
        n_points = point_costs.shape[0]

        # Sample T points uniformly at random
        sampled_indices = np.random.choice(a=n_points, size=self._coreset_size, replace=True)

        # The weight of a sampled point p is: cost(A) / (T*cost(p,X))
        total_cost = np.sum(point_costs)
        scaled_costs = self._coreset_size * point_costs[sampled_indices]
        # Same as `weights = total_cost / scaled_costs` but avoids division by zero
        # which results in an array with no `inf` values.
        weights = np.true_divide(
            total_cost,
            scaled_costs,
            out=np.zeros_like(scaled_costs), 
            where=scaled_costs!=0,
        )

        return sampled_indices, weights
