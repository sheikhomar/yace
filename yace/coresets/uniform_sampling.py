import numpy as np

from yace.coresets.sampling import SamplingBasedAlgorithm


class UniformSampling(SamplingBasedAlgorithm):
    def __init__(self, n_clusters: int, coreset_size: int) -> None:
        super().__init__(n_clusters, coreset_size)
    
    def run(self, X):
        n_points = X.shape[0]
        # Sample T input points uniformly at random without replacement
        sampled_indices = np.random.choice(a=n_points, size=self._coreset_size, replace=False)
        # Add sampled points to the coreset
        coreset_points = X[sampled_indices]
        # Weigh each coreset point by n / T
        coreset_weights = (n_points / self._coreset_size) * np.ones(shape=self._coreset_size)
        return coreset_points, coreset_weights
