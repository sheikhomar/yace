import numpy as np

from yace.coresets.sampling import SamplingBasedAlgorithm


class UniformSampling(SamplingBasedAlgorithm):
    def __init__(self, n_clusters: int, coreset_size: int) -> None:
        super().__init__(n_clusters, coreset_size)
    
    def run(self, X):
        n_points = X.shape[0]
        sampled_indices = np.random.choice(a=n_points, size=self._coreset_size, replace=False)

        coreset_points = X[sampled_indices]
        coreset_weights = np.ones(shape=self._coreset_size) * (n_points / self._coreset_size)

        return coreset_points, coreset_weights
