import numpy as np
import scipy.sparse as sp_sparse

from scipy.sparse import issparse

from yace.coresets.sampling import SamplingBasedAlgorithm


class GroupSamplingSampling(SamplingBasedAlgorithm):
    def __init__(self, n_clusters: int, coreset_size: int) -> None:
        super().__init__(n_clusters, coreset_size)
    
    def run(self, X):
        D = self._compute_kmeans_cluster_distances(X=X)
        
        cluster_labels = np.argmin(D, axis=1)
        point_costs = np.min(D, axis=1)
