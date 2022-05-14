import numpy as np

from sklearn.utils.extmath import row_norms
from sklearn.utils.validation import check_random_state
from sklearn.metrics.pairwise import _euclidean_distances

from yace.clustering.kmeans import kmeans_plusplus


class SamplingBasedAlgorithm:
    def __init__(self, n_clusters: int, coreset_size: int) -> None:
        self._n_clusters = n_clusters
        self._coreset_size = coreset_size

    def _compute_kmeans_cluster_distances(self, X):
        """Computes an initial solution K by running k-means++ 
        the on input data set, and then compute squared
        Euclidean distances between the input points and
        points in the initial solution K.
        
        Returns a matrix D where entry D_{i,j} = ||p_i - K_j||^2_2
        """
        # Precompute squared norms of data points
        x_squared_norms = row_norms(X, squared=True)
        random_state = check_random_state(None)
        center_indices = kmeans_plusplus(
            X=X,
            n_clusters=self._n_clusters,
            random_state=random_state,
            x_squared_norms=x_squared_norms,
        )

        D = _euclidean_distances(
            X=X,
            Y=X[center_indices],
            X_norm_squared=x_squared_norms,
            Y_norm_squared=x_squared_norms[center_indices],
            squared=True
        )

        return D 
