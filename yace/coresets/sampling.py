import numpy as np

from sklearn.utils.extmath import row_norms
from sklearn.utils.validation import check_random_state
from sklearn.metrics.pairwise import _euclidean_distances

from yace.clustering.kmeans import kmeans_plusplus


class SamplingBasedAlgorithm:
    def __init__(self, n_clusters: int, coreset_size: int) -> None:
        self._n_clusters = n_clusters
        self._coreset_size = coreset_size

    def _compute_initial_solution_via_kmeans_plus_plus(self, A):
        """Computes an initial solution K by running k-means++ 
        the on input data set, and then compute squared
        Euclidean distances between the input points and
        points in the initial solution K.
        
        Returns a matrix D where entry D_{i,j} = ||p_i - K_j||^2_2
        """
        # Precompute squared norms of data points
        x_squared_norms = row_norms(A, squared=True)
        random_state = check_random_state(None)
        initial_solution_indices = kmeans_plusplus(
            X=A,
            n_clusters=self._n_clusters,
            random_state=random_state,
            x_squared_norms=x_squared_norms,
        )

        D = _euclidean_distances(
            X=A,
            Y=A[initial_solution_indices],
            X_norm_squared=x_squared_norms,
            Y_norm_squared=x_squared_norms[initial_solution_indices],
            squared=True
        )

        return D 
