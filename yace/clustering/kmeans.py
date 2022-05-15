import numpy as np

import scipy.sparse as sp

from numba import jit
from sklearn.metrics.pairwise import _euclidean_distances
from sklearn.utils.extmath import stable_cumsum
from sklearn.utils.validation import check_random_state
from sklearn.utils.extmath import row_norms


def kmeans_plusplus(X, n_clusters, x_squared_norms=None, random_state=None, n_local_trials=None):
    """Computational component for initialization of n_clusters by
    k-means++. Prior validation of data is assumed.

    Parameters
    ----------
    X : {ndarray, sparse matrix} of shape (n_samples, n_features)
        The data to pick seeds for.

    n_clusters : int
        The number of seeds to choose.

    x_squared_norms : ndarray of shape (n_samples,)
        Squared Euclidean norm of each data point.

    random_state : RandomState instance
        The generator used to initialize the centers.
        See :term:`Glossary <random_state>`.

    n_local_trials : int, default=None
        The number of seeding trials for each center (except the first),
        of which the one reducing inertia the most is greedily chosen.
        Set to None to make the number of trials depend logarithmically
        on the number of seeds (2+log(k)); this is the default.

    Returns
    -------
    center_indices : ndarray of shape (n_clusters,)
        The index location of the chosen centers in the data array X. For a
        given index and center, X[index] = center.
    """
    n_samples = X.shape[0]

    random_state = check_random_state(random_state)

    if x_squared_norms is None:
        x_squared_norms = row_norms(X, squared=True)

    # Set the number of local seeding trials if none is given
    if n_local_trials is None:
        # This is what Arthur/Vassilvitskii tried, but did not report
        # specific results for other than mentioning in the conclusion
        # that it helped.
        n_local_trials = 2 + int(np.log(n_clusters))

    # Pick first center randomly and track index of point
    center_id = random_state.randint(n_samples)
    center_indices = np.full(n_clusters, -1, dtype=int)
    center_indices[0] = center_id

    # Initialize list of closest distances and calculate current potential
    closest_dist_sq = _euclidean_distances(
        X[[center_id]], X, Y_norm_squared=x_squared_norms, squared=True
    )
    current_pot = closest_dist_sq.sum()

    # Pick the remaining n_clusters-1 points
    for c in range(1, n_clusters):
        # Choose center candidates by sampling with probability proportional
        # to the squared distance to the closest existing center
        rand_vals = random_state.random_sample(n_local_trials) * current_pot
        candidate_ids = np.searchsorted(stable_cumsum(closest_dist_sq), rand_vals)
        # XXX: numerical imprecision can result in a candidate_id out of range
        np.clip(candidate_ids, None, closest_dist_sq.size - 1, out=candidate_ids)

        # Compute distances to center candidates
        distance_to_candidates = _euclidean_distances(
            X[candidate_ids], X, Y_norm_squared=x_squared_norms, squared=True
        )
        
        # update closest distances squared and potential for each candidate
        np.minimum(closest_dist_sq, distance_to_candidates, out=distance_to_candidates)
        candidates_pot = distance_to_candidates.sum(axis=1)

        # Decide which candidate is the best
        best_candidate = np.argmin(candidates_pot)
        current_pot = candidates_pot[best_candidate]
        closest_dist_sq = distance_to_candidates[best_candidate]
        best_candidate = candidate_ids[best_candidate]

        # Permanently add best center candidate found in local tries
        center_indices[c] = best_candidate

    return center_indices


@jit(nopython=True, fastmath=True)
def squared_distance(p1, p2):
    n_dim = p1.shape[0]
    distance = 0.0
    for d in range(n_dim):
        distance += (p1[d] - p2[d])**2
    return distance


@jit(nopython=True, fastmath=True)
def kmeans_plusplus_with_weights(points: np.ndarray, n_clusters: int, weights: np.ndarray):
    n_points = points.shape[0]

    if weights is None:
        weights = np.ones(n_points)

    center_indices = [-1 for _ in range(n_clusters)]

    # Pick first point uniformly at random
    center_indices[0] = np.random.choice(n_points)

    distances = np.zeros(shape=n_points)

    # Pick the remaining n_clusters-1 points
    for i in range(0, n_clusters-1):
        closest_dist_sum = 0.0
        for j in range(n_points):
            # Compute distance between the input point j and center point i
            new_distance = weights[j] * squared_distance(points[j], points[center_indices[i]])
            if i == 0:
                # This is the first center point so store the distance
                distances[j] = new_distance
            else:
                # Determine if the distance between input point j is closer to
                # the current center point i than any of the previous center points.
                prev_smallest_distance = distances[j]
                distances[j] = np.minimum(prev_smallest_distance, new_distance)
            
            # Keep a running sum of distances
            closest_dist_sum += distances[j]

        # Uniform sample a number from [0, closest_dist_sum)
        random_number = closest_dist_sum * np.random.random_sample()

        # Pick the next candidate
        candidate_index = 0
        current_sum = distances[0]
        while random_number >= current_sum:
            candidate_index += 1
            current_sum += distances[candidate_index]

        center_indices[i+1] = candidate_index

    return center_indices
