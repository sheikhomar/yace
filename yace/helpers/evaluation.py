import numpy as np
import scipy.sparse as sp_sparse

from sklearn.metrics.pairwise import _euclidean_distances
from sklearn.preprocessing import normalize
from sklearn.utils.extmath import safe_sparse_dot

from yace.clustering.kmeans import kmeans_plusplus


def compute_coreset_cost(coreset_points: np.ndarray, coreset_weights: np.ndarray, candidate_solution: np.ndarray):
    dist_sq = _euclidean_distances(X=coreset_points, Y=candidate_solution, squared=True)
    closest_dist_sq = np.min(dist_sq, axis=1)
    coreset_cost = np.sum(coreset_weights * closest_dist_sq)
    return coreset_cost


def compute_input_cost(input_points: np.ndarray, candidate_solution: np.ndarray) -> float:
    dist_sq = _euclidean_distances(X=input_points, Y=candidate_solution, squared=True)
    closest_dist_sq = np.min(dist_sq, axis=1)
    input_cost = np.sum(closest_dist_sq)
    return input_cost


def generate_random_solution(n_dim: int, k: int):
    """Generate a candidate solution that consists of a set of k d-dimensional random vectors.
    """
    solution = np.random.normal(loc=0, scale=1, size=(k, n_dim))
    solution = normalize(solution)
    return solution


def generate_random_points_within_convex_hull(data_matrix: np.ndarray, k: int, n_samples: int = 2) -> np.ndarray:
    """Generates k random points within the convex hull of the data.
    """
    n_points = data_matrix.shape[0]

    n_points, n_dim = data_matrix.shape
    point_indices = np.arange(start=0, stop=n_points)
    generated_points = []
    
    for _ in range(k):
        # Generate a random vector
        random_vector = np.random.rand(n_samples, 1)

        # Scale the random vector to L1 unit norm
        proba_vector = random_vector / random_vector.sum()

        # Select data points
        if n_samples is None or n_samples == n_points:
            selected_indices = point_indices
        else:
            selected_indices = np.random.choice(point_indices, n_samples)

        # Generate a point by taking the convex combination of the selected input points.
        if sp_sparse.issparse(data_matrix):
            new_point = safe_sparse_dot(proba_vector.T, data_matrix[selected_indices])
            new_point = sp_sparse.csr_array(new_point)
        else:
            new_point = np.dot(proba_vector.T, data_matrix[selected_indices])

        generated_points.append(new_point)

    if sp_sparse.issparse(data_matrix):
        generated_points = sp_sparse.vstack(generated_points, format="csr")
    else:
        generated_points = np.vstack(generated_points)

    return generated_points


def generate_candidate_solution_for_adv_instance(data_matrix: np.ndarray, k: int, sample_size: int):
    n_points, n_dim = data_matrix.shape
    sampled_indices = np.random.choice(a=n_points, size=sample_size)
    sampled_points = data_matrix[sampled_indices]

    # We generate remaining points by sampling from points
    # in the unit circle. This can be done as follows:
    #  1) Sample X_i ~ N(0, 1) for i = 1, ..., d
    #  2) Compute lambda^2 = sum_i X_i^2
    #  3) Compute lambda = sqrt(lambda^2)
    #  4) Set X_i = X_i / lambda
    # Idea from: https://stats.stackexchange.com/questions/7977/how-to-generate-uniformly-distributed-points-on-the-surface-of-the-3-d-unit-sphe
    n_remaining_points = k - sample_size
    generated_points = np.random.multivariate_normal(
        mean=np.zeros(n_dim),
        size=n_remaining_points,
        cov=np.eye(n_dim),
    )
    lambda_squared = np.sum(np.power(generated_points, 2), axis=1)
    lambdas = np.sqrt(lambda_squared)
    generated_points = generated_points / lambdas[:,None]

    solution = np.vstack([sampled_points, generated_points])
    return solution


def generate_candidate_solution_via_kmeans_plus_plus(data_matrix: np.ndarray, k: int):
    center_indices = kmeans_plusplus(X=data_matrix, n_clusters=k)
    solution = data_matrix[center_indices]
    return solution
