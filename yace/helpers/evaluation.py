import numpy as np

from sklearn.metrics.pairwise import _euclidean_distances
from sklearn.preprocessing import normalize


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
