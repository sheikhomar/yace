import numpy as np

from scipy.sparse import csr_matrix


def generate_simplex(d: int, alpha: float=1):
    """Generates an d-dimensional simplex."""
    data = np.ones(d) * alpha
    indices = np.arange(d)
    data = csr_matrix(arg1=(data, (indices, indices)), shape=(d, d))
    return data


def generate_adv_instance(simplex_size: int, alpha: float, n_points_per_simplex: int, simplex_points_variance: float):
    n_dim = simplex_size
    simplexes = np.eye(simplex_size) * alpha
    cov = np.eye(simplex_size) * (simplex_points_variance / np.sqrt(n_dim))
    raw_data = [
        np.random.multivariate_normal(
            mean=simplexes[m],
            size=n_points_per_simplex,
            cov=cov,
            random_state=42,
        )
        for m in range(simplex_size)
    ]
    return np.vstack(raw_data)
