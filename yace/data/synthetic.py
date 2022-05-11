import numpy as np

from scipy.sparse import csr_matrix


def generate_simplex(d: int, alpha: float=1):
    """Generates an d-dimensional simplex."""
    data = np.ones(d) * alpha
    indices = np.arange(d)
    data = csr_matrix(arg1=(data, (indices, indices)), shape=(d, d))
    return data
