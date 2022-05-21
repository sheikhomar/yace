from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp_sparse

from sklearn.metrics.pairwise import _euclidean_distances
from sklearn.preprocessing import normalize
from sklearn.utils.extmath import safe_sparse_dot

from yace.clustering.kmeans import kmeans_plusplus, kmeans_plusplus_with_weights


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


def generate_candidate_solution_for_adv_instance(data_matrix: np.ndarray, k: int, sample_size: int, rng: np.random.Generator):
    n_points, n_dim = data_matrix.shape
    sampled_indices = rng.choice(a=n_points, size=sample_size)
    sampled_points = data_matrix[sampled_indices]

    # We generate remaining points by sampling from points
    # in the unit circle. This can be done as follows:
    #  1) Sample X_i ~ N(0, 1) for i = 1, ..., d
    #  2) Compute lambda^2 = sum_i X_i^2
    #  3) Compute lambda = sqrt(lambda^2)
    #  4) Set X_i = X_i / lambda
    # Idea from: https://stats.stackexchange.com/questions/7977/how-to-generate-uniformly-distributed-points-on-the-surface-of-the-3-d-unit-sphe
    n_remaining_points = k - sample_size
    generated_points = rng.multivariate_normal(
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


class DistortionCalculator:
    def __init__(self,
        working_dir: Path,
        k: int,
        input_points: np.ndarray,
        coreset_points: np.ndarray,
        coreset_weights: np.ndarray,
        ) -> None:
        self.working_dir = working_dir
        self.input_points = input_points
        self.coreset_points = coreset_points
        self.coreset_weights = coreset_weights
        self.k = k
    
    def calc_distortions(self, solution_generator, solution_type: str, n_repetitions: int) -> pd.DataFrame:
        results = []
        for iter in range(n_repetitions):
            solution = solution_generator()

            coreset_cost = compute_coreset_cost(
                coreset_points=self.coreset_points,
                coreset_weights=self.coreset_weights,
                candidate_solution=solution,
            )

            input_cost = compute_input_cost(
                input_points=self.input_points,
                candidate_solution=solution,
            )

            solution = None
            del solution

            distortion = max(float(input_cost/coreset_cost), float(coreset_cost/input_cost))

            results.append(dict(
                iteration=iter,
                solution_type=solution_type,
                coreset_cost=coreset_cost,
                input_cost=input_cost,
                distortion=distortion,
            ))
        return pd.DataFrame(results)

    def on_random(self):
        dist_random_path = self.working_dir/"distortions-random-solutions.feather"
        if not dist_random_path.exists():
            n_dim = self.input_points.shape[1]
            random_solution_generator = lambda: generate_random_solution(n_dim=n_dim, k=self.k)

            df_dist_random = self.calc_distortions(
                solution_generator=random_solution_generator,
                solution_type="random",
                n_repetitions=50,
            )

            df_dist_random.to_feather(dist_random_path)
            # df_dist_random.to_csv(str(dist_random_path).replace(".feather", ".csv"))

    def on_convex(self):
        dist_convex_path = self.working_dir/"distortions-convex-solutions.feather"
        if not dist_convex_path.exists():
            convex_solution_generator = lambda: generate_random_points_within_convex_hull(
                data_matrix=self.input_points, k=self.k, n_samples=2,
            )
            df_dist_convex = self.calc_distortions(
                solution_generator=convex_solution_generator,
                solution_type="convex",
                n_repetitions=50,
            )

            df_dist_convex.to_feather(dist_convex_path)
            # df_dist_convex.to_csv(str(dist_convex_path).replace(".feather", ".csv"))

    def on_adv(self, ratio: float, rng: np.random.Generator, is_random_seed_fixed: bool=False):
        sol_type = f"adv{ratio:0.2f}".replace(".", "_")
        if is_random_seed_fixed:
            sol_type = f"{sol_type}_fixed"
        output_path = self.working_dir/f"distortions-{sol_type}-solutions.feather"
        if not output_path.exists():
            sample_size = int(np.power(self.k, ratio))
            solution_generator = lambda: generate_candidate_solution_for_adv_instance(
                data_matrix=self.input_points, k=self.k, sample_size=sample_size, rng=rng,
            )
            df_distortions = self.calc_distortions(
                solution_generator=solution_generator,
                solution_type=sol_type,
                n_repetitions=50,
            )
            df_distortions.to_feather(output_path)
            # df_distortions.to_csv(str(output_path).replace(".feather", ".csv"))

    def on_kmeans_plus_plus_input(self):
        solution_type = "kmeans++-on-input"
        output_path = self.working_dir/f"distortions-{solution_type}-solutions.feather"
        if not output_path.exists():
            solution_generator = lambda: generate_candidate_solution_via_kmeans_plus_plus(
                data_matrix=self.input_points, k=self.k,
            )

            distortions_list = []
            for i in range(2):
                df_local_distortions = self.calc_distortions(
                    solution_generator=solution_generator,
                    solution_type=solution_type,
                    n_repetitions=5,
                )
                max_distortion_idx = df_local_distortions['distortion'].idxmax()
                max_distortion_row = df_local_distortions.iloc[max_distortion_idx].copy()
                distortions_list.append(max_distortion_row.to_dict())

            df_distortions = pd.DataFrame(distortions_list)

            # Reset iteration column
            df_distortions["iteration"] = np.arange(len(distortions_list))
            
            df_distortions.to_feather(output_path)
            # df_distortions.to_csv(str(output_path).replace(".feather", ".csv"))

    def on_kmeans_plus_plus_coreset(self):
        solution_type = "kmeans++-on-coreset"
        output_path = self.working_dir/f"distortions-{solution_type}-solutions.feather"
        if not output_path.exists():

            def solution_generator():
                # Best solution is the solution with lowest cost over
                # 5 runs of k-means++ on weighted coreset points.
                best_solution = None
                best_cost = None
                for iteration in range(5):
                    center_indices = kmeans_plusplus_with_weights(
                        points=self.coreset_points, 
                        weights=self.coreset_weights,
                        n_clusters=self.k,
                    )
                    solution = self.coreset_points[center_indices]

                    cost = compute_coreset_cost(
                        coreset_points=self.coreset_points,
                        coreset_weights=self.coreset_weights,
                        candidate_solution=solution,
                    )

                    if best_cost is None or cost < best_cost:
                        best_cost = cost
                        best_solution = solution
                return best_solution

            df_distortions = self.calc_distortions(
                solution_generator=solution_generator,
                solution_type=solution_type,
                n_repetitions=1,
            )

            df_distortions.to_feather(output_path)
            # df_distortions.to_csv(str(output_path).replace(".feather", ".csv"))

    def calc_distortions_for_adv_instance(self, rng: np.random.Generator):
        self.on_adv(ratio=1/2, rng=rng)

    def calc_distortions_for_simple_instance(self):
        self.on_random()
        self.on_convex()
