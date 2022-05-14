import numpy as np
import scipy.sparse as sp_sparse

from scipy.sparse import issparse

from yace.coresets.sampling import SamplingBasedAlgorithm


class SensitivitySampling(SamplingBasedAlgorithm):
    def __init__(self, n_clusters: int, coreset_size: int) -> None:
        super().__init__(n_clusters, coreset_size)
    
    def run(self, A):
        # Compute squared Euclidean distances between the input points 
        # and points in the initial solution K: 
        #   D_{i,j} = ||p_i - K_j||^2_2
        D = self._compute_kmeans_cluster_distances(X=A)

        print(f"Distance matrix shape: {D.shape}")

        cluster_labels = np.argmin(D, axis=1)

        # For each input point p in A compute:
        #  cost(p, K) = min_{k in K} ||p-k||^2
        point_costs = np.min(D, axis=1)

        sampled_indices, sampled_weights = self._sample_points_for_coreset(point_costs=point_costs)

        center_weights = self._compute_cluster_center_weights(
            coreset_indices=sampled_indices,
            coreset_weights=sampled_weights,
            cluster_labels=cluster_labels,
        )

        # First add sampled points to the coreset
        coreset_points = [A[sampled_indices]]

        # Then add k-means++ cluster centers to the coreset
        for c in range(self._n_clusters):
            cluster_members = A[cluster_labels == c]

            # Compute the mean of the clusters
            centroid = cluster_members.mean(axis = 0)

            if issparse(A):
                centroid = sp_sparse.csr_array(centroid)

            coreset_points.append(centroid)

        # Combine coreset points
        if issparse(A):
            coreset_points = sp_sparse.vstack(coreset_points, format="csr")
        else:
            coreset_points = np.vstack(coreset_points)

        # Combine coreset weights
        coreset_weights = np.concatenate([sampled_weights, center_weights])

        return coreset_points, coreset_weights

    def _sample_points_for_coreset(self, point_costs: np.ndarray):
        n_points = point_costs.shape[0]

        # Compute the total cost: cost_A(K) = sum_{p in A} cost_A(p, K)
        # where K is the initial solution obtained by running k-means++.
        total_cost = np.sum(point_costs)

        # Compute the sampling distribution: cost_A(p, K) / cost_A(K)
        sampling_distribution = point_costs / total_cost

        sampled_indices = np.random.choice(a=n_points, size=self._coreset_size, replace=True, p=sampling_distribution)

        # We scale the cost of the sampled point by a factor of T i.e. T * cost_A(p, K)
        scaled_costs = self._coreset_size * point_costs[sampled_indices]

        # The weight of the sampled point is now: cost_A(K) / (T * cost_A(p, K))
        # Same as `weights = total_cost / scaled_costs` but avoids division by zero
        # which results in an array with no `inf` values.
        weights = np.true_divide(
            total_cost,
            scaled_costs,
            out=np.zeros_like(scaled_costs), 
            where=scaled_costs!=0,
        )

        # for i in range(len(sampled_indices)):
        #     print(f"Sampled point {sampled_indices[i]:3}  cost={scaled_costs[i]:0.5}  gets weight={weights[i]:0.5}")

        return sampled_indices, weights

    def _compute_cluster_center_weights(self, coreset_indices: np.ndarray, coreset_weights: np.ndarray, cluster_labels: np.ndarray):
        # Find the cluster labels of coreset points
        coreset_cluster_labels = cluster_labels[coreset_indices]

        # The cluster center weight is now sum_{p in coreset and p in C_i} coreset_weight where coreset_weight=cost(A)/(T*cost(p,A))
        center_weights = np.bincount(coreset_cluster_labels, weights=coreset_weights)

        if center_weights.shape[0] < self._n_clusters:
            # Ensure that there is weight for each cluster center even if no
            # points have been assigned to a cluster.
            m = self._n_clusters - center_weights.shape[0]
            center_weights = np.pad(center_weights, pad_width=(0, m))
        
        # Compute |C_i|  i.e. number of member points in each cluster
        cluster_indices, member_counts = np.unique(cluster_labels, return_counts=True)

        # Compute max(0, |C_i| - w_i)
        center_weights = np.maximum(0, member_counts - center_weights)

        return center_weights
