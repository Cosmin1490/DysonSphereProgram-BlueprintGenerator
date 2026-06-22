import numpy as np
from scipy.spatial import KDTree
from .BaseOptimizer import BaseOptimizer


class ForceDirectedOptimizer(BaseOptimizer):
    """
    Force-directed optimizer for Tammes sphere packing.

    Uses overdamped repulsive dynamics: pairs within a dynamic cutoff
    repel each other with linear forces. Produces jammed packings where
    nearly all contact pairs are at the same distance — the signature
    of optimal Tammes solutions.

    No TensorFlow, no smooth loss function. Pure numpy + KDTree.
    """

    def __init__(self, points, min_distance=0.00511225,
                 initial_step=0.1, decay=0.99999,
                 initial_cutoff_margin=3.0, final_cutoff_margin=1.05,
                 num_epochs=500000, verbose=True):
        super().__init__()
        self.n = len(points)
        self.min_distance = min_distance
        self.initial_step = initial_step
        self.step = initial_step
        self.decay = decay
        self.initial_cutoff_margin = initial_cutoff_margin
        self.final_cutoff_margin = final_cutoff_margin
        self.num_epochs = num_epochs
        self.verbose = verbose

        # Normalize to unit sphere
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        self.points = points / norms

        self.best_points = self.points.copy()
        self.best_min_dist_sq = 0.0

    def _compute_min_sq_dist(self, tree):
        dists, _ = tree.query(self.points, 2)
        return np.min(dists[:, 1]) ** 2

    def optimize(self):
        points = self.points

        try:
            for epoch in range(self.num_epochs):
                tree = KDTree(points)

                # Current minimum distance
                dists, _ = tree.query(points, 2)
                min_dist = np.min(dists[:, 1])
                min_dist_sq = min_dist ** 2

                # Annealing cutoff: wide early (global rearrangement), tight late (jammed packing)
                progress = epoch / self.num_epochs
                margin = self.initial_cutoff_margin - (self.initial_cutoff_margin - self.final_cutoff_margin) * progress
                cutoff = min_dist * margin

                # Find all interacting pairs (vectorized)
                pairs = tree.query_pairs(cutoff, output_type='ndarray')

                if len(pairs) > 0:
                    i_idx = pairs[:, 0]
                    j_idx = pairs[:, 1]

                    # Compute pairwise differences and distances
                    diffs = points[i_idx] - points[j_idx]
                    pair_dists = np.linalg.norm(diffs, axis=1, keepdims=True)

                    # Linear repulsive force: magnitude = (cutoff - dist)
                    # Direction: away from the other point (normalized)
                    magnitudes = (cutoff - pair_dists) / (pair_dists + 1e-15)
                    force_vectors = magnitudes * diffs

                    # Accumulate forces (Newton's 3rd law)
                    forces = np.zeros_like(points)
                    np.add.at(forces, i_idx, force_vectors)
                    np.add.at(forces, j_idx, -force_vectors)

                    # Apply forces
                    points = points + self.step * forces

                    # Project back to unit sphere
                    norms = np.linalg.norm(points, axis=1, keepdims=True)
                    points = points / norms

                # Track best
                if min_dist_sq > self.best_min_dist_sq:
                    self.best_min_dist_sq = min_dist_sq
                    self.best_points = points.copy()

                # Decay step size
                self.step *= self.decay

                # Progress reporting
                if (epoch + 1) % 100 == 0 and self.verbose:
                    valid = min_dist_sq >= self.min_distance
                    status = "VALID" if valid else "-----"
                    n_pairs = len(pairs) if len(pairs) > 0 else 0
                    print(f'Epoch {epoch + 1}, '
                          f'Min sq dist: {min_dist_sq:.8f} / {self.min_distance} '
                          f'[{status}] (best: {self.best_min_dist_sq:.8f}) '
                          f'[step:{self.step:.2e} margin:{margin:.2f} pairs:{n_pairs}]')

                if (epoch + 1) % 10000 == 0 and self.verbose:
                    with open('./saves/force_directed.txt', 'w') as f:
                        np.savetxt(f, self.best_points)

        except KeyboardInterrupt:
            if self.verbose:
                print("Optimization stopped by user.")
                print(f"Best min sq distance: {self.best_min_dist_sq:.8f}")

        self.points = points

    def get_updated_points(self):
        return self.best_points
