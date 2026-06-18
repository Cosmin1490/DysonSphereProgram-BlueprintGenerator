import numpy as np
import tensorflow as tf
from scipy.spatial import KDTree
from .BaseOptimizer import BaseOptimizer
from .ThresholdPenaltyOptimizer import ThresholdPenaltyOptimizer


class PerturbOptimizer(BaseOptimizer):
    """
    Basin hopping optimizer that escapes local minima by:
    1. Identifying the worst-violating node pairs
    2. Randomly perturbing those nodes
    3. Reoptimizing with a fresh ThresholdPenaltyOptimizer (fresh Adam state)
    4. Keeping the result if it improved

    Intended use: Stage 3, after EnergyOptimizer + ThresholdPenaltyOptimizer
    have converged to a local minimum near the target.
    """

    def __init__(self, points, min_distance=0.00511225, num_rounds=100,
                 top_k_pairs=20, initial_perturb_scale=0.03,
                 perturb_decay=0.97, reopt_epochs=20000):
        super().__init__()
        self.points = np.array(points, dtype=np.float64)
        self.min_distance = min_distance
        self.num_rounds = num_rounds
        self.top_k_pairs = top_k_pairs
        self.initial_perturb_scale = initial_perturb_scale
        self.perturb_decay = perturb_decay
        self.reopt_epochs = reopt_epochs
        self.best_points = self.points.copy()
        self.best_min_dist = 0.0

    def _project_to_sphere(self, points):
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        return points / norms

    def _compute_min_sq_distance(self, points):
        projected = self._project_to_sphere(points)
        tree = KDTree(projected)
        distances, _ = tree.query(projected, k=2)
        return np.min(distances[:, 1]) ** 2

    def _find_worst_nodes(self, points, top_k):
        """Find unique node indices involved in the top_k closest pairs."""
        projected = self._project_to_sphere(points)
        tree = KDTree(projected)
        distances, indices = tree.query(projected, k=2)
        sq_dists = distances[:, 1] ** 2

        # Get the top_k nodes with smallest nearest-neighbor distance
        worst_indices = np.argsort(sq_dists)[:top_k]
        # Also include their nearest neighbors
        neighbor_indices = indices[worst_indices, 1]
        all_indices = np.unique(np.concatenate([worst_indices, neighbor_indices]))
        return all_indices

    def optimize(self):
        self.best_min_dist = self._compute_min_sq_distance(self.points)
        self.best_points = self.points.copy()

        print(f"PerturbOptimizer: starting min sq dist = {self.best_min_dist:.8f} "
              f"/ {self.min_distance}")

        try:
            for round_num in range(self.num_rounds):
                scale = max(
                    self.initial_perturb_scale * (self.perturb_decay ** round_num),
                    0.008  # floor: don't let perturbation become negligible
                )

                # Find the bottleneck nodes
                worst_nodes = self._find_worst_nodes(self.best_points, self.top_k_pairs)

                # Perturb those nodes
                candidate = self.best_points.copy()
                perturbation = np.random.randn(len(worst_nodes), 3) * scale
                candidate[worst_nodes] += perturbation
                # Re-normalize to unit sphere
                norms = np.linalg.norm(candidate[worst_nodes], axis=1, keepdims=True)
                candidate[worst_nodes] = candidate[worst_nodes] / norms

                # Reoptimize with fresh optimizer state
                opt = ThresholdPenaltyOptimizer(
                    np.array(candidate, dtype=np.float64),
                    min_distance=self.min_distance,
                    num_epochs=self.reopt_epochs,
                    energy_weight=0.1,
                    penalty_weight=1e7,
                    softmin_weight=1e4,
                    softmin_alpha=500.0,
                    learning_rate=0.0005,
                    verbose=False,
                )
                opt.optimize()
                result = opt.get_updated_points()
                del opt
                tf.keras.backend.clear_session()

                new_min_dist = self._compute_min_sq_distance(result)

                if new_min_dist > self.best_min_dist:
                    self.best_min_dist = new_min_dist
                    self.best_points = result.copy()
                    valid = new_min_dist >= self.min_distance
                    status = "VALID" if valid else "improved"
                    print(f"Round {round_num + 1}: {status} -> {new_min_dist:.8f} "
                          f"/ {self.min_distance} (scale={scale:.6f}, "
                          f"perturbed {len(worst_nodes)} nodes)")

                    with open('./saves/perturb.txt', 'w') as f:
                        np.savetxt(f, self.best_points)

                    if valid:
                        print("TARGET REACHED!")
                        break
                else:
                    print(f"Round {round_num + 1}: no improvement "
                          f"({new_min_dist:.8f} vs best {self.best_min_dist:.8f}, "
                          f"scale={scale:.6f})")

        except KeyboardInterrupt:
            print(f"\nStopped. Best min sq distance: {self.best_min_dist:.8f}")

    def get_updated_points(self):
        return self.best_points
