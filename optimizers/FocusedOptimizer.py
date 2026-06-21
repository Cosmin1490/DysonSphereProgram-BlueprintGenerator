import numpy as np
import tensorflow as tf
from scipy.spatial import KDTree
from .BaseOptimizer import BaseOptimizer


class FocusedOptimizer(BaseOptimizer):
    """
    Freezes most nodes and only optimizes the worst-violating ones.

    Instead of computing the full 2582x2582 pairwise distance matrix,
    only computes movable-movable and movable-frozen distances (~130x cheaper).
    Re-identifies worst nodes each round so focus naturally shifts.

    Intended use: Stage 3, after EnergyOptimizer + ThresholdPenaltyOptimizer
    have converged to ~99.9% of target.
    """

    def __init__(self, points, min_distance=0.00511225, num_rounds=200,
                 top_k=20, num_epochs=10000, perturb_scale=0.02,
                 penalty_weight=1e7, softmin_weight=1e4, softmin_alpha=500.0,
                 surface_weight=10000, learning_rate=0.001):
        super().__init__()
        self.all_points = np.array(points, dtype=np.float64)
        self.min_distance = min_distance
        self.num_rounds = num_rounds
        self.top_k = top_k
        self.perturb_scale = perturb_scale
        self.num_epochs = num_epochs
        self.penalty_weight = penalty_weight
        self.softmin_weight = softmin_weight
        self.softmin_alpha = softmin_alpha
        self.surface_weight = surface_weight
        self.learning_rate = learning_rate
        self.best_points = self.all_points.copy()
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

        worst_indices = np.argsort(sq_dists)[:top_k]
        neighbor_indices = indices[worst_indices, 1]
        all_indices = np.unique(np.concatenate([worst_indices, neighbor_indices]))
        return all_indices

    def _optimize_focused(self, all_points, movable_indices):
        """Run gradient descent on only the movable nodes."""
        frozen_mask = np.ones(len(all_points), dtype=bool)
        frozen_mask[movable_indices] = False
        frozen_indices = np.where(frozen_mask)[0]

        # Perturb movable nodes for random exploration
        movable_points = all_points[movable_indices].copy()
        movable_points += np.random.randn(*movable_points.shape) * self.perturb_scale
        norms = np.linalg.norm(movable_points, axis=1, keepdims=True)
        movable_points = movable_points / norms

        movable = tf.Variable(movable_points, dtype=tf.float64)
        frozen = tf.constant(all_points[frozen_indices], dtype=tf.float64)

        n_movable = len(movable_indices)
        min_dist = tf.constant(self.min_distance, dtype=tf.float64)
        penalty_w = tf.constant(self.penalty_weight, dtype=tf.float64)
        softmin_w = tf.constant(self.softmin_weight, dtype=tf.float64)
        softmin_a = tf.constant(self.softmin_alpha, dtype=tf.float64)
        surface_w = tf.constant(self.surface_weight, dtype=tf.float64)

        lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=self.learning_rate,
            decay_steps=self.num_epochs,
            alpha=self.learning_rate * 0.001
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

        @tf.function
        def compute_loss():
            epsilon = 1e-7

            # Movable-movable squared distances
            mm_dot = tf.matmul(movable, tf.transpose(movable))
            mm_norms = tf.linalg.diag_part(mm_dot)
            mm_sq_dists = (tf.reshape(mm_norms, (-1, 1))
                           + tf.reshape(mm_norms, (1, -1))
                           - 2 * mm_dot)

            # Movable-frozen squared distances
            mf_dot = tf.matmul(movable, tf.transpose(frozen))
            f_norms = tf.reduce_sum(frozen * frozen, axis=1)
            mf_sq_dists = (tf.reshape(mm_norms, (-1, 1))
                           + tf.reshape(f_norms, (1, -1))
                           - 2 * mf_dot)

            # --- Penalty term ---
            mm_mask = 1.0 - tf.eye(n_movable, dtype=tf.float64)
            mm_violation = tf.nn.relu(min_dist - mm_sq_dists) * mm_mask
            mf_violation = tf.nn.relu(min_dist - mf_sq_dists)
            penalty = (tf.reduce_sum(mm_violation * mm_violation) / 2
                       + tf.reduce_sum(mf_violation * mf_violation))

            # --- Softmin term ---
            mm_masked = mm_sq_dists + tf.eye(n_movable, dtype=tf.float64) * 1e6
            all_dists = tf.concat([
                tf.reshape(mm_masked, [n_movable, -1]),
                mf_sq_dists
            ], axis=1)
            softmin = -tf.reduce_logsumexp(-softmin_a * all_dists) / softmin_a

            # --- Surface constraint ---
            norms = tf.sqrt(tf.reduce_sum(movable * movable, axis=1) + epsilon)
            surface = tf.reduce_sum((norms - 1.0) ** 2)

            return (penalty_w * penalty
                    - softmin_w * softmin
                    + surface_w * surface) / tf.cast(n_movable, tf.float64)

        @tf.function
        def train_step():
            with tf.GradientTape() as tape:
                loss = compute_loss()
            gradients = tape.gradient(loss, [movable])
            optimizer.apply_gradients(zip(gradients, [movable]))
            return loss

        for epoch in range(self.num_epochs):
            train_step()

        return movable.numpy()

    def optimize(self):
        self.best_min_dist = self._compute_min_sq_distance(self.all_points)
        self.best_points = self.all_points.copy()

        print(f"FocusedOptimizer: starting min sq dist = {self.best_min_dist:.8f} "
              f"/ {self.min_distance}")

        try:
            for round_num in range(self.num_rounds):
                worst_nodes = self._find_worst_nodes(self.best_points, self.top_k)

                # Optimize only the worst nodes (with random perturbation)
                new_movable = self._optimize_focused(self.best_points, worst_nodes)

                # Update points
                candidate = self.best_points.copy()
                candidate[worst_nodes] = new_movable
                # Re-normalize to unit sphere
                norms = np.linalg.norm(candidate[worst_nodes], axis=1, keepdims=True)
                candidate[worst_nodes] = candidate[worst_nodes] / norms

                new_min_dist = self._compute_min_sq_distance(candidate)

                if new_min_dist > self.best_min_dist:
                    self.best_min_dist = new_min_dist
                    self.best_points = candidate.copy()
                    valid = new_min_dist >= self.min_distance
                    status = "VALID" if valid else "improved"
                    print(f"Round {round_num + 1}: {status} -> {new_min_dist:.8f} "
                          f"/ {self.min_distance} "
                          f"(optimized {len(worst_nodes)} nodes)")

                    with open('./saves/focused.txt', 'w') as f:
                        np.savetxt(f, self.best_points)

                    if valid:
                        print("TARGET REACHED!")
                        break
                else:
                    print(f"Round {round_num + 1}: no improvement "
                          f"({new_min_dist:.8f} vs best {self.best_min_dist:.8f}, "
                          f"optimized {len(worst_nodes)} nodes)")

                tf.keras.backend.clear_session()

        except KeyboardInterrupt:
            print(f"\nStopped. Best min sq distance: {self.best_min_dist:.8f}")

    def get_updated_points(self):
        return self.best_points
