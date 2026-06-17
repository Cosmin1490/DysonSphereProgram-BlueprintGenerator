import tensorflow as tf
import numpy as np
from scipy.spatial import KDTree
from .BaseOptimizer import BaseOptimizer


class ThresholdPenaltyOptimizer(BaseOptimizer):
    """
    Hybrid optimizer that combines Coulomb energy repulsion with a soft penalty
    for pairs that violate the game's minimum distance constraint.

    Intended use: run EnergyOptimizer first to get a good global distribution,
    then feed those points here to push violating pairs apart.

    Loss = energy_weight * coulomb_repulsion
         + penalty_weight * sum(max(0, threshold - sq_dist)^2)
         + surface_weight * surface_constraint
    """

    def __init__(self, points, min_distance=0.00511225, energy_weight=1.0,
                 penalty_weight=1e6, surface_weight=10000,
                 learning_rate=0.001, num_epochs=10000):
        super().__init__()
        self.n = len(points)
        self.min_distance = min_distance
        self.initial_energy_weight = energy_weight
        self.initial_penalty_weight = penalty_weight
        self.energy_weight = tf.Variable(energy_weight, dtype=tf.float64)
        self.penalty_weight = tf.Variable(penalty_weight, dtype=tf.float64)
        self.surface_weight = surface_weight
        self.x = tf.Variable(points, dtype=tf.float64)
        self.num_epochs = num_epochs
        lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=learning_rate,
            decay_steps=num_epochs,
            alpha=learning_rate * 0.001  # decay to 0.1% of initial LR
        )
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        self.best_x = tf.Variable(points, dtype=tf.float64)
        self.best_min_dist = 0.0

    def _pairwise_sq_distances(self, points):
        """Compute pairwise squared Euclidean distances."""
        dot = tf.matmul(points, tf.transpose(points))
        sq_norms = tf.linalg.diag_part(dot)
        sq_dists = (tf.reshape(sq_norms, (-1, 1))
                    + tf.reshape(sq_norms, (1, -1))
                    - 2 * dot)
        return sq_dists

    @tf.function
    def compute_loss(self):
        epsilon = 1e-7

        sq_dists = self._pairwise_sq_distances(self.x)

        # --- Coulomb energy term (global uniformity) ---
        pairwise_dist = tf.sqrt(sq_dists + epsilon)
        mask = 1.0 - tf.eye(self.n, dtype=tf.float64)
        reciprocal = mask / (pairwise_dist + tf.eye(self.n, dtype=tf.float64))
        energy = tf.reduce_sum(reciprocal) / 2

        # --- Threshold penalty term (push violations apart) ---
        # Only penalize pairs closer than min_distance (squared)
        violation = tf.nn.relu(self.min_distance - sq_dists)
        # Zero out diagonal (self-distances)
        violation = violation * mask
        penalty = tf.reduce_sum(violation * violation) / 2

        # --- Surface constraint (stay on unit sphere) ---
        norms = tf.sqrt(tf.linalg.diag_part(tf.matmul(self.x, tf.transpose(self.x))) + epsilon)
        surface = tf.reduce_sum((norms - 1.0) ** 2)

        loss = (self.energy_weight * energy
                + self.penalty_weight * penalty
                + self.surface_weight * surface) / self.n
        return loss

    @tf.function
    def _train_step(self):
        with tf.GradientTape() as tape:
            loss = self.compute_loss()
        gradients = tape.gradient(loss, [self.x])
        self.optimizer.apply_gradients(zip(gradients, [self.x]))
        return loss

    def _compute_min_sq_distance(self):
        """Compute the current minimum squared distance between any two points (on unit sphere)."""
        points = self.x.numpy()
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        projected = points / norms
        tree = KDTree(projected)
        distances, _ = tree.query(projected, k=2)
        return np.min(distances[:, 1]) ** 2

    def _update_weights(self, epoch):
        """Shift focus from uniformity to constraint satisfaction over time."""
        progress = epoch / self.num_epochs  # 0.0 -> 1.0
        # Energy weight decays from initial to 1% of initial
        self.energy_weight.assign(
            self.initial_energy_weight * (1.0 - 0.99 * progress)
        )
        # Penalty weight grows from initial to 100x initial
        self.penalty_weight.assign(
            self.initial_penalty_weight * (1.0 + 99.0 * progress)
        )

    def optimize(self):
        try:
            for epoch in range(self.num_epochs):
                self._update_weights(epoch)
                loss = self._train_step()

                if (epoch + 1) % 100 == 0:
                    min_sq_dist = self._compute_min_sq_distance()
                    valid = min_sq_dist >= self.min_distance
                    status = "VALID" if valid else "-----"

                    if min_sq_dist > self.best_min_dist:
                        self.best_min_dist = min_sq_dist
                        self.best_x.assign(self.x.value())

                    print(f'Epoch {epoch + 1}, Loss: {loss.numpy():.6f}, '
                          f'Min sq dist: {min_sq_dist:.8f} / {self.min_distance} '
                          f'[{status}] (best: {self.best_min_dist:.8f}) '
                          f'[E:{self.energy_weight.numpy():.4f} P:{self.penalty_weight.numpy():.0f}]')

                if (epoch + 1) % 10000 == 0:
                    updated_points = self.get_updated_points()
                    with open('./saves/threshold_penalty.txt', 'w') as f:
                        np.savetxt(f, updated_points)
        except KeyboardInterrupt:
            print("Optimization stopped by user.")
            print(f"Best min sq distance: {self.best_min_dist:.8f}")

    def get_updated_points(self):
        return self.best_x.numpy()
