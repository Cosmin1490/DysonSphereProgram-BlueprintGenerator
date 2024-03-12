import tensorflow as tf
import numpy as np

class SphereOptimizer:
    def __init__(self, points, learning_rate=0.00001, num_epochs=1000):
        self.n_points = len(points)
        assert self.n_points % 2 == 0, "Number of points must be even."
        self.points = tf.Variable(points[:self.n_points//2], dtype=tf.float32)
        self.points.assign(tf.nn.l2_normalize(self.points, axis=-1))
        self.optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
        self.num_epochs = num_epochs
        self.best_points = tf.Variable(self.points.value(), dtype=tf.float32)
        self.best_loss = float('inf')

    def compute_antipodal_points(self):
        return -1 * self.points

    def compute_full_points(self):
        return tf.concat([self.points, self.compute_antipodal_points()], axis=0)

    @tf.function
    def compute_loss(self):
        full_points = self.compute_full_points()
        # Calculate the pairwise Euclidean distances
        pairwise_sq_distances = tf.reduce_sum(tf.square(full_points[:, None] - full_points[None, :]), axis=-1)
        # Set the diagonal to a large value so that it doesn't affect the minimum distance
        pairwise_sq_distances += tf.linalg.diag(tf.fill((self.n_points,), np.inf))
        # Return the negative of the minimum pairwise squared distance as the loss
        return -tf.reduce_min(pairwise_sq_distances)

    def optimize(self):
        try:
            for epoch in range(self.num_epochs):
                self.train_step()
                loss = self.compute_loss().numpy()

                # Update best_points and best_loss if a better configuration is found
                if loss < self.best_loss:
                    self.best_points.assign(self.points.value())
                    self.best_loss = loss

                if epoch % 100 == 0:
                    print(f'Epoch {epoch + 1}, Loss: {loss}')
        except KeyboardInterrupt:
            print("Optimization stopped by user.")

    @tf.function
    def train_step(self):
        with tf.GradientTape() as tape:
            L = self.compute_loss()
        grads = tape.gradient(L, [self.points])
        self.optimizer.apply_gradients(zip(grads, [self.points]))
        self.points.assign(tf.nn.l2_normalize(self.points, axis=-1))

    def get_updated_points(self):
        return np.concatenate([self.best_points.numpy(), -1 * self.best_points.numpy()], axis=0)

# Example usage
#initial_points = np.random.normal(size=(100, 3))
#optimizer = SphereOptimizer(initial_points)
#optimizer.optimize()
#updated_points = optimizer.get_updated_points()
