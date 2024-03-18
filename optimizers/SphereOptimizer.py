import tensorflow as tf
import numpy as np

class SphereOptimizer:
    def __init__(self, points, learning_rate=0.00001, num_epochs=1000):
        self.n_points = len(points)
        self.points = tf.Variable(points, dtype=tf.float32)
        self.points.assign(tf.nn.l2_normalize(self.points, axis=-1))
        self.optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
        self.num_epochs = num_epochs

    def compute_loss(self):
        cosines = tf.matmul(self.points, tf.transpose(self.points))
        cosines += tf.linalg.diag(tf.fill((self.n_points,), -2.0))
        return tf.reduce_max(cosines)

    def optimize(self):
        try:
            for epoch in range(self.num_epochs):
                self.train_step()
                if epoch % 100 == 0:
                    print(f'Epoch {epoch + 1}, Loss: {self.compute_loss().numpy()}')
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
        return self.points.numpy()

# Example usage
#initial_points = np.random.normal(size=(100, 3))
#optimizer = SphereOptimizer(initial_points)
#optimizer.optimize()
#updated_points = optimizer.get_updated_points()
