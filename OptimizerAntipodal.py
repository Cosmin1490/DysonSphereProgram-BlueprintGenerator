import tensorflow as tf
import numpy as np

class Optimizer:
    def __init__(self, points, k=10000, learning_rate=0.0001, num_epochs=10000):
        self.n = len(points)
        assert self.n % 2 == 0, "Number of points must be even."
        self.k = k
        self.x = tf.Variable(points[:self.n//2], dtype=tf.float64)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.num_epochs = num_epochs
        self.best_x = tf.Variable(self.x.value(), dtype=tf.float64)
        self.best_loss = float('inf')

    def compute_antipodal_points(self):
        return -1 * self.x

    def compute_full_points(self):
        return tf.concat([self.x, self.compute_antipodal_points()], axis=0)

    @tf.function
    def compute_loss(self):
        full_points = self.compute_full_points()
        epsilon = 1e-7
        tf_S = tf.matmul(full_points, tf.transpose(full_points))
        tf_pp_sq_dist = tf.linalg.diag_part(tf_S)
        tf_p_roll = tf.tile(tf.reshape(tf_pp_sq_dist, (1, -1)), (self.n, 1))
        tf_q_roll = tf.tile(tf.reshape(tf_pp_sq_dist, (-1, 1)), (1, self.n))
        tf_pq_sq_dist = tf_p_roll + tf_q_roll - 2 * tf_S
        tf_pq_dist = tf.sqrt(tf_pq_sq_dist + epsilon)
        tf_pp_dist = tf.sqrt(tf_pp_sq_dist + epsilon)
        tf_surface_dist_sq = (tf_pp_dist - tf.ones(self.n, dtype=tf.float64)) ** 2
        tf_rec_pq_dist = 1 / (tf_pq_dist + tf.eye(self.n, dtype=tf.float64)) - tf.eye(self.n, dtype=tf.float64)
        L_tf = (tf.reduce_sum(tf_rec_pq_dist) / 2 + self.k * tf.reduce_sum(tf_surface_dist_sq)) / self.n
        return L_tf

    @tf.function
    def optimize_step(self):
        with tf.GradientTape() as tape:
            loss = self.compute_loss()
        gradients = tape.gradient(loss, [self.x])
        self.optimizer.apply_gradients(zip(gradients, [self.x]))
        return loss

    def optimize(self):
        try:
            for epoch in range(self.num_epochs):
                loss = self.optimize_step()

                # Update best_x and best_loss if a better configuration is found
                if loss.numpy() < self.best_loss:
                    self.best_x.assign(self.x.value())
                    self.best_loss = loss.numpy()
                if (epoch + 1) % 10000 == 0:
                    updated_points = self.get_updated_points()
                    with open('stage1.txt', 'w') as f:
                        np.savetxt(f, updated_points)

                print(f'Epoch {epoch + 1}, Loss: {loss.numpy()}')
        except KeyboardInterrupt:
            print("Optimization stopped by user.")

    def get_updated_points(self):
        return np.concatenate([self.best_x.numpy(), -1 * self.best_x.numpy()], axis=0)
