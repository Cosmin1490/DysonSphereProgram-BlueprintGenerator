import tensorflow as tf
import numpy as np

class Optimizer:
    def __init__(self, points, k=10000, learning_rate=0.0001, num_epochs=10000):
        self.n = len(points)
        self.k = k
        self.x = tf.Variable(points, dtype=tf.float64)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.num_epochs = num_epochs

    def compute_loss(self):
        epsilon = 1e-7
        tf_S = tf.matmul(self.x, tf.transpose(self.x))
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

    def optimize(self):
        for epoch in range(self.num_epochs):
            with tf.GradientTape() as tape:
                loss = self.compute_loss()
            gradients = tape.gradient(loss, [self.x])
            self.optimizer.apply_gradients(zip(gradients, [self.x]))
            print(f'Epoch {epoch + 1}, Loss: {loss.numpy()}')

    def get_updated_points(self):
        return self.x.numpy()

# Example usage:
#points = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
#
#optimizer = Optimizer(points)
#optimizer.optimize()
#new_points = optimizer.get_updated_points()
#print("Final values of x:")
#print(new_points)
