import numpy as np
from .BaseOptimizer import BaseOptimizer


class FibonacciSphereGenerator(BaseOptimizer):
    """
    Generates quasi-uniform points on a unit sphere using a Fibonacci spiral.

    Not an iterative optimizer — produces an analytically uniform distribution
    in O(n) time. Useful as an alternative to random + Coulomb for Stage 1.
    """

    def __init__(self, n=2582):
        super().__init__()
        self.n = n
        self.points = None

    def optimize(self):
        golden_angle = np.pi * (3 - np.sqrt(5))
        points = []
        for i in range(self.n):
            theta = golden_angle * i
            phi = np.arccos(1 - 2 * (i + 0.5) / self.n)
            x = np.sin(phi) * np.cos(theta)
            y = np.sin(phi) * np.sin(theta)
            z = np.cos(phi)
            points.append([x, y, z])
        self.points = np.array(points, dtype=np.float64)

    def get_updated_points(self):
        return self.points
