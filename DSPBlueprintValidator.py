import numpy as np

from scipy.spatial import KDTree

from Polyhedron import Polyhedron

class DSPBlueprintValidator:

    @staticmethod
    def validate_vertices(polyhedron, min_distance=0.07155, tolerance=1e-9):
        vertices = Polyhedron.project_to_sphere(polyhedron.vertices, 1)
        tree = KDTree(vertices)
        for i in range(len(vertices)):
            # Query the KDTree to find the nearest neighbor distance
            distances, indices = tree.query(vertices[i], 2)  # 2 because the nearest neighbor will be the point itself
            nearest_neighbor_distance = distances[1]  # 0 index will be the distance to the point itself, which is 0
            if nearest_neighbor_distance - min_distance < tolerance:
                return False
        return True

    @staticmethod
    def validate_polyhedron(polyhedron, min_distance=0.07155, tolerance=1e-9):
        return DSPBlueprintValidator.validate_vertices(polyhedron, min_distance, tolerance)


