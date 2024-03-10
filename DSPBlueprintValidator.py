import numpy as np

from scipy.spatial import KDTree

from Polyhedron import Polyhedron

class DSPBlueprintValidator:

    @staticmethod
    def validate_vertices(polyhedron, min_distance=0.00511225):
        vertices = Polyhedron.project_to_sphere(polyhedron.vertices, 1)
        tree = KDTree(vertices)
        for i in range(len(vertices)):
            # Query the KDTree to find the nearest neighbor distance
            distances, indices = tree.query(vertices[i], 2)  # 2 because the nearest neighbor will be the point itself
            nearest_neighbor_distance = distances[1] ** 2  # 0 index will be the distance to the point itself, which is 0
            if nearest_neighbor_distance < min_distance :
                return False
        return True

    @staticmethod
    def validate_polyhedron(polyhedron, min_distance=0.00511225):
        return DSPBlueprintValidator.validate_vertices(polyhedron, min_distance, tolerance)

    @staticmethod
    def correct_polyhedron(polyhedron, min_distance=0.00511225):
        def is_valid_vertex(vertex, tree):
            distances, indices = tree.query(vertex, 2)
            nearest_neighbor_distance = distances[1] ** 2
            return nearest_neighbor_distance >= min_distance

        vertices = Polyhedron.project_to_sphere(polyhedron.vertices, 1)
        tree = KDTree(vertices)

        # Continue checking and removing invalid vertices until no change
        while True:
            invalid_indices = [i for i, v in enumerate(vertices) if not is_valid_vertex(v, tree)]

            if not invalid_indices:
                break

            # Remove the first invalid vertex and update the tree
            index_to_remove = invalid_indices[0]
            polyhedron.delete_vertex(index_to_remove)
            vertices = np.delete(vertices, index_to_remove, axis=0)
            tree = KDTree(vertices)

        return polyhedron


