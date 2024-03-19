import numpy as np

from scipy.spatial import KDTree

from Polyhedron import Polyhedron

class DSPBlueprintValidator:

    @staticmethod
    def validate_vertices(polyhedron, min_distance=0.00511225):
        vertices = Polyhedron.project_to_sphere(polyhedron.vertices, 1)
        tree = KDTree(vertices)
        for i in range(len(vertices)):
            distances, indices = tree.query(vertices[i], 2)
            nearest_neighbor_distance = distances[1] ** 2
            if nearest_neighbor_distance < min_distance:
                return False
        return True

    @staticmethod
    def validate_edges(polyhedron, min_distance=0.00275625):
        vertices = np.array(polyhedron.vertices)
        for edge in polyhedron.edges:
            v1, v2 = edge
            for i, vertex in enumerate(vertices):
                if i not in edge:
                    distance_sqr = DSPBlueprintValidator.point_to_segment_sqr(vertices[v1], vertices[v2], vertex)
                    if distance_sqr < min_distance:
                        return False
        return True

    @staticmethod
    def validate_polyhedron(polyhedron, min_distance=0.00511225):
        return (
            DSPBlueprintValidator.validate_vertices(polyhedron, min_distance) and
            DSPBlueprintValidator.validate_edges(polyhedron, min_distance)
        )

    @staticmethod
    def correct_polyhedron(polyhedron, min_distance=0.00511225):
        def is_valid_vertex(vertex, tree):
            distances, indices = tree.query(vertex, 2)
            nearest_neighbor_distance = distances[1] ** 2
            return nearest_neighbor_distance >= min_distance

        vertices = Polyhedron.project_to_sphere(polyhedron.vertices, 1)
        tree = KDTree(vertices)

        while True:
            invalid_indices = [i for i, v in enumerate(vertices) if not is_valid_vertex(v, tree)]

            if not invalid_indices:
                break

            index_to_remove = invalid_indices[0]
            polyhedron.delete_vertex(index_to_remove)
            vertices = np.delete(vertices, index_to_remove, axis=0)
            tree = KDTree(vertices)

        return polyhedron

    @staticmethod
    def point_to_segment_sqr(begin, end, point):
        rhs = end - begin
        sqr_magnitude = np.sum(rhs ** 2)

        if sqr_magnitude < 1e-10:
            return np.sum((point - begin) ** 2)

        num2 = np.clip(np.dot(point - begin, rhs) / sqr_magnitude, 0, 1)
        return np.sum(((begin + num2 * rhs) / np.linalg.norm(begin + num2 * rhs) - point) ** 2)
