import numpy as np
from Polyhedron import Polyhedron

class DSPBlueprintValidator:

    @staticmethod
    def euclidean_distance(v1, v2):
        return np.sqrt(np.sum((np.array(v1) - np.array(v2)) ** 2))

    @staticmethod
    def validate_vertices(polyhedron, min_distance=0.0715, tolerance=1e-9):
        vertices = Polyhedron.project_to_sphere(polyhedron.vertices, 1)
        for face in polyhedron.faces:
            for i in range(len(face)):
                v1 = vertices[face[i]]
                v2 = vertices[face[(i + 1) % len(face)]]
                distance = DSPBlueprintValidator.euclidean_distance(v1, v2)
                if distance - min_distance < tolerance:
                    return False
        return True

    @staticmethod
    def validate_polyhedron(polyhedron, min_distance=0.0715, tolerance=1e-9):
        return DSPBlueprintValidator.validate_vertices(polyhedron, min_distance, tolerance)


