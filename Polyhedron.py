import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

class Polyhedron:

    def __init__(self, vertices, faces):
        self._vertices = vertices
        self._faces = faces

    def __init__(self, vertices, faces):
        self._vertices = vertices
        self._faces = faces

    @property
    def vertices(self):
        return self._vertices

    @property
    def faces(self):
        return self._faces

    @staticmethod
    def icosahedron_vertices():
        phi = (1 + math.sqrt(5)) / 2
        vertices = [
            (-1, phi, 0),
            (1, phi, 0),
            (-1, -phi, 0),
            (1, -phi, 0),

            (0, -1, phi),
            (0, 1, phi),
            (0, -1, -phi),
            (0, 1, -phi),

            (phi, 0, -1),
            (phi, 0, 1),
            (-phi, 0, -1),
            (-phi, 0, 1)
        ]
        return vertices

    @staticmethod
    def icosahedron_faces():
        faces = [
            (0, 11, 5),
            (0, 5, 1),
            (0, 1, 7),
            (0, 7, 10),
            (0, 10, 11),

            (1, 5, 9),
            (5, 4, 11),
            (11, 2, 10),
            (10, 6, 7),
            (7, 8, 1),

            (3, 9, 4),
            (2, 3, 4),
            (6, 3, 2),
            (8, 3, 6),
            (9, 3, 8),

            (4, 9, 5),
            (4, 11, 2),
            (2, 6, 10),
            (6, 8, 7),
            (8, 9, 1),
        ]
        return faces

    @classmethod
    def create_icosahedron(cls):
        icosahedron_verts = cls.icosahedron_vertices()
        icosahedron_faces = cls.icosahedron_faces()
        return cls(icosahedron_verts, icosahedron_faces)

    @staticmethod
    def centroid(points):
        num_points = len(points)
        x_sum, y_sum, z_sum = 0, 0, 0
        for point in points:
            x_sum += point[0]
            y_sum += point[1]
            z_sum += point[2]
        return (x_sum / num_points, y_sum / num_points, z_sum / num_points)

    def kis_operator(self):
        new_faces = []
        new_vertices = list(self._vertices)  # Make a copy of the original vertices

        for face in self._faces:
            face_centroid = self.centroid([self._vertices[v] for v in face])
            centroid_idx = len(new_vertices)
            new_vertices.append(face_centroid)

            new_tri_faces = [(face[i], face[(i + 1) % len(face)], centroid_idx) for i in range(len(face))]
            new_faces.extend(new_tri_faces)

        self._vertices = new_vertices
        self._faces = new_faces

    def coxeter_operator(self):
        vertices = self._vertices
        faces = self._faces
        new_faces = []
        new_vertices = list(vertices)  # Make a copy of the original vertices
        edge_midpoint_indices = {}
        for face in faces:
            face_edge_midpoints = []
            for i in range(len(face)):
                v1 = face[i]
                v2 = face[(i + 1) % len(face)]

                edge_key = tuple(sorted([v1, v2]))
                if edge_key not in edge_midpoint_indices:
                    midpoint = self.centroid([vertices[v1], vertices[v2]])
                    midpoint_idx = len(new_vertices)
                    edge_midpoint_indices[edge_key] = midpoint_idx
                    new_vertices.append(midpoint)

                face_edge_midpoints.append(edge_midpoint_indices[edge_key])

            new_tri_faces = [
                (face[0], face_edge_midpoints[0], face_edge_midpoints[2]),
                (face[1], face_edge_midpoints[1], face_edge_midpoints[0]),
                (face[2], face_edge_midpoints[2], face_edge_midpoints[1]),
                (face_edge_midpoints[0], face_edge_midpoints[1], face_edge_midpoints[2])
            ]

            new_faces.extend(new_tri_faces)
        self._vertices = new_vertices
        self._faces = new_faces

    @staticmethod
    def normalize_vertex(vertex):
        norm = math.sqrt(vertex[0]**2 + vertex[1]**2 + vertex[2]**2)
        return (vertex[0]/norm, vertex[1]/norm, vertex[2]/norm)

    @staticmethod
    def angle(a, b):
        dx, dy, dz = a[0] - b[0], a[1] - b[1], a[2] - b[2]
        return math.atan2(dy, dx)

    @classmethod
    def angle_between_vectors(cls, v1, v2, ref):
        v1_normalized = cls.normalize_vertex(v1 - ref)
        v2_normalized = cls.normalize_vertex(v2 - ref)
        cosine_angle = np.dot(v1_normalized, v2_normalized)
        angle = np.arccos(np.clip(cosine_angle, -1, 1))
        return angle

    @classmethod
    def sort_adjacent_faces(cls, vertices, faces, common_vertex):
        if len(faces) <= 1:
            return faces

        ref_vector = vertices[faces[0]]
        sorted_faces = [faces[0]]
        remaining_faces = faces[1:]

        while remaining_faces:
            face_angles = [(face, cls.angle_between_vectors(ref_vector, vertices[face], common_vertex)) for face in remaining_faces]
            next_face, _ = min(face_angles, key=lambda x: x[1])
            ref_vector = vertices[next_face]
            sorted_faces.append(next_face)
            remaining_faces.remove(next_face)

        return sorted_faces

    def dual_operator(self):
        vertices = [np.array(v) for v in self._vertices]  # Convert input vertices to numpy arrays
        dual_vertices = [self.centroid([vertices[i] for i in face]) for face in self._faces]
        vertex_to_adjacent_faces = {i: [f_idx for f_idx, f in enumerate(self._faces) if i in f] for i in range(len(vertices))}

        dual_faces = []
        for vertex_index, adjacent_faces in vertex_to_adjacent_faces.items():
            sorted_faces = self.sort_adjacent_faces(dual_vertices, adjacent_faces, vertices[vertex_index])
            dual_faces.append(sorted_faces)

        self._vertices = dual_vertices
        self._faces = dual_faces

    @staticmethod
    def project_to_sphere(vertices, radius=1):
        projected_vertices = []
        for x, y, z in vertices:
            length = math.sqrt(x**2 + y**2 + z**2)
            x_proj, y_proj, z_proj = radius * x / length, radius * y / length, radius * z / length

            projected_vertices.append((x_proj, y_proj, z_proj))
        return projected_vertices

    def print_polyhedron(self, radius=1):
        vertices = self.project_to_sphere(self._vertices, radius)
        print("Index |        X        |        Y        |        Z        ")
        print("------|-----------------|-----------------|-----------------")
        for idx, vertex in enumerate(vertices):
            print("{:5d} | {: 15.8f} | {: 15.8f} | {: 15.8f}".format(idx, *vertex))

    def plot_polyhedron(self):
        vertices = self.project_to_sphere(self._vertices)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Create a list of polygons for each face
        polygons = [[vertices[vertex_idx] for vertex_idx in face] for face in self._faces]

        # Create a Poly3DCollection object
        poly3d = Poly3DCollection(polygons, edgecolor='k', lw=1, alpha=0.9)

        # Add the polygons to the axes
        ax.add_collection3d(poly3d)

        # Draw the vertices
        x, y, z = zip(*vertices)
        ax.scatter(x, y, z, color='r', s=3)

        # Set limits and labels
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        ax.set_axis_off()
        ax.set_box_aspect([1, 1, 1])

        # View the plot from different angles
        ax.view_init(elev=20, azim=-35)

        num_vertices = len(self._vertices)
        num_faces = len(self._faces)

        plt.annotate(
            f'Vertices: {num_vertices}\nFaces: {num_faces}',
            xy=(0, 1), xycoords='axes fraction',
            xytext=(5, -5), textcoords='offset points',
            fontsize=12,
            ha='left', va='top'
        )

        plt.show()


# Example usage:
icosahedron = Polyhedron.create_icosahedron()
icosahedron.coxeter_operator()
icosahedron.kis_operator()
icosahedron.dual_operator()
icosahedron.print_polyhedron()
icosahedron.plot_polyhedron()
