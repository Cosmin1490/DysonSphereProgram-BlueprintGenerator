#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

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

icosahedron_verts = icosahedron_vertices()
icosahedron_faces = [
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

def divide_faces(vertices, faces):
    new_faces = []
    new_vertices = list(vertices)  # Make a copy of the original vertices

    for face in faces:
        face_centroid = centroid([vertices[v] for v in face])
        centroid_idx = len(new_vertices)
        new_vertices.append(face_centroid)

        new_tri_faces = [(face[i], face[(i + 1) % len(face)], centroid_idx) for i in range(len(face))]
        new_faces.extend(new_tri_faces)

    return new_vertices, new_faces

def kis_operator(vertices, faces):
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
                midpoint = centroid([vertices[v1], vertices[v2]])
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

    return new_vertices, new_faces


# TODO: fix duplicate vertices
def kis_operator_9(vertices, faces):
    new_faces = []
    new_vertices = list(vertices)  # Make a copy of the original vertices
    edge_point_indices = {}
    face_centroid_indices = {}

    for face in faces:
        for i in range(len(face)):
            v1 = face[i]
            v2 = face[(i + 1) % len(face)]

            edge_key_1_3 = tuple([v1, v2]) + (1/3,)
            edge_key_2_3 = tuple([v1, v2]) + (2/3,)
            if edge_key_1_3 not in edge_point_indices:
                point_1_3 = tuple(vertices[v1][j] + 1/3 * (vertices[v2][j] - vertices[v1][j]) for j in range(3))
                point_1_3_idx = len(new_vertices)
                edge_point_indices[edge_key_1_3] = point_1_3_idx
                new_vertices.append(point_1_3)

            if edge_key_2_3 not in edge_point_indices:
                point_2_3 = tuple(vertices[v1][j] + 2/3 * (vertices[v2][j] - vertices[v1][j]) for j in range(3))
                point_2_3_idx = len(new_vertices)
                edge_point_indices[edge_key_2_3] = point_2_3_idx
                new_vertices.append(point_2_3)
        face_centroid = centroid([vertices[i] for i in face])
        face_centroid_idx = len(new_vertices)
        face_centroid_indices[face] = face_centroid_idx
        new_vertices.append(face_centroid)

    for face in faces:
        for i in range(len(face)):
            v1 = face[i]
            v2 = face[(i + 1) % len(face)]

            edge_key_1_3 = tuple([v1, v2]) + (1/3,)
            edge_key_2_3 = tuple([v1, v2]) + (2/3,)
            point_1_3_idx = edge_point_indices[edge_key_1_3]
            point_2_3_idx = edge_point_indices[edge_key_2_3]
            face_centroid_idx = face_centroid_indices[face]

            new_faces.append((point_1_3_idx, point_2_3_idx, face_centroid_idx))

        for i in range(len(face)):
            v1 = face[i]
            v2 = face[(i + 1) % len(face)]
            v3 = face[(i + 2) % len(face)]

            edge_1_key_2_3 = tuple([v1, v2]) + (2/3,)
            edge_2_key_1_3 = tuple([v2, v3]) + (1/3,)
            edge_1_point_2_3_idx = edge_point_indices[edge_1_key_2_3]
            edge_2_point_1_3_idx = edge_point_indices[edge_2_key_1_3]
            face_centroid_idx = face_centroid_indices[face]

            new_faces.append((edge_1_point_2_3_idx, v2, edge_2_point_1_3_idx))
            new_faces.append((edge_1_point_2_3_idx, face_centroid_idx, edge_2_point_1_3_idx))

    return new_vertices, new_faces

def centroid(vertices):
    x, y, z = 0, 0, 0
    for vertex in vertices:
        x += vertex[0]
        y += vertex[1]
        z += vertex[2]
    return (x/len(vertices), y/len(vertices), z/len(vertices))

def normalize(v):
    norm = math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
    return (v[0]/norm, v[1]/norm, v[2]/norm)

def angle(vertex, centroid):
    dx, dy, dz = vertex[0] - centroid[0], vertex[1] - centroid[1], vertex[2] - centroid[2]
    return math.atan2(dy, dx)

def sort_face_vertices_ccw(face, vertices):
    face_vertices = [vertices[i] for i in face]
    face_centroid = centroid(face_vertices)
    sorted_face = sorted(face, key=lambda v_idx: angle(vertices[v_idx], face_centroid))
    return sorted_face

def angle_between_vectors(v1, v2, ref):
    v1_normalized = normalize(v1 - ref)
    v2_normalized = normalize(v2 - ref)
    cosine_angle = np.dot(v1_normalized, v2_normalized)
    angle = np.arccos(np.clip(cosine_angle, -1, 1))
    return angle

def sort_adjacent_faces(vertices, faces, vertex):
    if len(faces) <= 1:
        return faces

    ref_vector = vertices[faces[0]]
    sorted_faces = [faces[0]]
    remaining_faces = faces[1:]

    while remaining_faces:
        face_angles = [(face, angle_between_vectors(ref_vector, vertices[face], vertex)) for face in remaining_faces]
        next_face, _ = min(face_angles, key=lambda x: x[1])
        ref_vector = vertices[next_face]
        sorted_faces.append(next_face)
        remaining_faces.remove(next_face)

    return sorted_faces

def dual_polyhedron(vertices, faces):
    vertices = [np.array(v) for v in vertices]  # Convert input vertices to numpy arrays
    dual_vertices = [centroid([vertices[i] for i in face]) for face in faces]
    vertex_to_adjacent_faces = {i: [f_idx for f_idx, f in enumerate(faces) if i in f] for i in range(len(vertices))}

    dual_faces = []
    for vertex_index, adjacent_faces in vertex_to_adjacent_faces.items():
        sorted_faces = sort_adjacent_faces(dual_vertices, adjacent_faces, vertices[vertex_index])
        dual_faces.append(sorted_faces)

    return dual_vertices, dual_faces

def project_to_sphere(vertices, radius=1):
    projected_vertices = []
    for x, y, z in vertices:
        length = math.sqrt(x**2 + y**2 + z**2)
        x_proj, y_proj, z_proj = radius * x / length, radius * y / length, radius * z / length

        # Convert to latitude and longitude
        latitude = math.degrees(math.asin(z_proj / radius))
        longitude = math.degrees(math.atan2(y_proj, x_proj))

        projected_vertices.append((x_proj, y_proj, z_proj, latitude, longitude))
    return projected_vertices

def print_projected_vertices(projected_vertices):
    print("Index |        X        |        Y        |        Z        |   Latitude   |  Longitude")
    print("------|-----------------|-----------------|-----------------|--------------|------------")
    for idx, vertex in enumerate(projected_vertices):
        print("{:5d} | {: 15.8f} | {: 15.8f} | {: 15.8f} | {: 13.7f} | {: 13.7f}".format(idx, *vertex))

def project_to_sphere2(vertices, radius=2):
    projected_vertices = []
    for x, y, z in vertices:
        length = math.sqrt(x**2 + y**2 + z**2)
        x_proj, y_proj, z_proj = radius * x / length, radius * y / length, radius * z / length

        projected_vertices.append((x_proj, y_proj, z_proj))
    return projected_vertices

vertices, faces  = icosahedron_verts, icosahedron_faces


# 80 Vertex
#vertices, faces = kis_operator(vertices, faces)
#vertices, faces = dual_polyhedron(vertices, faces)

# 180 Vertex
# bug, too many vertices
#vertices, faces = kis_operator_9(vertices, faces)
#vertices, faces = dual_polyhedron(vertices, faces)

# 240 Vertex
vertices, faces = kis_operator(vertices, faces)
vertices, faces = dual_polyhedron(vertices, faces)
vertices, faces = divide_faces(vertices, faces)
vertices, faces = dual_polyhedron(vertices, faces)

# 320 Vertex
# vertices, faces = kis_operator(vertices, faces)
# vertices, faces = kis_operator(vertices, faces)
# vertices, faces = dual_polyhedron(vertices, faces)

print ("Debug: vertices: " +  str(len(vertices)) + " faces: " + str(len(faces)))

print_projected_vertices(project_to_sphere(vertices))
vertices = project_to_sphere2(vertices)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Create a list of polygons for each face
polygons = [[vertices[vertex_idx] for vertex_idx in face] for face in faces]

# Create a Poly3DCollection object
poly3d = Poly3DCollection(polygons, edgecolor='k', lw=1, alpha=0.9)

# Add the polygons to the axes
ax.add_collection3d(poly3d)

# Draw the vertices
x, y, z = zip(*vertices)
ax.scatter(x, y, z, color='r', s=3)

# Set limits and labels
ax.set_xlim([-2, 2])
ax.set_ylim([-2, 2])
ax.set_zlim([-2, 2])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

ax.set_axis_off()
ax.set_box_aspect([1, 1, 1])

# View the plot from different angles
ax.view_init(elev=20, azim=-35)

plt.show()

