import numpy as np

from scipy.spatial import KDTree

from Polyhedron import Polyhedron

class DSPBlueprintValidator:

    # Game constants from decompiled source
    MIN_NODE_DISTANCE_SQ = 0.00511225       # UIDysonBrush_Node.RecalcCollides
    MAX_FRAME_LENGTH = 0.518                 # UIDysonBrush_Frame.CheckCondition
    MIN_NODE_FRAME_DISTANCE_SQ = 0.00275625  # UIDysonBrush_Node.CheckCondition
    MIN_FRAME_FRAME_DISTANCE_SQ = 0.00275625 # UIDysonBrush_Frame.CheckCondition (segment-to-segment)
    MAX_SHELL_CENTROID_DISTANCE_SQ = 0.1609944 # UIDysonBrush_Shell._OnUpdate (0.268324f * 0.6f)

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
    def validate_frame_lengths(polyhedron, max_length=0.518):
        """Check that no edge exceeds the game's maximum frame length.

        Game code: (endn - beginn).magnitude > 0.518f
        """
        vertices = Polyhedron.project_to_sphere(polyhedron.vertices, 1)
        for edge in polyhedron.edges:
            a = np.array(vertices[edge[0]])
            b = np.array(vertices[edge[1]])
            dist = np.linalg.norm(a - b)
            if dist > max_length:
                return False
        return True

    @staticmethod
    def validate_frame_crossings(polyhedron):
        """Check that no two non-adjacent frames (edges) intersect on the sphere.

        Uses KDTree on edge midpoints to skip far-apart pairs.
        """
        vertices = Polyhedron.project_to_sphere(polyhedron.vertices, 1)
        edges = polyhedron.edges
        vertices_arr = np.array(vertices)

        # Precompute edge midpoints, normals, and endpoint arrays
        edge_midpoints = []
        edge_normals = []
        edge_endpoints = []
        max_edge_len = 0.0

        for edge in edges:
            a1 = vertices_arr[edge[0]]
            a2 = vertices_arr[edge[1]]
            mid = (a1 + a2) / 2
            mid_norm = np.linalg.norm(mid)
            if mid_norm > 1e-10:
                mid = mid / mid_norm
            edge_midpoints.append(mid)

            normal = np.cross(a1, a2)
            norm_n = np.linalg.norm(normal)
            if norm_n > 1e-10:
                normal = normal / norm_n
            edge_normals.append(normal)
            edge_endpoints.append((a1, a2))

            edge_len = np.linalg.norm(a1 - a2)
            if edge_len > max_edge_len:
                max_edge_len = edge_len

        # Two edges can only cross if their midpoints are within max_edge_length of each other
        search_radius = max_edge_len * 2
        tree = KDTree(edge_midpoints)

        for i in range(len(edges)):
            nearby = tree.query_ball_point(edge_midpoints[i], search_radius)
            a1, a2 = edge_endpoints[i]
            normal_a = edge_normals[i]
            if np.linalg.norm(np.cross(a1, a2)) < 1e-10:
                continue

            for j in nearby:
                if j <= i:
                    continue
                if edges[i][0] in edges[j] or edges[i][1] in edges[j]:
                    continue

                b1, b2 = edge_endpoints[j]
                normal_b = edge_normals[j]
                if np.linalg.norm(np.cross(b1, b2)) < 1e-10:
                    continue

                cross_line = np.cross(normal_a, normal_b)
                norm_cl = np.linalg.norm(cross_line)
                if norm_cl < 1e-10:
                    continue
                cross_line = cross_line / norm_cl

                for candidate in [cross_line, -cross_line]:
                    in_a = (np.dot(np.cross(a1, candidate), normal_a) >= -1e-9 and
                            np.dot(np.cross(candidate, a2), normal_a) >= -1e-9)
                    if not in_a:
                        continue
                    in_b = (np.dot(np.cross(b1, candidate), normal_b) >= -1e-9 and
                            np.dot(np.cross(candidate, b2), normal_b) >= -1e-9)
                    if in_b:
                        return False

        return True

    @staticmethod
    def _point_to_segment_sq(begin, end, point):
        """Squared distance from a point to a segment on the unit sphere.

        Matches game's PointToSegmentSqr: project point onto segment,
        normalize the projected point, then compute squared distance.
        """
        rhs = end - begin
        sq_mag = np.dot(rhs, rhs)
        if sq_mag < 1e-10:
            return np.sum((point - begin) ** 2)
        t = np.clip(np.dot(point - begin, rhs) / sq_mag, 0.0, 1.0)
        projected = begin + t * rhs
        proj_norm = np.linalg.norm(projected)
        if proj_norm < 1e-10:
            return np.sum(point ** 2)
        projected_normalized = projected / proj_norm
        return np.sum((projected_normalized - point) ** 2)

    @staticmethod
    def validate_node_frame_proximity(polyhedron, min_distance_sq=0.00275625):
        """Check that no node is too close to a non-adjacent frame segment.

        Game code: PointToSegmentSqr(begin, end, point) < 0.00275625f

        Uses KDTree on edge midpoints to skip far-apart node-edge pairs.
        """
        vertices = Polyhedron.project_to_sphere(polyhedron.vertices, 1)
        edges = polyhedron.edges
        vertices_arr = np.array(vertices)

        # Precompute edge midpoints and max edge half-length
        edge_midpoints = []
        max_half_len = 0.0
        for edge in edges:
            a = vertices_arr[edge[0]]
            b = vertices_arr[edge[1]]
            mid = (a + b) / 2
            mid_norm = np.linalg.norm(mid)
            if mid_norm > 1e-10:
                mid = mid / mid_norm
            edge_midpoints.append(mid)
            half_len = np.linalg.norm(a - b) / 2
            if half_len > max_half_len:
                max_half_len = half_len

        tree = KDTree(edge_midpoints)
        # A node can only violate proximity if it's within sqrt(min_distance_sq) + half edge length
        search_radius = np.sqrt(min_distance_sq) + max_half_len

        for vi in range(len(vertices)):
            point = vertices_arr[vi]
            nearby = tree.query_ball_point(point, search_radius)
            for ei in nearby:
                edge = edges[ei]
                if vi == edge[0] or vi == edge[1]:
                    continue
                begin = vertices_arr[edge[0]]
                end = vertices_arr[edge[1]]
                sq_dist = DSPBlueprintValidator._point_to_segment_sq(begin, end, point)
                if sq_dist < min_distance_sq:
                    return False
        return True

    @staticmethod
    def validate_frame_frame_proximity(polyhedron, min_distance_sq=0.00275625):
        """Check that no two non-adjacent frames are too close to each other.

        Game code: For each pair of non-adjacent frames, checks
        PointToSegmentSqr(existing_begin, existing_end, new_point) < 0.00275625f
        for sample points along the new frame against segments of existing frames.

        For non-Euler frames each frame is a single segment [begin, end], so the
        check reduces to testing each endpoint of frame A against the segment of
        frame B (skipping endpoints that are shared nodes).

        Uses KDTree on edge midpoints to skip far-apart pairs.
        """
        vertices = Polyhedron.project_to_sphere(polyhedron.vertices, 1)
        edges = polyhedron.edges
        vertices_arr = np.array(vertices)

        # Precompute edge midpoints and max edge length
        edge_midpoints = []
        max_edge_len = 0.0
        for edge in edges:
            a = vertices_arr[edge[0]]
            b = vertices_arr[edge[1]]
            mid = (a + b) / 2
            mid_norm = np.linalg.norm(mid)
            if mid_norm > 1e-10:
                mid = mid / mid_norm
            edge_midpoints.append(mid)
            edge_len = np.linalg.norm(a - b)
            if edge_len > max_edge_len:
                max_edge_len = edge_len

        tree = KDTree(edge_midpoints)
        # Two edges can only be close if midpoints are within range
        search_radius = max_edge_len + np.sqrt(min_distance_sq)

        for i in range(len(edges)):
            nearby = tree.query_ball_point(edge_midpoints[i], search_radius)
            a1 = vertices_arr[edges[i][0]]
            a2 = vertices_arr[edges[i][1]]

            for j in nearby:
                if j <= i:
                    continue
                # Skip if frames share a node (adjacent)
                shared_nodes = set(edges[i]) & set(edges[j])
                if len(shared_nodes) == 2:
                    continue  # same edge

                b1 = vertices_arr[edges[j][0]]
                b2 = vertices_arr[edges[j][1]]

                # Check endpoints of frame A against segment of frame B
                # Skip endpoint if it's a shared node (game skips these via flag2-flag5)
                if edges[i][0] not in shared_nodes:
                    if DSPBlueprintValidator._point_to_segment_sq(b1, b2, a1) < min_distance_sq:
                        return False
                if edges[i][1] not in shared_nodes:
                    if DSPBlueprintValidator._point_to_segment_sq(b1, b2, a2) < min_distance_sq:
                        return False

                # Check endpoints of frame B against segment of frame A
                if edges[j][0] not in shared_nodes:
                    if DSPBlueprintValidator._point_to_segment_sq(a1, a2, b1) < min_distance_sq:
                        return False
                if edges[j][1] not in shared_nodes:
                    if DSPBlueprintValidator._point_to_segment_sq(a1, a2, b2) < min_distance_sq:
                        return False

        return True

    @staticmethod
    def validate_shell_size(polyhedron, max_distance_sq=0.1609944):
        """Check that no shell face is too large (vertex too far from face centroid).

        Game code (UIDysonBrush_Shell._OnUpdate):
            centroid = Normalize(sum of vertex positions)
            for each vertex: if (centroid - vertex.normalized).sqrMagnitude > 0.268324f * 0.6f -> CycleTooLarge
        """
        vertices = Polyhedron.project_to_sphere(polyhedron.vertices, 1)
        vertices_arr = np.array(vertices)

        for face in polyhedron.faces:
            # Centroid = normalized sum of face vertex positions
            face_vertices = vertices_arr[face]
            centroid = np.sum(face_vertices, axis=0)
            centroid_norm = np.linalg.norm(centroid)
            if centroid_norm < 1e-10:
                return False
            centroid = centroid / centroid_norm

            # Check each vertex distance to centroid
            for vi in face:
                diff = centroid - vertices_arr[vi]
                sq_dist = np.dot(diff, diff)
                if sq_dist > max_distance_sq:
                    return False

        return True

    @staticmethod
    def validate_polyhedron(polyhedron, min_distance=0.00511225):
        return DSPBlueprintValidator.validate_vertices(polyhedron, min_distance)

    @staticmethod
    def validate_all(polyhedron):
        """Run all game-accurate validation checks. Returns a dict of results."""
        results = {
            'node_distance': DSPBlueprintValidator.validate_vertices(polyhedron),
            'frame_length': DSPBlueprintValidator.validate_frame_lengths(polyhedron),
            'frame_crossings': DSPBlueprintValidator.validate_frame_crossings(polyhedron),
            'node_frame_proximity': DSPBlueprintValidator.validate_node_frame_proximity(polyhedron),
            'frame_frame_proximity': DSPBlueprintValidator.validate_frame_frame_proximity(polyhedron),
            'shell_size': DSPBlueprintValidator.validate_shell_size(polyhedron),
        }
        results['all_valid'] = all(results.values())
        return results

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
