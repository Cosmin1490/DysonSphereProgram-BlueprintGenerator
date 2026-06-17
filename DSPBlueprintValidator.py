import numpy as np

from scipy.spatial import KDTree

from Polyhedron import Polyhedron

class DSPBlueprintValidator:

    # Game constants from decompiled source
    MIN_NODE_DISTANCE_SQ = 0.00511225       # UIDysonBrush_Node.RecalcCollides
    MAX_FRAME_LENGTH = 0.518                 # UIDysonBrush_Frame.CheckCondition
    MIN_NODE_FRAME_DISTANCE_SQ = 0.00275625  # UIDysonBrush_Node.CheckCondition

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

        The game checks crossings on frame segments (subdivided arcs). For full
        node-to-node edges, we verify the actual intersection point of the two
        great circles lies within both arc segments.
        """
        vertices = Polyhedron.project_to_sphere(polyhedron.vertices, 1)
        edges = polyhedron.edges

        for i in range(len(edges)):
            a1 = np.array(vertices[edges[i][0]])
            a2 = np.array(vertices[edges[i][1]])
            normal_a = np.cross(a1, a2)
            norm_na = np.linalg.norm(normal_a)
            if norm_na < 1e-10:
                continue
            normal_a = normal_a / norm_na

            for j in range(i + 1, len(edges)):
                # Skip adjacent edges (they share a vertex)
                if edges[i][0] in edges[j] or edges[i][1] in edges[j]:
                    continue

                b1 = np.array(vertices[edges[j][0]])
                b2 = np.array(vertices[edges[j][1]])
                normal_b = np.cross(b1, b2)
                norm_nb = np.linalg.norm(normal_b)
                if norm_nb < 1e-10:
                    continue
                normal_b = normal_b / norm_nb

                # Intersection line of the two great circle planes
                cross_line = np.cross(normal_a, normal_b)
                norm_cl = np.linalg.norm(cross_line)
                if norm_cl < 1e-10:
                    continue  # Great circles are parallel/identical
                cross_line = cross_line / norm_cl

                # Two candidate intersection points on the sphere
                for candidate in [cross_line, -cross_line]:
                    # Check if candidate lies within arc A (between a1 and a2)
                    # Using: dot(cross(a1, candidate), normal_a) >= 0
                    #    and dot(cross(candidate, a2), normal_a) >= 0
                    in_a = (np.dot(np.cross(a1, candidate), normal_a) >= -1e-9 and
                            np.dot(np.cross(candidate, a2), normal_a) >= -1e-9)
                    if not in_a:
                        continue

                    # Check if candidate lies within arc B (between b1 and b2)
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
        """
        vertices = Polyhedron.project_to_sphere(polyhedron.vertices, 1)
        edges = polyhedron.edges

        for vi in range(len(vertices)):
            point = np.array(vertices[vi])
            for edge in edges:
                # Skip edges that contain this vertex
                if vi == edge[0] or vi == edge[1]:
                    continue
                begin = np.array(vertices[edge[0]])
                end = np.array(vertices[edge[1]])
                sq_dist = DSPBlueprintValidator._point_to_segment_sq(begin, end, point)
                if sq_dist < min_distance_sq:
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
