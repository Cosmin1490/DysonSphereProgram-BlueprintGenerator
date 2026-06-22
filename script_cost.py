"""Calculate SP and CP cost of a Dyson Sphere blueprint at a given orbit radius."""
import math
import sys
import numpy as np

from parser.BlueprintParser import BlueprintParser
from Polyhedron import Polyhedron


def compute_seg_count(pos_a, pos_b, radius):
    """Game formula from DysonFrame.cs:251-262, using float32 to match game.

    Game imports blueprint nodes as: pos = pos.normalized * orbitRadius
    Then: segCount = int((acos(dot(nodeA.pos.normalized, nodeB.pos.normalized)) * nodeA.pos.magnitude / 600) + 0.5) * 2)
    Since pos.magnitude == orbitRadius after import, we use radius directly.
    """
    a = np.array(pos_a, dtype=np.float32)
    b = np.array(pos_b, dtype=np.float32)
    a_norm = a / np.float32(np.linalg.norm(a))
    b_norm = b / np.float32(np.linalg.norm(b))
    dot = np.float32(np.clip(np.dot(a_norm, b_norm), -1.0, 1.0))
    angle = np.float32(math.acos(float(dot)))
    seg_count = int(float(angle * np.float32(radius) / np.float32(600.0) + np.float32(0.5))) * 2
    if seg_count <= 0:
        seg_count = 2
    return seg_count


def compute_sp_cost(polyhedron, radius):
    """Compute total Structure Point cost (rockets).

    From decompiled game code:
    - Each node: 30 SP
    - Each frame: segCount * 10 SP
    """
    unit_verts = Polyhedron.project_to_sphere(polyhedron.vertices, radius)

    node_sp = len(polyhedron.vertices) * 30

    frame_sp = 0
    seg_counts = []
    for a_idx, b_idx in polyhedron.edges:
        sc = compute_seg_count(unit_verts[a_idx], unit_verts[b_idx], radius)
        seg_counts.append(sc)
        frame_sp += sc * 10

    return node_sp, frame_sp, seg_counts


def compute_cp_cost(polyhedron, radius):
    """Estimate Cell Point cost (solar sails).

    From decompiled DysonShell.cs:486-489:
    - gridScale = max(1, round((radius/4000)^0.75 + 0.5))
    - cpPerVertex = gridScale^2 * 2
    - cellPointMax = vertexCount * cpPerVertex

    vertexCount = number of triangular grid points filling the face.
    We estimate this from face solid angle and grid density.
    """
    grid_scale = max(1, round(math.pow(radius / 4000.0, 0.75) + 0.5))
    grid_size = grid_scale * 80.0
    cp_per_vertex = grid_scale * grid_scale * 2

    # Estimate vertex count per face from its solid angle
    # The triangular grid has spacing grid_size, so area per triangle ≈ (grid_size^2 * sqrt(3)/4)
    # On a sphere of given radius, face area = solid_angle * radius^2
    # vertex_count ≈ face_area / (grid_size^2 * sqrt(3)/4) * 0.5 (roughly 0.5 vertices per small triangle)
    # More precisely: vertex density ≈ 2 / (sqrt(3) * grid_size^2) per unit area
    vertex_density = 2.0 / (math.sqrt(3) * grid_size * grid_size)

    unit_verts = np.array(Polyhedron.project_to_sphere(polyhedron.vertices, 1))
    total_vertex_count = 0

    for face in polyhedron.faces:
        # Compute solid angle of face using spherical excess
        face_verts = unit_verts[[*face]]
        n = len(face_verts)
        # Sum of interior angles on the sphere
        angle_sum = 0.0
        for i in range(n):
            a = face_verts[(i - 1) % n]
            b = face_verts[i]
            c = face_verts[(i + 1) % n]
            # Tangent vectors at b along edges ba and bc
            ba = a - b * np.dot(a, b)
            bc = c - b * np.dot(c, b)
            ba_norm = np.linalg.norm(ba)
            bc_norm = np.linalg.norm(bc)
            if ba_norm > 1e-12 and bc_norm > 1e-12:
                cos_angle = np.clip(np.dot(ba, bc) / (ba_norm * bc_norm), -1, 1)
                angle_sum += math.acos(cos_angle)

        solid_angle = angle_sum - (n - 2) * math.pi  # spherical excess
        face_area = solid_angle * radius * radius
        vert_count = int(face_area * vertex_density + 0.5)
        total_vertex_count += max(vert_count, 1)

    total_cp = total_vertex_count * cp_per_vertex
    return total_cp, grid_scale, cp_per_vertex, total_vertex_count


def main():
    if len(sys.argv) < 2:
        print("Usage: python script_cost.py <blueprint_file> [radius=10000]")
        sys.exit(1)

    arg = sys.argv[1]
    radius = float(sys.argv[2]) if len(sys.argv) > 2 else 10000.0

    try:
        with open(arg) as f:
            bp_string = f.read().strip()
    except (FileNotFoundError, OSError):
        bp_string = arg.strip()

    bp = BlueprintParser.parse(bp_string)
    poly = BlueprintParser.to_polyhedron(bp)

    print(f"Blueprint: {arg}")
    print(f"  Nodes: {len(bp.nodes)}, Frames: {len(bp.frames)}, Shells: {len(bp.shells)}")
    print(f"  Orbit radius: {radius:.0f}")
    print()

    # SP cost
    node_sp, frame_sp, seg_counts = compute_sp_cost(poly, radius)
    total_sp = node_sp + frame_sp
    print(f"=== Structure Points (Rockets) ===")
    print(f"  Node SP:  {node_sp:>8} ({len(poly.vertices)} nodes x 30)")
    print(f"  Frame SP: {frame_sp:>8} ({len(poly.edges)} frames, {sum(seg_counts)} total segments)")
    print(f"  Total SP: {total_sp:>8}")
    print(f"  Avg seg/frame: {np.mean(seg_counts):.1f}, min: {min(seg_counts)}, max: {max(seg_counts)}")
    print()

    # CP cost (estimate)
    total_cp, grid_scale, cp_per_vertex, total_verts = compute_cp_cost(poly, radius)
    print(f"=== Cell Points (Solar Sails) — estimate ===")
    print(f"  Grid scale: {grid_scale}, CP per vertex: {cp_per_vertex}")
    print(f"  Est. total grid vertices: {total_verts}")
    print(f"  Est. total CP: {total_cp:>8}")
    print()

    print(f"=== Summary ===")
    print(f"  Total SP (rockets): {total_sp}")
    print(f"  Est. CP (sails):    {total_cp}")


if __name__ == "__main__":
    main()
