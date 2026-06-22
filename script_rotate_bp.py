"""Rotate a Dyson Sphere blueprint to minimize max absolute latitude (max_stress)."""
import base64
import gzip
import io
import math
import numpy as np
import sys
from scipy.optimize import minimize
from scipy.spatial import ConvexHull, Delaunay

from lib.dspbptk.MD5 import DysonSphereMD5
from lib.dspbptk.Tools import DateTimeTools
from BinaryWriter import BinaryWriter
from DysonFrame import DysonFrame
from DysonNode import DysonNode
from DysonShell import DysonShell
from DysonSphereLayer import DysonSphereLayer
from Polyhedron import Polyhedron
from parser.BlueprintParser import BlueprintParser

# --- Parse input blueprint ---
input_file = sys.argv[1] if len(sys.argv) > 1 else "60.txt"
with open(input_file) as f:
    bp_string = f.read().strip()

bp = BlueprintParser.parse(bp_string)
print(f"Parsed: {len(bp.nodes)} nodes, {len(bp.frames)} frames, {len(bp.shells)} shells")
print(f"Original max_stress: {bp.max_stress}")

# Extract positions as numpy array and normalize to unit sphere
positions = np.array([n.pos for n in bp.nodes])  # (N, 3)
norms = np.linalg.norm(positions, axis=1, keepdims=True)
radius = norms.mean()
print(f"Average radius: {radius:.4f}")
positions_unit = positions / norms  # unit sphere for optimization

# --- Find optimal pole axis ---
# We want unit vector n that minimizes max_i |p_i . n|
# Parameterize n by (theta, phi) in spherical coords

def max_abs_z(params, points):
    theta, phi = params
    n = np.array([math.sin(theta) * math.cos(phi),
                  math.sin(theta) * math.sin(phi),
                  math.cos(theta)])
    return np.max(np.abs(points @ n))

# Multi-start optimization
best_val = float('inf')
best_params = None
rng = np.random.default_rng(42)

for _ in range(200):
    theta0 = np.arccos(2 * rng.random() - 1)  # uniform on sphere
    phi0 = 2 * np.pi * rng.random()
    res = minimize(max_abs_z, [theta0, phi0], args=(positions_unit,), method='Nelder-Mead',
                   options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 10000})
    if res.fun < best_val:
        best_val = res.fun
        best_params = res.x

theta, phi = best_params
n_opt = np.array([math.sin(theta) * math.cos(phi),
                  math.sin(theta) * math.sin(phi),
                  math.cos(theta)])

# Current max |y| on unit sphere (game uses Y for latitude)
current_max_y = np.max(np.abs(positions_unit[:, 1]))
current_lat = math.degrees(math.asin(min(current_max_y, 1.0)))
optimal_lat = math.degrees(math.asin(min(best_val, 1.0)))
print(f"\nCurrent max |y|: {current_max_y:.6f} -> latitude: {current_lat:.1f}°")
print(f"Optimal max |y|: {best_val:.6f} -> latitude: {optimal_lat:.1f}°")
print(f"Optimal pole axis: [{n_opt[0]:.6f}, {n_opt[1]:.6f}, {n_opt[2]:.6f}]")

# --- Build rotation matrix: rotate n_opt to y-axis ---
# Game uses Y-component for latitude (not Z)
y_axis = np.array([0.0, 1.0, 0.0])
cross = np.cross(n_opt, y_axis)
cross_norm = np.linalg.norm(cross)

if cross_norm < 1e-10:
    # n_opt is already ~y or ~-y
    if n_opt[1] > 0:
        R = np.eye(3)
    else:
        R = np.diag([-1.0, -1.0, 1.0])  # 180° around z
else:
    k = cross / cross_norm
    cos_a = np.dot(n_opt, z_axis)
    sin_a = cross_norm
    # Rodrigues' formula: R = I + sin(a)*K + (1-cos(a))*K^2
    K = np.array([[0, -k[2], k[1]],
                  [k[2], 0, -k[0]],
                  [-k[1], k[0], 0]])
    R = np.eye(3) + sin_a * K + (1 - cos_a) * (K @ K)

# Apply rotation to original (non-unit) positions
rotated = (R @ positions.T).T  # (N, 3)

# Verify on unit sphere (game uses Y for latitude)
rotated_unit = rotated / np.linalg.norm(rotated, axis=1, keepdims=True)
new_max_y = np.max(np.abs(rotated_unit[:, 1]))
new_lat = math.degrees(math.asin(min(new_max_y, 1.0)))
new_max_stress = min(math.ceil(new_lat), 90)
print(f"\nAfter rotation: max |y|={new_max_y:.6f}, latitude={new_lat:.1f}°, max_stress={new_max_stress}")

# --- Re-export blueprint ---
# Rebuild nodes with rotated positions, keeping all other properties from original
nodes = []
for i, orig_node in enumerate(bp.nodes):
    pos = (float(rotated[i, 0]), float(rotated[i, 1]), float(rotated[i, 2]))
    node = DysonNode(
        nodeId=orig_node.node_id,
        protoId=orig_node.proto_id,
        use=orig_node.use,
        reserved=orig_node.reserved,
        pos=pos,
        spMax=orig_node.sp_max,
        rid=orig_node.rid,
        frameTurn=orig_node.frame_turn,
        shellTurn=orig_node.shell_turn,
        color=orig_node.color,
        spReq=orig_node.sp_req,
        cpReq=orig_node.cp_req,
    )
    nodes.append(node)

# Frames stay the same (reference node IDs, not positions)
frames = []
for orig_frame in bp.frames:
    frame = DysonFrame(
        frameId=orig_frame.frame_id,
        protoId=orig_frame.proto_id,
        reserved=orig_frame.reserved,
        nodeAid=orig_frame.node_a_id,
        nodeBid=orig_frame.node_b_id,
        euler=orig_frame.euler,
        spA=0,
        spB=0,
        spMax=orig_frame.sp_max,
        color=orig_frame.color,
    )
    frames.append(frame)

# Shells stay the same
shells = []
for orig_shell in bp.shells:
    shell = DysonShell(
        shellId=orig_shell.shell_id,
        protoId=orig_shell.proto_id,
        randSeed=orig_shell.rand_seed,
        color=orig_shell.color,
        nodes=orig_shell.nodes,
    )
    shells.append(shell)

# Export
memory_stream = io.BytesIO()
with memory_stream as f:
    layer = DysonSphereLayer.create_with_defaults(nodes, frames, shells)
    writer = BinaryWriter(f)
    writer.write(0)
    layer.export_as_blueprint(writer)

    timestamp = DateTimeTools.csharp_now()
    game_version = bp.game_version

    compressed_content = gzip.compress(memory_stream.getvalue())
    encoded_content = base64.b64encode(compressed_content).decode("utf-8")
    to_hash = "DYBP:0,{},{},1,{}\"{}".format(timestamp, game_version, new_max_stress, encoded_content)
    hash_value = DysonSphereMD5(DysonSphereMD5.Variant.MD5F).update(to_hash.encode("utf-8")).hexdigest()

    formatted_output = "{}\"{}".format(to_hash, hash_value.upper())

    output_file = input_file.replace('.txt', '_rotated.txt')
    with open(output_file, 'w') as out:
        out.write(formatted_output)
    print(f"\nSaved rotated blueprint to {output_file}")
