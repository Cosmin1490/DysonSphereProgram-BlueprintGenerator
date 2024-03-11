import base64
import datetime
import gzip
import io
import numpy as np
import struct

from lib.dspbptk.MD5 import DysonSphereMD5
from lib.dspbptk.Tools import DateTimeTools
from scipy.spatial import ConvexHull, Delaunay

from BinaryWriter import BinaryWriter
from DSPBlueprintValidator import DSPBlueprintValidator
from DysonFrame import DysonFrame
from DysonNode import DysonNode
from DysonShell import DysonShell
from DysonSphereLayer import DysonSphereLayer
from Polyhedron import Polyhedron

#points  = np.random.rand(42, 3)
#from Optimizer import Optimizer
#optimizer = Optimizer(points, num_epochs=30000)
#optimizer.optimize()
#points = optimizer.get_updated_points()

#points  = np.random.rand(642, 3)
#points  = np.random.rand(2647, 3)
points  = np.random.rand(2648, 3)

#from Optimizer import Optimizer
from OptimizerAntipodal import Optimizer
optimizer = Optimizer(points, num_epochs=50000)
optimizer.optimize()
points = optimizer.get_updated_points()

Polyhedron([point.tolist() for point in points], []).plot_polyhedron()
from SphereOptimizer import SphereOptimizer
optimizer = SphereOptimizer(points, num_epochs=5000000)
optimizer.optimize()
points = optimizer.get_updated_points()

points = [point.tolist() for point in points]

delaunay = Delaunay(points)

# Get the faces by creating the convex hull of the Delaunay triangulation
hull = ConvexHull(delaunay.points)

# Deduplicate faces using a dictionary
face_dict = {}
for face in hull.simplices:
    sorted_face_tuple = tuple(sorted(face))
    if sorted_face_tuple not in face_dict:
        face_dict[sorted_face_tuple] = face

unique_faces = list(face_dict.values())
unique_faces = [face.tolist() for face in unique_faces]


# Create the Polyhedron using the original vertices and unique_faces
polyhedron = Polyhedron(points, unique_faces)
#polyhedron = Polyhedron(points, [])
#polyhedron.dual_operator()

polyhedron.plot_polyhedron()
polyhedron = DSPBlueprintValidator.correct_polyhedron(polyhedron)
polyhedron.plot_polyhedron()

if not DSPBlueprintValidator.validate_polyhedron(polyhedron):
    print("The polyhedron cannot be created within the game.")
    exit(1)

nodes = []
frames = []
shells = []
max_stress = polyhedron.absolute_latitude()

for index, vertex in enumerate(polyhedron.vertices):
    nodes.append(DysonNode.create_with_defaults(index + 1, vertex))

for index, edge in enumerate(polyhedron.edges):
    frames.append(DysonFrame.create_with_defaults(index + 1, edge[0] + 1, edge[1] + 1))

for index, face in enumerate(polyhedron.faces):
    incremented_face = [vertex + 1 for vertex in face]
    shells.append(DysonShell.create_with_defaults(index + 1, incremented_face))

memory_stream = io.BytesIO()
with memory_stream as f:
    node = DysonSphereLayer.create_with_defaults(nodes, frames, shells)
    writer = BinaryWriter(f)
    writer.write(0)
    node.export_as_blueprint(writer)

    timestamp = DateTimeTools.csharp_now()
    game_version = "0.10.29.21950"

    compressed_content = gzip.compress(memory_stream.getvalue())
    encoded_content = base64.b64encode(compressed_content).decode("utf-8")
    to_hash = "DYBP:0,{},{},1,{}\"{}".format(timestamp, game_version, max_stress, encoded_content)
    hash_value = DysonSphereMD5(DysonSphereMD5.Variant.MD5F).update(to_hash.encode("utf-8")).hexdigest()

    formatted_output = "{}\"{}".format(to_hash, hash_value.upper())
    print(formatted_output)
