import struct
import datetime
import gzip
import base64
import urllib.parse
from lib.dspbptk.MD5 import DysonSphereMD5
from lib.dspbptk.Tools import DateTimeTools
import io
from BinaryWriter import BinaryWriter
from DysonFrame import DysonFrame
from DysonNode import DysonNode
from DysonShell import DysonShell
from DysonSphereLayer import DysonSphereLayer
from Polyhedron import Polyhedron

polyhedron = Polyhedron.create_icosahedron()
polyhedron.coxeter_operator()
polyhedron.dual_operator()
polyhedron.tessellate_edges(3)

#with open("poly.obj", "r") as file:
#    file_content = file.read()
#
#polyhedron = Polyhedron.create_from_polyhedronisme_obj_file(file_content)


#polyhedron.plot_polyhedron()

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
