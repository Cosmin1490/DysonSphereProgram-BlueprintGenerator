import struct
import datetime
import gzip
import base64
import urllib.parse
from lib.dspbptk.MD5 import DysonSphereMD5
import io
from BinaryWriter import BinaryWriter
from DysonFrame import DysonFrame
from DysonNode import DysonNode
from DysonShell import DysonShell
from DysonSphereLayer import DysonSphereLayer
from Polyhedron import Polyhedron

icosahedron = Polyhedron.create_icosahedron()
icosahedron.coxeter_operator()
icosahedron.coxeter_operator()
icosahedron.dual_operator()

nodes = []
frames = []
shells = []
for index, vertex in enumerate(icosahedron.vertices):
    nodes.append(DysonNode.create_with_defaults(index + 1, vertex))

for index, edge in enumerate(icosahedron.edges):
    frames.append(DysonFrame.create_with_defaults(index + 1, edge[0] + 1, edge[1] + 1))

for index, face in enumerate(icosahedron.faces):
    incremented_face = [vertex + 1 for vertex in face]
    shells.append(DysonShell.create_with_defaults(index + 1, incremented_face))

memory_stream = io.BytesIO()

with memory_stream as f:
    node = DysonSphereLayer.create_with_defaults(nodes, frames, shells)
    writer = BinaryWriter(f)
    writer.write(0)
    node.export_as_blueprint(writer)
    memory_stream_content = memory_stream.getvalue()

    compressed_content = gzip.compress(memory_stream_content)
    encoded_content = base64.b64encode(compressed_content).decode("utf-8")
    to_hash = "DYBP:0,638446503868746433,0.10.29.21950,1,90\"" + encoded_content
    hash_value = DysonSphereMD5(DysonSphereMD5.Variant.MD5F).update(to_hash.encode("utf-8")).hexdigest()
    print(to_hash + "\"" + hash_value.upper())
