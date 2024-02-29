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


def from_blueprint_string(bp_string, validate_hash = True):
    if validate_hash:
        index = bp_string.rindex("\"")
        hashed_data = bp_string[:index]
        ref_value = bp_string[index + 1 : ].lower().strip()
        hash_value = DysonSphereMD5(DysonSphereMD5.Variant.MD5F).update(hashed_data.encode("utf-8")).hexdigest()
        if ref_value != hash_value:
            raise ValueError("Blueprint string has invalid hash value.")
    assert(bp_string.startswith("DYBP:"))


with open("bp.txt") as f:
    from_blueprint_string(f.read())

with open("bp2.txt") as f:
    from_blueprint_string(f.read())

#with open('bpgen.txt', 'w') as f:
#    node = DysonNode.create_with_defaults(1, (0.0, 1.0, 2.0))
#    writer = BinaryWriter(f, hex_output = True)
#    node.export_as_blueprint(writer)
#    writer.close()
#with open('bpgen2.txt', 'w') as f:
#    node = DysonFrame.create_with_defaults(1, 1, 2)
#    writer = BinaryWriter(f, hex_output = True)
#    node.export_as_blueprint(writer)
#    writer.close()
#
#with open('bpgen3.txt', 'w') as f:
#    node = DysonShell.create_with_defaults(1, [1, 2, 3])
#    writer = BinaryWriter(f, hex_output = True)
#    node.export_as_blueprint(writer)
#    writer.close()
#
#with open('bpgen4.txt', 'w') as f:
#    node = DysonSphereLayer.create_with_defaults([DysonNode.create_with_defaults(1, (0.0, 1.0, 2.0)), DysonNode.create_with_defaults(2, (5.0, 1.0, 2.0)), DysonNode.create_with_defaults(3, (10.0, 1.0, 2.0))], [], [])
#    writer = BinaryWriter(f, hex_output = True)
#    node.export_as_blueprint(writer)
#    writer.close()


memory_stream = io.BytesIO()

with memory_stream as f:
    node = DysonSphereLayer.create_with_defaults([DysonNode.create_with_defaults(1, (0.0, 1.0, 0.0)), DysonNode.create_with_defaults(2, (1.0, 0.0, 0.0)), DysonNode.create_with_defaults(3, (0.0, 0.0, 1.0))], [], [])
    writer = BinaryWriter(f)
    writer.write(0)
    node.export_as_blueprint(writer)
    memory_stream_content = memory_stream.getvalue()

    compressed_content = gzip.compress(memory_stream_content)
    encoded_content = base64.b64encode(compressed_content).decode("utf-8")
    writer.close()
    to_hash = "DYBP:0,638446503868746433,0.10.29.21950,1,90\"" + encoded_content
    hash_value = DysonSphereMD5(DysonSphereMD5.Variant.MD5F).update(to_hash.encode("utf-8")).hexdigest().upper()
    print(to_hash + "\"" + hash_value)


# compressed_data = gzip.compress(self._data)
# b64_data = base64.b64encode(compressed_data).decode("utf-8")

#DYBP:0,638446503868746433,0.10.29.21950,1,90" + b64_data + """ + hash_value


