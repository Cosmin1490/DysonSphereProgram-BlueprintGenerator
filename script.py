import struct
import datetime
import gzip
import base64
import urllib.parse
from lib.dspbptk.MD5 import DysonSphereMD5
from DysonNode import DysonNode
from DysonFrame import DysonFrame
from DysonShell import DysonShell
from BinaryWriter import BinaryWriter


def from_blueprint_string(bp_string, validate_hash = True):
    if validate_hash:
        index = bp_string.rindex("\"")
        hashed_data = bp_string[:index]
        ref_value = bp_string[index + 1 : ].lower().strip()
        hash_value = DysonSphereMD5(DysonSphereMD5.Variant.MD5F).update(hashed_data.encode("utf-8")).hexdigest()
        if ref_value != hash_value:
            raise ValueError("Blueprint string has invalid has value.")
    assert(bp_string.startswith("DYBP:"))


with open("bp.txt") as f:
    from_blueprint_string(f.read())

with open('bpgen.txt', 'w') as f:
    node = DysonNode.create_with_defaults(1, (0.0, 1.0, 2.0))
    writer = BinaryWriter(f, hex_output = True)
    node.export_as_blueprint(writer)
    writer.close()
with open('bpgen2.txt', 'w') as f:
    node = DysonFrame.create_with_defaults(1, 1, 2)
    writer = BinaryWriter(f, hex_output = True)
    node.export_as_blueprint(writer)
    writer.close()

with open('bpgen3.txt', 'w') as f:
    node = DysonShell.create_with_defaults(1, [1, 2, 3])
    writer = BinaryWriter(f, hex_output = True)
    node.export_as_blueprint(writer)
    writer.close()
