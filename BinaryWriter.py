import struct
import numpy as np

class BinaryWriter:
    def __init__(self, file, hex_output=False):
        self.file = file
        self.hex_output = hex_output

    def write(self, value):
        if self.hex_output:
            self._write_hex(value)
        else:
            self._write_binary(value)

    def _write_binary(self, value):
        if isinstance(value, bool):
            self.file.write(struct.pack('<B', value))
        elif isinstance(value, int):
            self.file.write(struct.pack('<i', value))
        elif isinstance(value, float):
            self.file.write(struct.pack('<f', value))
        elif isinstance(value, np.int8):
            self.file.write(struct.pack('<B', value))
        else:
            raise TypeError(f"Unsupported type: {type(value)}")

    def _write_hex(self, value):
        if isinstance(value, bool):
            self.file.write(struct.pack('<B', value).hex())
        elif isinstance(value, int):
            self.file.write(struct.pack('<i', value).hex())
        elif isinstance(value, float):
            self.file.write(struct.pack('<f', value).hex())
        elif isinstance(value, np.int8):
            self.file.write(struct.pack('<B', value).hex())
        else:
            raise TypeError(f"Unsupported type: {type(value)}")

    def close(self):
        self.file.close()
