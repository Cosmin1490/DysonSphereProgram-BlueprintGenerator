import struct


class BinaryReader:
    def __init__(self, data: bytes):
        self.data = data
        self.pos = 0

    def read_int(self) -> int:
        value = struct.unpack_from('<i', self.data, self.pos)[0]
        self.pos += 4
        return value

    def read_float(self) -> float:
        value = struct.unpack_from('<f', self.data, self.pos)[0]
        self.pos += 4
        return value

    def read_bool(self) -> bool:
        value = struct.unpack_from('<B', self.data, self.pos)[0]
        self.pos += 1
        return bool(value)

    def read_byte(self) -> int:
        value = struct.unpack_from('<B', self.data, self.pos)[0]
        self.pos += 1
        return value

    @property
    def remaining(self) -> int:
        return len(self.data) - self.pos
