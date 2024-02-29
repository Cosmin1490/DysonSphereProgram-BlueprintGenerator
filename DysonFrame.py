from dataclasses import dataclass
from typing import Tuple
import numpy as np

@dataclass
class DysonFrame:
    frameId: int
    protoId: int
    reserved: bool
    nodeAid: int
    nodeBid: int
    euler: bool
    spA: int
    spB: int
    spMax: int
    color: Tuple[np.int8, np.int8, np.int8, np.int8]

    def __init__(self, frameId: int, protoId: int, reserved: bool, nodeAid: int, nodeBid: int, euler: bool,
                 spA: int, spB: int, spMax: int, color: Tuple[int, int, int, int]):
        self.frameId = frameId
        self.protoId = protoId
        self.reserved = reserved
        self.nodeAid = nodeAid
        self.nodeBid = nodeBid
        self.euler = euler
        self.spA = spA
        self.spB = spB
        self.spMax = spMax
        self.color = (np.int8(color[0]), np.int8(color[1]), np.int8(color[2]), np.int8(color[3]))

    @classmethod
    def create_with_defaults(cls, frameId: int, nodeAid: int, nodeBid: int):
        default_protoId = 1
        default_reserved = False
        default_euler = False
        default_spA = 0
        default_spB = 0
        default_spMax = 0
        default_color = (0, 0, 0, 0)

        return cls(frameId, default_protoId, default_reserved, nodeAid, nodeBid, default_euler,
                   default_spA, default_spB, default_spMax, default_color)

    def export_as_blueprint(self, w):
        w.write(1)
        w.write(self.frameId)
        w.write(self.protoId)
        w.write(self.reserved)
        w.write(self.nodeAid)
        w.write(self.nodeBid)
        w.write(self.euler)
        w.write(self.spMax)
        w.write(self.color[0])
        w.write(self.color[1])
        w.write(self.color[2])
        w.write(self.color[3])
