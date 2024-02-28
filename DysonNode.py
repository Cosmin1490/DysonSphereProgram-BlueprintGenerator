import numpy as np
from dataclasses import dataclass
from typing import Tuple

@dataclass
class DysonNode:
    nodeId: int
    protoId: int
    use: bool
    reserved: bool
    pos: Tuple[float, float, float]
    spMax: int
    rid: int
    frameTurn: int
    shellTurn: int
    color: Tuple[np.int8, np.int8, np.int8, np.int8]
    spReq: int
    cpReq: int

    def __init__(self, nodeId: int, protoId: int, use: bool, reserved: bool, pos: Tuple[float, float, float], spMax: int,
                 rid: int, frameTurn: int, shellTurn: int, color: Tuple[int, int, int, int], spReq: int, cpReq: int):
        self.nodeId = nodeId
        self.protoId = protoId
        self.use = use
        self.reserved = reserved
        self.pos = pos
        self.spMax = spMax
        self.rid = rid
        self.frameTurn = frameTurn
        self.shellTurn = shellTurn
        self.color = (np.int8(color[0]), np.int8(color[1]), np.int8(color[2]), np.int8(color[3]))
        self.spReq = spReq
        self.cpReq = cpReq

    @classmethod
    def create_with_defaults(cls, nodeId: int, pos: Tuple[float, float, float]):
        default_protoId = 0
        default_use = False
        default_reserved = False
        default_spMax = 30
        default_rid = nodeId
        default_frameTurn = 0
        default_shellTurn = 0
        default_color = (0, 0, 0, 0)
        default_spReq = 30
        default_cpReq = 0

        return cls(nodeId, default_protoId, default_use, default_reserved, pos, default_spMax, default_rid,
                   default_frameTurn, default_shellTurn, default_color, default_spReq, default_cpReq)

    def export_as_blueprint(self, w):
        w.write(5)
        w.write(self.nodeId)
        w.write(self.protoId)
        w.write(self.use)
        w.write(self.reserved)
        w.write(self.pos[0])
        w.write(self.pos[1])
        w.write(self.pos[2])
        w.write(self.spMax)
        w.write(self.rid)
        w.write(self.frameTurn)
        w.write(self.shellTurn)
        w.write(self.spReq)
        w.write(self.cpReq)
        w.write(self.color[0])
        w.write(self.color[1])
        w.write(self.color[2])
        w.write(self.color[3])
