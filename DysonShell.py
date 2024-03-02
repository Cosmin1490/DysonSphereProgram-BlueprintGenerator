from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
import random

@dataclass
class DysonShell:
    shellId: int
    protoId: int
    randSeed: int
    color: Tuple[np.int8, np.int8, np.int8, np.int8]
    nodes: List[int]

    def __init__(self, shellId: int, protoId: int, randSeed: int, color: Tuple[int, int, int, int], nodes: List[int]):
        self.shellId = shellId
        self.protoId = protoId
        self.randSeed = randSeed
        self.color = (np.int8(color[0]), np.int8(color[1]), np.int8(color[2]), np.int8(color[3]))
        self.nodes = nodes

    @classmethod
    def create_with_defaults(cls, shellId: int, nodes: List[int]):
        default_protoId = 0
        default_randSeed = random.randint(0, 2**31 - 1)
        default_color = (0, 0, 0, 0)

        return cls(shellId, default_protoId, default_randSeed, default_color, nodes)


        return cls(shellId, default_protoId, default_randSeed, default_color, nodes)

    def export_as_blueprint(self, w) -> None:
        w.write(2)
        w.write(self.shellId)
        w.write(self.protoId)
        w.write(self.randSeed)
        w.write(self.color[0])
        w.write(self.color[1])
        w.write(self.color[2])
        w.write(self.color[3])
        w.write(len(self.nodes))
        for node in self.nodes:
            w.write(node)
