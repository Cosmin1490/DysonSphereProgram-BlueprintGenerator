from dataclasses import dataclass
from typing import List

from DysonFrame import DysonFrame
from DysonNode import DysonNode
from DysonShell import DysonShell

@dataclass
class DysonSphereLayer:
    _nodes: List[DysonNode]
    _frames: List[DysonFrame]
    _shells: List[DysonShell]

    _paintingGridMode: int

    def __init__(self, nodes: List[DysonNode], frames: List[DysonFrame], shells: List[DysonShell], paintingGridMode: int):
        self._nodes = nodes
        self._frames = frames
        self._shells = shells
        self._paintingGridMode = paintingGridMode

    @classmethod
    def create_with_defaults(cls, nodes: List[DysonNode] = None, frames: List[DysonFrame] = None, shells: List[DysonShell] = None):
        default_paintingGridMode = 0


        return cls(nodes=nodes if nodes is not None else [], frames=frames if frames is not None else [], shells=shells if shells is not None else [], paintingGridMode=default_paintingGridMode)


    def compute_capacity(self, n: int) -> int:
        if n < 64:
            return 64
        else:
            power_of_two = 1
            while (power_of_two * 64) <= n:
                power_of_two *= 2
            return power_of_two * 64

    def export_as_blueprint(self, w):
        w.write(1)
        w.write(self.compute_capacity(len(self._nodes) + 1))
        w.write(len(self._nodes) + 1)
        w.write(0)
        for node in self._nodes:
            node.export_as_blueprint(w)
        w.write(self.compute_capacity(len(self._frames) + 1))
        w.write(len(self._frames) + 1)
        w.write(0)
        for frame in self._frames:
            frame.export_as_blueprint(w)

        w.write(self.compute_capacity(len(self._shells) + 1))
        w.write(len(self._shells) + 1)
        w.write(0)
        for shell in self._shells:
            shell.export_as_blueprint(w)
        w.write(self._paintingGridMode)
        w.write(False)
