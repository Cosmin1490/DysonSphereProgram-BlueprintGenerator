import base64
import gzip
from dataclasses import dataclass, field
from typing import List, Tuple

from .BinaryReader import BinaryReader


@dataclass
class ParsedNode:
    node_id: int
    proto_id: int
    use: bool
    reserved: bool
    pos: Tuple[float, float, float]
    sp_max: int
    rid: int
    frame_turn: int
    shell_turn: int
    color: Tuple[int, int, int, int]
    sp_req: int
    cp_req: int


@dataclass
class ParsedFrame:
    frame_id: int
    proto_id: int
    reserved: bool
    node_a_id: int
    node_b_id: int
    euler: bool
    sp_max: int
    color: Tuple[int, int, int, int]


@dataclass
class ParsedShell:
    shell_id: int
    proto_id: int
    rand_seed: int
    color: Tuple[int, int, int, int]
    nodes: List[int]


@dataclass
class ParsedBlueprint:
    timestamp: str
    game_version: str
    max_stress: str
    hash: str
    nodes: List[ParsedNode] = field(default_factory=list)
    frames: List[ParsedFrame] = field(default_factory=list)
    shells: List[ParsedShell] = field(default_factory=list)
    painting_grid_mode: int = 0


class BlueprintParser:

    @staticmethod
    def parse(blueprint_string: str) -> ParsedBlueprint:
        header_and_data, hash_value = BlueprintParser._split_blueprint(blueprint_string)
        timestamp, game_version, max_stress, encoded_data = BlueprintParser._parse_header(header_and_data)
        raw_bytes = BlueprintParser._decode_data(encoded_data)
        reader = BinaryReader(raw_bytes)

        bp = ParsedBlueprint(
            timestamp=timestamp,
            game_version=game_version,
            max_stress=max_stress,
            hash=hash_value,
        )

        # Initial marker
        reader.read_int()  # 0

        # Layer version
        reader.read_int()  # 1

        # Nodes
        _node_capacity = reader.read_int()
        node_count = reader.read_int()
        _node_reserved = reader.read_int()  # 0
        for _ in range(node_count - 1):  # first slot is padding
            _pool_id = reader.read_int()
            bp.nodes.append(BlueprintParser._read_node(reader))

        # Frames
        _frame_capacity = reader.read_int()
        frame_count = reader.read_int()
        _frame_reserved = reader.read_int()  # 0
        for _ in range(frame_count - 1):
            _pool_id = reader.read_int()
            bp.frames.append(BlueprintParser._read_frame(reader))

        # Shells
        _shell_capacity = reader.read_int()
        shell_count = reader.read_int()
        _shell_reserved = reader.read_int()  # 0
        for _ in range(shell_count - 1):
            _pool_id = reader.read_int()
            bp.shells.append(BlueprintParser._read_shell(reader))

        bp.painting_grid_mode = reader.read_int()
        # trailing bool
        if reader.remaining > 0:
            reader.read_bool()

        return bp

    @staticmethod
    def _split_blueprint(blueprint_string: str):
        # Format: DYBP:0,timestamp,version,1,max_stress"base64data"hash
        # Split from the right on " to get the hash
        last_quote = blueprint_string.rfind('"')
        second_last_quote = blueprint_string.rfind('"', 0, last_quote)

        hash_value = blueprint_string[last_quote + 1:]
        encoded_data_and_header = blueprint_string[:last_quote]

        return encoded_data_and_header, hash_value

    @staticmethod
    def _parse_header(header_and_data: str):
        # Format: DYBP:0,timestamp,version,1,max_stress"base64data
        # Split on first " to separate header from data
        quote_pos = header_and_data.find('"')
        header = header_and_data[:quote_pos]
        encoded_data = header_and_data[quote_pos + 1:]

        # header = DYBP:0,timestamp,version,1,max_stress
        parts = header.split(',')
        # parts[0] = "DYBP:0", parts[1] = timestamp, parts[2] = version,
        # parts[3] = "1", parts[4] = max_stress
        timestamp = parts[1]
        game_version = parts[2]
        max_stress = parts[4]

        return timestamp, game_version, max_stress, encoded_data

    @staticmethod
    def _decode_data(encoded_data: str) -> bytes:
        compressed = base64.b64decode(encoded_data)
        return gzip.decompress(compressed)

    @staticmethod
    def _read_node(reader: BinaryReader) -> ParsedNode:
        version = reader.read_int()  # 5
        node_id = reader.read_int()
        proto_id = reader.read_int()
        use = reader.read_bool()
        reserved = reader.read_bool()
        x = reader.read_float()
        y = reader.read_float()
        z = reader.read_float()
        sp_max = reader.read_int()
        rid = reader.read_int()
        frame_turn = reader.read_int()
        shell_turn = reader.read_int()
        sp_req = reader.read_int()
        cp_req = reader.read_int()
        r = reader.read_byte()
        g = reader.read_byte()
        b = reader.read_byte()
        a = reader.read_byte()
        return ParsedNode(
            node_id=node_id, proto_id=proto_id, use=use, reserved=reserved,
            pos=(x, y, z), sp_max=sp_max, rid=rid,
            frame_turn=frame_turn, shell_turn=shell_turn,
            color=(r, g, b, a), sp_req=sp_req, cp_req=cp_req,
        )

    @staticmethod
    def _read_frame(reader: BinaryReader) -> ParsedFrame:
        version = reader.read_int()  # 1
        frame_id = reader.read_int()
        proto_id = reader.read_int()
        reserved = reader.read_bool()
        node_a_id = reader.read_int()
        node_b_id = reader.read_int()
        euler = reader.read_bool()
        sp_max = reader.read_int()
        r = reader.read_byte()
        g = reader.read_byte()
        b = reader.read_byte()
        a = reader.read_byte()
        return ParsedFrame(
            frame_id=frame_id, proto_id=proto_id, reserved=reserved,
            node_a_id=node_a_id, node_b_id=node_b_id, euler=euler,
            sp_max=sp_max, color=(r, g, b, a),
        )

    @staticmethod
    def _read_shell(reader: BinaryReader) -> ParsedShell:
        version = reader.read_int()  # 2
        shell_id = reader.read_int()
        proto_id = reader.read_int()
        rand_seed = reader.read_int()
        r = reader.read_byte()
        g = reader.read_byte()
        b = reader.read_byte()
        a = reader.read_byte()
        node_count = reader.read_int()
        nodes = [reader.read_int() for _ in range(node_count)]
        return ParsedShell(
            shell_id=shell_id, proto_id=proto_id, rand_seed=rand_seed,
            color=(r, g, b, a), nodes=nodes,
        )

    @staticmethod
    def to_polyhedron(bp: ParsedBlueprint):
        """Convert parsed blueprint to a Polyhedron for validation."""
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from Polyhedron import Polyhedron

        # Build vertex list from nodes, indexed by node_id
        node_id_to_index = {}
        vertices = []
        for i, node in enumerate(bp.nodes):
            node_id_to_index[node.node_id] = i
            vertices.append(list(node.pos))

        # Build faces from shells, remapping node IDs to 0-based indices
        faces = []
        for shell in bp.shells:
            face = [node_id_to_index[nid] for nid in shell.nodes]
            faces.append(face)

        return Polyhedron(vertices, faces)

    @staticmethod
    def summary(bp: ParsedBlueprint) -> str:
        return (
            f"Game version: {bp.game_version}\n"
            f"Nodes: {len(bp.nodes)}, Frames: {len(bp.frames)}, Shells: {len(bp.shells)}\n"
            f"Max stress: {bp.max_stress}"
        )
