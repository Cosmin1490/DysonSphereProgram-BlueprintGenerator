import base64
import gzip
import struct
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from urllib.parse import unquote

from .BinaryReader import BinaryReader


@dataclass
class BlueprintArea:
    index: int
    parent_index: int
    tropic_anchor: int
    area_segments: int
    anchor_local_offset_x: int
    anchor_local_offset_y: int
    width: int
    height: int


@dataclass
class BlueprintBuilding:
    index: int
    area_index: int
    local_offset: Tuple[float, float, float]
    local_offset2: Tuple[float, float, float]
    yaw: float
    yaw2: float
    item_id: int
    model_index: int
    output_obj_idx: int
    input_obj_idx: int
    output_to_slot: int
    input_from_slot: int
    output_from_slot: int
    input_to_slot: int
    output_offset: int
    input_offset: int
    recipe_id: int
    filter_id: int
    parameters: List[int]


@dataclass
class FactoryBlueprint:
    # Header
    layout: int
    icons: Tuple[int, int, int, int, int]
    timestamp: int
    game_version: str
    short_desc: str
    desc: str
    md5_hash: str
    # Binary payload
    version: int
    cursor_offset: Tuple[int, int]
    cursor_target_area: int
    drag_box_size: Tuple[int, int]
    primary_area_idx: int
    areas: List[BlueprintArea]
    buildings: List[BlueprintBuilding]


class FactoryBlueprintParser:

    @staticmethod
    def parse(blueprint_string: str) -> FactoryBlueprint:
        header, base64_data, md5_hash = FactoryBlueprintParser._split_blueprint(blueprint_string)
        header_fields = FactoryBlueprintParser._parse_header(header)
        binary_data = FactoryBlueprintParser._decode_data(base64_data)
        reader = BinaryReader(binary_data)

        version = reader.read_int()
        cursor_offset_x = reader.read_int()
        cursor_offset_y = reader.read_int()
        cursor_target_area = reader.read_int()
        drag_box_size_x = reader.read_int()
        drag_box_size_y = reader.read_int()
        primary_area_idx = reader.read_int()

        area_count = reader.read_byte()
        areas = []
        for _ in range(area_count):
            areas.append(FactoryBlueprintParser._read_area(reader))

        building_count = reader.read_int()
        buildings = []
        for _ in range(building_count):
            buildings.append(FactoryBlueprintParser._read_building(reader))

        return FactoryBlueprint(
            layout=header_fields['layout'],
            icons=header_fields['icons'],
            timestamp=header_fields['timestamp'],
            game_version=header_fields['game_version'],
            short_desc=header_fields['short_desc'],
            desc=header_fields['desc'],
            md5_hash=md5_hash,
            version=version,
            cursor_offset=(cursor_offset_x, cursor_offset_y),
            cursor_target_area=cursor_target_area,
            drag_box_size=(drag_box_size_x, drag_box_size_y),
            primary_area_idx=primary_area_idx,
            areas=areas,
            buildings=buildings,
        )

    @staticmethod
    def _split_blueprint(blueprint_string: str):
        first_quote = blueprint_string.index('"')
        last_quote = blueprint_string.rindex('"')
        header = blueprint_string[:first_quote]
        base64_data = blueprint_string[first_quote + 1:last_quote]
        md5_hash = blueprint_string[last_quote + 1:]
        return header, base64_data, md5_hash

    @staticmethod
    def _parse_header(header: str) -> dict:
        # Strip "BLUEPRINT:" prefix
        content = header[len("BLUEPRINT:"):]
        parts = content.split(',')
        # parts: 0=version(0), 1=layout, 2-6=icons, 7=reserved(0), 8=timestamp, 9=gameVersion, 10=shortDesc, 11+=desc
        return {
            'layout': int(parts[1]),
            'icons': (int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5]), int(parts[6])),
            'timestamp': int(parts[8]),
            'game_version': parts[9],
            'short_desc': unquote(parts[10]),
            'desc': unquote(','.join(parts[11:])) if len(parts) > 11 else '',
        }

    @staticmethod
    def _decode_data(base64_data: str) -> bytes:
        compressed = base64.b64decode(base64_data)
        return gzip.decompress(compressed)

    @staticmethod
    def _read_area(reader: BinaryReader) -> BlueprintArea:
        index = struct.unpack_from('<b', reader.data, reader.pos)[0]; reader.pos += 1
        parent_index = struct.unpack_from('<b', reader.data, reader.pos)[0]; reader.pos += 1
        tropic_anchor = struct.unpack_from('<h', reader.data, reader.pos)[0]; reader.pos += 2
        area_segments = struct.unpack_from('<h', reader.data, reader.pos)[0]; reader.pos += 2
        anchor_x = struct.unpack_from('<h', reader.data, reader.pos)[0]; reader.pos += 2
        anchor_y = struct.unpack_from('<h', reader.data, reader.pos)[0]; reader.pos += 2
        width = struct.unpack_from('<h', reader.data, reader.pos)[0]; reader.pos += 2
        height = struct.unpack_from('<h', reader.data, reader.pos)[0]; reader.pos += 2
        return BlueprintArea(
            index=index, parent_index=parent_index,
            tropic_anchor=tropic_anchor, area_segments=area_segments,
            anchor_local_offset_x=anchor_x, anchor_local_offset_y=anchor_y,
            width=width, height=height,
        )

    @staticmethod
    def _read_building(reader: BinaryReader) -> BlueprintBuilding:
        index = reader.read_int()
        area_index = struct.unpack_from('<b', reader.data, reader.pos)[0]; reader.pos += 1
        ox = reader.read_float()
        oy = reader.read_float()
        oz = reader.read_float()
        ox2 = reader.read_float()
        oy2 = reader.read_float()
        oz2 = reader.read_float()
        yaw = reader.read_float()
        yaw2 = reader.read_float()
        item_id = struct.unpack_from('<h', reader.data, reader.pos)[0]; reader.pos += 2
        model_index = struct.unpack_from('<h', reader.data, reader.pos)[0]; reader.pos += 2
        output_obj_idx = reader.read_int()
        input_obj_idx = reader.read_int()
        output_to_slot = struct.unpack_from('<b', reader.data, reader.pos)[0]; reader.pos += 1
        input_from_slot = struct.unpack_from('<b', reader.data, reader.pos)[0]; reader.pos += 1
        output_from_slot = struct.unpack_from('<b', reader.data, reader.pos)[0]; reader.pos += 1
        input_to_slot = struct.unpack_from('<b', reader.data, reader.pos)[0]; reader.pos += 1
        output_offset = struct.unpack_from('<b', reader.data, reader.pos)[0]; reader.pos += 1
        input_offset = struct.unpack_from('<b', reader.data, reader.pos)[0]; reader.pos += 1
        recipe_id = struct.unpack_from('<h', reader.data, reader.pos)[0]; reader.pos += 2
        filter_id = struct.unpack_from('<h', reader.data, reader.pos)[0]; reader.pos += 2
        param_count = struct.unpack_from('<h', reader.data, reader.pos)[0]; reader.pos += 2
        parameters = []
        for _ in range(param_count):
            parameters.append(reader.read_int())
        return BlueprintBuilding(
            index=index, area_index=area_index,
            local_offset=(ox, oy, oz), local_offset2=(ox2, oy2, oz2),
            yaw=yaw, yaw2=yaw2,
            item_id=item_id, model_index=model_index,
            output_obj_idx=output_obj_idx, input_obj_idx=input_obj_idx,
            output_to_slot=output_to_slot, input_from_slot=input_from_slot,
            output_from_slot=output_from_slot, input_to_slot=input_to_slot,
            output_offset=output_offset, input_offset=input_offset,
            recipe_id=recipe_id, filter_id=filter_id,
            parameters=parameters,
        )

    @staticmethod
    def summary(bp: FactoryBlueprint) -> str:
        lines = [
            f"Name: {bp.short_desc}",
            f"Game version: {bp.game_version}",
            f"Icons: {bp.icons}",
            f"Areas: {len(bp.areas)}",
            f"Buildings: {len(bp.buildings)}",
        ]
        # Count buildings by item_id
        from collections import Counter
        counts = Counter(b.item_id for b in bp.buildings)
        lines.append("Building counts by item ID:")
        for item_id, count in counts.most_common():
            lines.append(f"  item {item_id}: {count}")
        return '\n'.join(lines)
