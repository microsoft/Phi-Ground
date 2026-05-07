from __future__ import annotations

import ast
import re
from abc import ABC, abstractmethod
from typing import Optional

from cuactspot.registry import PARSER_REGISTRY
from cuactspot.types import BackendResponse, Coordinate, DatasetSample, GenerationRequest


class BaseCoordinateParser(ABC):
    @abstractmethod
    def parse(
        self,
        text: str,
        sample: Optional[DatasetSample] = None,
        request: Optional[GenerationRequest] = None,
        response: Optional[BackendResponse] = None,
    ) -> list[Coordinate]:
        raise NotImplementedError


@PARSER_REGISTRY.register("regex_coordinate_parser")
class RegexCoordinateParser(BaseCoordinateParser):
    def __init__(self, pattern: Optional[str] = None) -> None:
        self.pattern = re.compile(
            pattern
            or r"[\[(]\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*[\])]"
        )

    def parse(
        self,
        text: str,
        sample: Optional[DatasetSample] = None,
        request: Optional[GenerationRequest] = None,
        response: Optional[BackendResponse] = None,
    ) -> list[Coordinate]:
        matches = self.pattern.findall(text)
        return [(float(x), float(y)) for x, y in matches]


@PARSER_REGISTRY.register("action_coordinate_parser")
class ActionCoordinateParser(BaseCoordinateParser):
    def __init__(self) -> None:
        self.named_pair_patterns = [
            re.compile(
                r"(?:x|x1|start_x)\s*=\s*(-?\d+(?:\.\d+)?)\s*,\s*(?:y|y1|start_y)\s*=\s*(-?\d+(?:\.\d+)?)",
                re.IGNORECASE,
            ),
            re.compile(
                r"(?:x2|end_x)\s*=\s*(-?\d+(?:\.\d+)?)\s*,\s*(?:y2|end_y)\s*=\s*(-?\d+(?:\.\d+)?)",
                re.IGNORECASE,
            ),
        ]
        self.positional_pair_pattern = re.compile(
            r"\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)"
        )

    def parse(
        self,
        text: str,
        sample: Optional[DatasetSample] = None,
        request: Optional[GenerationRequest] = None,
        response: Optional[BackendResponse] = None,
    ) -> list[Coordinate]:
        matches: list[tuple[int, Coordinate]] = []
        for pattern in self.named_pair_patterns:
            for match in pattern.finditer(text):
                matches.append(
                    (match.start(), (float(match.group(1)), float(match.group(2))))
                )

        if matches:
            matches.sort(key=lambda item: item[0])
            return [coordinate for _, coordinate in matches]

        return [
            (float(x), float(y))
            for x, y in self.positional_pair_pattern.findall(text)
        ]


@PARSER_REGISTRY.register("phi_ground_coordinate_parser")
class PhiGroundCoordinateParser(BaseCoordinateParser):
    def __init__(self) -> None:
        self.tag_pattern = re.compile(
            r"<(point|bbox|box)>\s*([^<]+?)\s*</(?:point|bbox|box)>",
            re.IGNORECASE,
        )
        self.number_pattern = re.compile(r"-?\d+(?:\.\d+)?")
        self.fallback_pair_pattern = re.compile(
            r"(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)"
        )

    def parse(
        self,
        text: str,
        sample: Optional[DatasetSample] = None,
        request: Optional[GenerationRequest] = None,
        response: Optional[BackendResponse] = None,
    ) -> list[Coordinate]:
        coordinates: list[Coordinate] = []
        for _, content in self.tag_pattern.findall(text):
            numbers = [float(item) for item in self.number_pattern.findall(content)]
            if len(numbers) >= 4:
                coordinates.append(
                    ((numbers[0] + numbers[2]) / 2.0, (numbers[1] + numbers[3]) / 2.0)
                )
            elif len(numbers) >= 2:
                coordinates.append((numbers[0], numbers[1]))

        if coordinates:
            return coordinates

        fallback_matches = self.fallback_pair_pattern.findall(text)
        return [(float(x), float(y)) for x, y in fallback_matches]


@PARSER_REGISTRY.register("ui_tars_coordinate_parser")
class UITarsCoordinateParser(BaseCoordinateParser):
    """Parser for UI-TARS model output that only extracts coordinates from Action lines.

    Handles both ``start_box='(x,y)'`` and ``<point>x y</point>`` formats.
    """

    def __init__(self) -> None:
        self.box_pattern = re.compile(
            r"(?:start_box|end_box|start_point|end_point|point)\s*=\s*['\"]?"
            r"\(?(?:<point>)?\s*(-?\d+(?:\.\d+)?)[,\s]+(-?\d+(?:\.\d+)?)\s*(?:</point>)?\)?['\"]?",
        )

    def parse(
        self,
        text: str,
        sample: Optional[DatasetSample] = None,
        request: Optional[GenerationRequest] = None,
        response: Optional[BackendResponse] = None,
    ) -> list[Coordinate]:
        action_text = text
        action_idx = text.find("Action:")
        if action_idx != -1:
            action_text = text[action_idx:]
        matches = self.box_pattern.findall(action_text)
        return [(float(x), float(y)) for x, y in matches]


@PARSER_REGISTRY.register("bbox_center_coordinate_parser")
class BboxCenterCoordinateParser(BaseCoordinateParser):
    """Parser for ``[x1,y1,x2,y2]`` bbox output, returns the center point."""

    def __init__(self) -> None:
        self.bbox_pattern = re.compile(
            r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,"
            r"\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]"
        )

    def parse(
        self,
        text: str,
        sample: Optional[DatasetSample] = None,
        request: Optional[GenerationRequest] = None,
        response: Optional[BackendResponse] = None,
    ) -> list[Coordinate]:
        coordinates: list[Coordinate] = []
        for match in self.bbox_pattern.finditer(text):
            x1, y1, x2, y2 = (float(match.group(i)) for i in range(1, 5))
            coordinates.append(((x1 + x2) / 2.0, (y1 + y2) / 2.0))
        return coordinates


@PARSER_REGISTRY.register("xy_tag_coordinate_parser")
class XYTagCoordinateParser(BaseCoordinateParser):
    """Parser for ``<x>value</x>`` / ``<y>value</y>`` tag pairs."""

    def __init__(self) -> None:
        self.x_pattern = re.compile(r"<x>\s*(-?\d+(?:\.\d+)?)\s*</x>")
        self.y_pattern = re.compile(r"<y>\s*(-?\d+(?:\.\d+)?)\s*</y>")

    def parse(
        self,
        text: str,
        sample: Optional[DatasetSample] = None,
        request: Optional[GenerationRequest] = None,
        response: Optional[BackendResponse] = None,
    ) -> list[Coordinate]:
        x_values = [float(m) for m in self.x_pattern.findall(text)]
        y_values = [float(m) for m in self.y_pattern.findall(text)]
        count = min(len(x_values), len(y_values))
        return [(x_values[i], y_values[i]) for i in range(count)]


@PARSER_REGISTRY.register("mai_ui_grounding_parser")
class MAIUIGroundingParser(BaseCoordinateParser):
    """Parser for MAI-UI ``<answer>{"coordinate": [x,y]}</answer>`` format.

    Also handles ``{"action": "click", "coordinate": [x, y]}`` in tool_call tags.
    """

    def __init__(self) -> None:
        self.answer_pattern = re.compile(
            r"<answer>\s*(\{.*?\})\s*</answer>", re.DOTALL
        )
        self.tool_call_pattern = re.compile(
            r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL
        )

    def parse(
        self,
        text: str,
        sample: Optional[DatasetSample] = None,
        request: Optional[GenerationRequest] = None,
        response: Optional[BackendResponse] = None,
    ) -> list[Coordinate]:
        import json as _json

        coordinates: list[Coordinate] = []

        for match in self.answer_pattern.finditer(text):
            try:
                data = _json.loads(match.group(1))
                coord = data.get("coordinate", [])
                if len(coord) == 2:
                    coordinates.append((float(coord[0]), float(coord[1])))
            except (ValueError, _json.JSONDecodeError):
                pass

        if coordinates:
            return coordinates

        for match in self.tool_call_pattern.finditer(text):
            try:
                data = _json.loads(match.group(1))
                args = data.get("arguments", data)
                for key in ("coordinate", "start_coordinate", "end_coordinate"):
                    coord = args.get(key, [])
                    if len(coord) == 2:
                        coordinates.append((float(coord[0]), float(coord[1])))
            except (ValueError, _json.JSONDecodeError):
                pass

        return coordinates


@PARSER_REGISTRY.register("ui_tars_action_parser")
class UITarsActionParser(BaseCoordinateParser):
    def __init__(
        self,
        factor: int = 1000,
        model_type: str = "qwen25vl",
        max_pixels: int = 16384 * 28 * 28,
        min_pixels: int = 100 * 28 * 28,
    ) -> None:
        self.factor = factor
        self.model_type = model_type
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels

    def parse(
        self,
        text: str,
        sample: Optional[DatasetSample] = None,
        request: Optional[GenerationRequest] = None,
        response: Optional[BackendResponse] = None,
    ) -> list[Coordinate]:
        if sample is None or sample.image_size is None:
            return []

        try:
            from ui_tars.action_parser import parse_action_to_structure_output
        except ImportError as exc:
            raise RuntimeError(
                "ui-tars is required for the ui_tars_action_parser. Install it with `pip install ui-tars`."
            ) from exc

        image_width, image_height = sample.image_size
        parsed_actions = parse_action_to_structure_output(
            text,
            factor=self.factor,
            origin_resized_height=image_height,
            origin_resized_width=image_width,
            model_type=self.model_type,
            max_pixels=self.max_pixels,
            min_pixels=self.min_pixels,
        )

        coordinates: list[Coordinate] = []
        for action in parsed_actions:
            action_inputs = action.get("action_inputs", {})
            for key in ("start_box", "end_box"):
                raw_box = action_inputs.get(key)
                if not raw_box:
                    continue
                box = self._parse_box(raw_box)
                if box is None:
                    continue
                coordinates.append(self._box_center(box))
        return coordinates

    def _parse_box(self, raw_box: object) -> Optional[list[float]]:
        if isinstance(raw_box, (list, tuple)):
            values = [float(value) for value in raw_box]
            return values if values else None
        if not isinstance(raw_box, str):
            return None
        try:
            values = ast.literal_eval(raw_box)
        except (SyntaxError, ValueError):
            return None
        if not isinstance(values, (list, tuple)):
            return None
        return [float(value) for value in values]

    def _box_center(self, box: list[float]) -> Coordinate:
        if len(box) >= 4:
            return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)
        if len(box) >= 2:
            return (box[0], box[1])
        raise ValueError(f"Unsupported UI-TARS box format: {box!r}")


@PARSER_REGISTRY.register("gpt5_computer_call_parser")
class GPT5ComputerCallParser(BaseCoordinateParser):
    """Parser for GPT-5.4 computer-use tool call output.

    Extracts coordinates from structured actions stored in ``response.raw``
    and falls back to text-based named-pair extraction.
    """

    def __init__(self) -> None:
        self.named_pair_patterns = [
            re.compile(
                r"(?:x|x1|start_x)\s*=\s*(-?\d+(?:\.\d+)?)\s*,\s*(?:y|y1|start_y)\s*=\s*(-?\d+(?:\.\d+)?)",
                re.IGNORECASE,
            ),
            re.compile(
                r"(?:x2|end_x)\s*=\s*(-?\d+(?:\.\d+)?)\s*,\s*(?:y2|end_y)\s*=\s*(-?\d+(?:\.\d+)?)",
                re.IGNORECASE,
            ),
        ]

    def parse(
        self,
        text: str,
        sample: Optional[DatasetSample] = None,
        request: Optional[GenerationRequest] = None,
        response: Optional[BackendResponse] = None,
    ) -> list[Coordinate]:
        if response is not None and isinstance(response.raw, dict):
            actions = response.raw.get("actions", [])
            coords = self._extract_from_actions(actions)
            if coords:
                return coords

        return self._extract_from_text(text)

    def _extract_from_actions(self, actions: list) -> list[Coordinate]:
        coordinates: list[Coordinate] = []
        for action in actions:
            if not isinstance(action, dict):
                continue
            action_type = action.get("type", "")
            if action_type in ("click", "double_click", "right_click", "scroll"):
                x = action.get("x")
                y = action.get("y")
                if x is not None and y is not None:
                    coordinates.append((float(x), float(y)))
            elif action_type == "drag":
                path = action.get("path")
                if isinstance(path, list) and len(path) >= 2:
                    first = path[0]
                    last = path[-1]
                    if isinstance(first, dict) and isinstance(last, dict):
                        coordinates.append((float(first["x"]), float(first["y"])))
                        coordinates.append((float(last["x"]), float(last["y"])))
                else:
                    sx = action.get("startX") or action.get("start_x")
                    sy = action.get("startY") or action.get("start_y")
                    ex = action.get("endX") or action.get("end_x")
                    ey = action.get("endY") or action.get("end_y")
                    if sx is not None and sy is not None:
                        coordinates.append((float(sx), float(sy)))
                    if ex is not None and ey is not None:
                        coordinates.append((float(ex), float(ey)))
        return coordinates

    def _extract_from_text(self, text: str) -> list[Coordinate]:
        matches: list[tuple[int, Coordinate]] = []
        for pattern in self.named_pair_patterns:
            for match in pattern.finditer(text):
                matches.append(
                    (match.start(), (float(match.group(1)), float(match.group(2))))
                )
        if matches:
            matches.sort(key=lambda item: item[0])
            return [coordinate for _, coordinate in matches]
        return []


@PARSER_REGISTRY.register("infigui_grounding_parser")
class InfiGUIGroundingParser(BaseCoordinateParser):
    """Parser for InfiGUI grounding output.

    Handles ``<think>...</think>`` reasoning followed by JSON:
    ``[{"point_2d": [x, y], "label": "..."}]``
    """

    def __init__(self) -> None:
        self.think_pattern = re.compile(r"<think>.*?</think>", re.DOTALL)
        self.json_block_pattern = re.compile(r"```json\s*(.*?)\s*```", re.DOTALL)
        self.point_pattern = re.compile(
            r'"point_2d"\s*:\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]'
        )

    def parse(
        self,
        text: str,
        sample: Optional[DatasetSample] = None,
        request: Optional[GenerationRequest] = None,
        response: Optional[BackendResponse] = None,
    ) -> list[Coordinate]:
        import json as _json

        cleaned = self.think_pattern.sub("", text).strip()

        json_match = self.json_block_pattern.search(cleaned)
        if json_match:
            try:
                data = _json.loads(json_match.group(1))
                coordinates: list[Coordinate] = []
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and "point_2d" in item:
                            pt = item["point_2d"]
                            if len(pt) >= 2:
                                coordinates.append((float(pt[0]), float(pt[1])))
                if coordinates:
                    return coordinates
            except (ValueError, _json.JSONDecodeError):
                pass

        # Try parsing raw JSON array (no code block markers)
        try:
            bracket_start = cleaned.find("[")
            if bracket_start != -1:
                raw_data = _json.loads(cleaned[bracket_start:])
                coordinates = []
                if isinstance(raw_data, list):
                    for item in raw_data:
                        if isinstance(item, dict) and "point_2d" in item:
                            pt = item["point_2d"]
                            if len(pt) >= 2:
                                coordinates.append((float(pt[0]), float(pt[1])))
                if coordinates:
                    return coordinates
        except (ValueError, _json.JSONDecodeError):
            pass

        matches = self.point_pattern.findall(cleaned)
        if matches:
            return [(float(x), float(y)) for x, y in matches]

        return []


@PARSER_REGISTRY.register("infigui_g1_grounding_parser")
class InfiGUIGroundingG1Parser(InfiGUIGroundingParser):
    """Parser for InfiGUI-G1 output. Same JSON point_2d format as InfiGUI-R1."""
    pass


@PARSER_REGISTRY.register("os_atlas_bbox_parser")
class OSAtlasBboxParser(BaseCoordinateParser):
    """Parser for OS-Atlas output: ``<|box_start|>(x1,y1),(x2,y2)<|box_end|>`` or plain ``(x1,y1),(x2,y2)``."""

    def __init__(self) -> None:
        self.box_tag_pattern = re.compile(
            r"\|box_start\|\>\s*\((\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\)\s*,\s*\((\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\)\s*<\|box_end\|"
        )
        self.pair_pattern = re.compile(
            r"\((\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\)"
        )

    def parse(
        self,
        text: str,
        sample: Optional[DatasetSample] = None,
        request: Optional[GenerationRequest] = None,
        response: Optional[BackendResponse] = None,
    ) -> list[Coordinate]:
        for m in self.box_tag_pattern.finditer(text):
            x1, y1, x2, y2 = float(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4))
            return [((x1 + x2) / 2.0, (y1 + y2) / 2.0)]

        pairs = self.pair_pattern.findall(text)
        if len(pairs) >= 2:
            x1, y1 = float(pairs[0][0]), float(pairs[0][1])
            x2, y2 = float(pairs[1][0]), float(pairs[1][1])
            return [((x1 + x2) / 2.0, (y1 + y2) / 2.0)]
        if len(pairs) == 1:
            return [(float(pairs[0][0]), float(pairs[0][1]))]
        return []


@PARSER_REGISTRY.register("seeclick_point_parser")
class SeeClickPointParser(BaseCoordinateParser):
    """Parser for SeeClick output: ``(x, y)`` with values in [0,1]."""

    def __init__(self) -> None:
        self.point_pattern = re.compile(
            r"\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)"
        )

    def parse(
        self,
        text: str,
        sample: Optional[DatasetSample] = None,
        request: Optional[GenerationRequest] = None,
        response: Optional[BackendResponse] = None,
    ) -> list[Coordinate]:
        matches = self.point_pattern.findall(text)
        if matches:
            return [(float(matches[0][0]), float(matches[0][1]))]
        return []
