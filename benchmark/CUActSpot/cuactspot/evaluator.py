from __future__ import annotations

from eval_utils import validate_against_json_file

from cuactspot.types import Coordinate, DatasetSample


class CoordinateEvaluator:
    def evaluate(
        self,
        sample: DatasetSample,
        coordinates: list[Coordinate],
    ) -> tuple[bool, str]:
        return validate_against_json_file(coordinates, sample.raw_annotation)