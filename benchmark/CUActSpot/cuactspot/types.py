from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

Coordinate = Tuple[float, float]


@dataclass
class PolygonAnnotation:
    polygon_id: str
    polygon_type: str
    points: List[Coordinate]
    rank: str = ""
    visible: bool = True


@dataclass
class DatasetSample:
    sample_id: str
    json_path: Path
    image_path: Path
    task: Optional[str]
    task_en: Optional[str]
    modality: Optional[str]
    polygons: List[PolygonAnnotation]
    raw_annotation: Dict[str, Any]
    image_size: Optional[Tuple[int, int]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def task_text(self) -> str:
        return self.task_en or self.task or ""


@dataclass
class PromptSpec:
    user_prompt: str
    system_prompt: Optional[str] = None
    messages: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PreparedImage:
    source_path: Path
    payload: Any
    width: Optional[int] = None
    height: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationRequest:
    sample: DatasetSample
    prompt: PromptSpec
    image: Optional[PreparedImage] = None
    generation_kwargs: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BackendResponse:
    text: str
    raw: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelPrediction:
    sample_id: str
    raw_output: str
    coordinates: List[Coordinate]
    request_metadata: Dict[str, Any] = field(default_factory=dict)
    response_metadata: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SampleEvaluationResult:
    sample_id: str
    passed: bool
    error_message: str
    coordinates: List[Coordinate]
    raw_output: str
    task: str
    modality: Optional[str]
    duration_sec: float
    model_input: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "data_id": self.sample_id,
            "sample_id": self.sample_id,
            "modality": self.modality,
            "task": self.task,
            "model_input": self.model_input,
            "model_output": self.raw_output,
            "extracted_coordinates": [[x, y] for x, y in self.coordinates],
            "correct": self.passed,
            "error_message": self.error_message,
            "duration_sec": self.duration_sec,
            "metadata": self.metadata,
        }