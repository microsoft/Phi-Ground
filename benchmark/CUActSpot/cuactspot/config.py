from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


@dataclass
class ComponentConfig:
    target: str
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetConfig:
    data_dir: str = "data/ActSpot"
    benchmark: str = "actspot"
    annotations_dir: Optional[str] = None
    sample_ids: List[str] = field(default_factory=list)
    include_modalities: List[str] = field(default_factory=list)
    exclude_modalities: List[str] = field(default_factory=list)
    limit: Optional[int] = None
    image_extensions: List[str] = field(
        default_factory=lambda: [".png", ".jpg", ".jpeg", ".webp"]
    )


@dataclass
class ModelConfig:
    name: str
    preset: Optional[str] = None
    adapter: Optional[ComponentConfig] = None
    backend: Optional[ComponentConfig] = None
    prompt_builder: Optional[ComponentConfig] = None
    image_preprocessor: Optional[ComponentConfig] = None
    parser: Optional[ComponentConfig] = None
    transformer: Optional[ComponentConfig] = None
    system_prompt: Optional[str] = None
    generation: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OutputConfig:
    output_dir: str = "outputs/default"
    report_file: str = "report.json"
    visualize_dir: Optional[str] = None
    overwrite: bool = False


@dataclass
class RuntimeConfig:
    continue_on_error: bool = True
    max_samples: Optional[int] = None
    save_raw_output: bool = True
    verbose: bool = False
    num_workers: int = 1
    batch_size: int = 1
    best_of_n: int = 1
    best_of_temperature: float = 0.7


@dataclass
class RunConfig:
    dataset: DatasetConfig
    model: ModelConfig
    output: OutputConfig = field(default_factory=OutputConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)


def load_run_config(config_path: Union[str, Path]) -> RunConfig:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as file_obj:
        payload = json.load(file_obj)
    return parse_run_config(payload)


def parse_run_config(payload: Dict[str, Any]) -> RunConfig:
    dataset_payload = dict(payload.get("dataset", {}))
    model_payload = dict(payload.get("model", {}))
    output_payload = dict(payload.get("output", {}))
    runtime_payload = dict(payload.get("runtime", {}))

    if "name" not in model_payload:
        raise ValueError("model.name is required in the run config")

    dataset_fields = {f.name for f in DatasetConfig.__dataclass_fields__.values()}
    dataset_kwargs = {k: v for k, v in dataset_payload.items() if k in dataset_fields}

    return RunConfig(
        dataset=DatasetConfig(**dataset_kwargs),
        model=ModelConfig(
            name=model_payload["name"],
            preset=model_payload.get("preset"),
            adapter=_parse_component(model_payload.get("adapter")),
            backend=_parse_component(model_payload.get("backend")),
            prompt_builder=_parse_component(model_payload.get("prompt_builder")),
            image_preprocessor=_parse_component(
                model_payload.get("image_preprocessor")
            ),
            parser=_parse_component(model_payload.get("parser")),
            transformer=_parse_component(model_payload.get("transformer")),
            system_prompt=model_payload.get("system_prompt"),
            generation=dict(model_payload.get("generation", {})),
            metadata=dict(model_payload.get("metadata", {})),
        ),
        output=OutputConfig(
            output_dir=output_payload.get("output_dir", "outputs/default"),
            report_file=output_payload.get("report_file", "report.json"),
            visualize_dir=output_payload.get("visualize_dir"),
            overwrite=output_payload.get("overwrite", False),
        ),
        runtime=RuntimeConfig(
            continue_on_error=runtime_payload.get("continue_on_error", True),
            max_samples=runtime_payload.get("max_samples"),
            save_raw_output=runtime_payload.get("save_raw_output", True),
            verbose=runtime_payload.get("verbose", False),
            num_workers=runtime_payload.get("num_workers", 1),
            batch_size=runtime_payload.get("batch_size", 1),
            best_of_n=runtime_payload.get("best_of_n", 1),
            best_of_temperature=runtime_payload.get("best_of_temperature", 0.7),
        ),
    )


def _parse_component(payload: Any) -> Optional[ComponentConfig]:
    if payload is None:
        return None
    if isinstance(payload, str):
        return ComponentConfig(target=payload)
    if isinstance(payload, dict):
        target = payload.get("target") or payload.get("name")
        if not target:
            raise ValueError(f"Invalid component config: {payload}")
        return ComponentConfig(target=target, kwargs=dict(payload.get("kwargs", {})))
    raise TypeError(f"Unsupported component config: {payload!r}")