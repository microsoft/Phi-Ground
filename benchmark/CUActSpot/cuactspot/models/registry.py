from __future__ import annotations

from cuactspot.models.spec import ModelSpec

MODEL_SPECS: dict[str, ModelSpec] = {}


def register_model_spec(spec: ModelSpec) -> ModelSpec:
    if spec.name in MODEL_SPECS:
        raise ValueError(f"Model preset '{spec.name}' is already registered")
    MODEL_SPECS[spec.name] = spec
    return spec


def get_model_spec(name: str) -> ModelSpec:
    try:
        return MODEL_SPECS[name]
    except KeyError as exc:
        raise KeyError(f"Unknown model preset: {name}") from exc