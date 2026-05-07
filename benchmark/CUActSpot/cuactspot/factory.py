from __future__ import annotations

from dataclasses import replace
from typing import Any, Optional, Type, TypeVar

import cuactspot.adapters  # noqa: F401
import cuactspot.backends  # noqa: F401
import cuactspot.components  # noqa: F401
import cuactspot.models  # noqa: F401
from cuactspot.adapters.base import BaseModelAdapter
from cuactspot.backends.base import BaseGenerationBackend
from cuactspot.components.images import BaseImagePreprocessor
from cuactspot.components.parsing import BaseCoordinateParser
from cuactspot.components.prompting import BasePromptBuilder
from cuactspot.components.transforms import BaseCoordinateTransformer
from cuactspot.config import ComponentConfig, ModelConfig
from cuactspot.models.registry import get_model_spec
from cuactspot.registry import (
    ADAPTER_REGISTRY,
    BACKEND_REGISTRY,
    IMAGE_PREPROCESSOR_REGISTRY,
    PARSER_REGISTRY,
    PROMPT_BUILDER_REGISTRY,
    TRANSFORMER_REGISTRY,
    Registry,
)
from cuactspot.utils.importing import load_symbol

ComponentT = TypeVar("ComponentT")


def build_model_adapter(model_config: ModelConfig) -> BaseModelAdapter:
    resolved = apply_model_preset(model_config)
    backend = create_component(
        resolved.backend,
        BACKEND_REGISTRY,
        BaseGenerationBackend,
    )
    prompt_builder = create_component(
        resolved.prompt_builder,
        PROMPT_BUILDER_REGISTRY,
        BasePromptBuilder,
    )
    image_preprocessor = create_component(
        resolved.image_preprocessor,
        IMAGE_PREPROCESSOR_REGISTRY,
        BaseImagePreprocessor,
    )
    parser = create_component(
        resolved.parser,
        PARSER_REGISTRY,
        BaseCoordinateParser,
    )
    transformer = create_component(
        resolved.transformer,
        TRANSFORMER_REGISTRY,
        BaseCoordinateTransformer,
    )

    adapter = create_component(
        resolved.adapter,
        ADAPTER_REGISTRY,
        BaseModelAdapter,
        name=resolved.name,
        backend=backend,
        prompt_builder=prompt_builder,
        image_preprocessor=image_preprocessor,
        parser=parser,
        transformer=transformer,
        system_prompt=resolved.system_prompt,
        generation_kwargs=resolved.generation,
        metadata=resolved.metadata,
    )
    if adapter is None:
        raise ValueError("model.adapter must be configured")
    return adapter


def apply_model_preset(model_config: ModelConfig) -> ModelConfig:
    if not model_config.preset:
        return model_config

    spec = get_model_spec(model_config.preset)
    return replace(
        model_config,
        adapter=_merge_component(spec.adapter, model_config.adapter),
        backend=_merge_component(spec.backend, model_config.backend),
        prompt_builder=_merge_component(
            spec.prompt_builder,
            model_config.prompt_builder,
        ),
        image_preprocessor=_merge_component(
            spec.image_preprocessor,
            model_config.image_preprocessor,
        ),
        parser=_merge_component(spec.parser, model_config.parser),
        transformer=_merge_component(spec.transformer, model_config.transformer),
        system_prompt=model_config.system_prompt or spec.system_prompt,
        generation={**spec.generation, **model_config.generation},
        metadata={**spec.metadata, **model_config.metadata},
    )


def create_component(
    component_config: Optional[ComponentConfig],
    registry: Registry[type],
    expected_base: Optional[Type[ComponentT]] = None,
    **extra_kwargs: Any,
) -> Optional[ComponentT]:
    if component_config is None:
        return None

    target = _resolve_target(component_config.target, registry)
    if expected_base is not None and not issubclass(target, expected_base):
        raise TypeError(
            f"{target} is not a subclass of {expected_base.__name__}"
        )

    init_kwargs = {
        key: value
        for key, value in extra_kwargs.items()
        if value is not None
    }
    init_kwargs.update(component_config.kwargs)
    return target(**init_kwargs)


def _resolve_target(target: str, registry: Registry[type]) -> type:
    if registry.has(target):
        return registry.get(target)
    symbol = load_symbol(target)
    if not isinstance(symbol, type):
        raise TypeError(f"Resolved target is not a class: {target}")
    return symbol


def _merge_component(
    base: Optional[ComponentConfig],
    override: Optional[ComponentConfig],
) -> Optional[ComponentConfig]:
    if base is None:
        return override
    if override is None:
        return base
    if override.target and override.target != base.target:
        return override
    return ComponentConfig(
        target=override.target or base.target,
        kwargs={**base.kwargs, **override.kwargs},
    )