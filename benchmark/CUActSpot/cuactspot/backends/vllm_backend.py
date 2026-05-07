from __future__ import annotations

from cuactspot.backends.base import BaseGenerationBackend
from cuactspot.registry import BACKEND_REGISTRY
from cuactspot.types import BackendResponse, GenerationRequest


@BACKEND_REGISTRY.register("vllm")
class VLLMBackend(BaseGenerationBackend):
    def __init__(self, model: str, **kwargs) -> None:
        self.model = model
        self.kwargs = kwargs

    def generate(self, request: GenerationRequest) -> BackendResponse:
        raise NotImplementedError(
            "VLLMBackend is a template backend. Create a model-specific subclass or replace it with your own backend that knows how to format multimodal vLLM requests for the target model."
        )