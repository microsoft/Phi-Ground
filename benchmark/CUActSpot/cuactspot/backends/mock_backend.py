from __future__ import annotations

from cuactspot.backends.base import BaseGenerationBackend
from cuactspot.registry import BACKEND_REGISTRY
from cuactspot.types import BackendResponse, GenerationRequest


@BACKEND_REGISTRY.register("static_response")
class StaticResponseBackend(BaseGenerationBackend):
    def __init__(self, response_text: str = "[(0, 0)]") -> None:
        self.response_text = response_text

    def generate(self, request: GenerationRequest) -> BackendResponse:
        return BackendResponse(
            text=self.response_text,
            raw={"response_text": self.response_text},
            metadata={"backend": "static_response"},
        )