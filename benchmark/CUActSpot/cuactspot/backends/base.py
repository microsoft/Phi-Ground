from __future__ import annotations

from abc import ABC, abstractmethod

from cuactspot.types import BackendResponse, GenerationRequest


class BaseGenerationBackend(ABC):
    @abstractmethod
    def generate(self, request: GenerationRequest) -> BackendResponse:
        raise NotImplementedError

    def close(self) -> None:
        return None