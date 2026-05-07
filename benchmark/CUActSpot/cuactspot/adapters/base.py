from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from cuactspot.backends.base import BaseGenerationBackend
from cuactspot.components.images import BaseImagePreprocessor, IdentityImagePreprocessor
from cuactspot.components.parsing import BaseCoordinateParser, RegexCoordinateParser
from cuactspot.components.prompting import BasePromptBuilder, PlainTaskPromptBuilder
from cuactspot.components.transforms import (
    BaseCoordinateTransformer,
    IdentityCoordinateTransformer,
)
from cuactspot.types import BackendResponse, DatasetSample, GenerationRequest, ModelPrediction
from cuactspot.utils.serialize import make_jsonable


class BaseModelAdapter(ABC):
    def __init__(
        self,
        name: str,
        backend: Optional[BaseGenerationBackend] = None,
        prompt_builder: Optional[BasePromptBuilder] = None,
        image_preprocessor: Optional[BaseImagePreprocessor] = None,
        parser: Optional[BaseCoordinateParser] = None,
        transformer: Optional[BaseCoordinateTransformer] = None,
        system_prompt: Optional[str] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.name = name
        self.backend = backend
        self.prompt_builder = prompt_builder or PlainTaskPromptBuilder()
        self.image_preprocessor = image_preprocessor or IdentityImagePreprocessor()
        self.parser = parser or RegexCoordinateParser()
        self.transformer = transformer or IdentityCoordinateTransformer()
        self.system_prompt = system_prompt
        self.generation_kwargs = dict(generation_kwargs or {})
        self.metadata = dict(metadata or {})

    @abstractmethod
    def predict(self, sample: DatasetSample) -> ModelPrediction:
        raise NotImplementedError

    def build_request(self, sample: DatasetSample) -> GenerationRequest:
        prompt = self.prompt_builder.build(sample, system_prompt=self.system_prompt)
        image = self.image_preprocessor.process(sample)
        request_metadata = make_jsonable(
            {
                **self.metadata,
                "prompt_text": prompt.user_prompt,
                "system_prompt": prompt.system_prompt,
                "messages": prompt.messages,
                "image_path": str(sample.image_path),
                "generation_kwargs": dict(self.generation_kwargs),
            }
        )
        return GenerationRequest(
            sample=sample,
            prompt=prompt,
            image=image,
            generation_kwargs=dict(self.generation_kwargs),
            metadata=request_metadata,
        )

    def build_prediction(
        self,
        sample: DatasetSample,
        request: GenerationRequest,
        response: BackendResponse,
    ) -> ModelPrediction:
        coordinates = self.parser.parse(
            response.text,
            sample=sample,
            request=request,
            response=response,
        )
        transformed_coordinates = self.transformer.transform(
            coordinates,
            sample=sample,
            request=request,
            response=response,
        )
        return ModelPrediction(
            sample_id=sample.sample_id,
            raw_output=response.text,
            coordinates=transformed_coordinates,
            request_metadata=request.metadata,
            response_metadata=make_jsonable(response.metadata),
            metadata=make_jsonable(dict(self.metadata)),
        )

    def close(self) -> None:
        if self.backend is not None:
            self.backend.close()

    def predict_batch(self, samples: list) -> list:
        """Batch predict for multiple samples. Falls back to sequential if backend lacks batch support."""
        if self.backend is None or not hasattr(self.backend, "generate_batch"):
            return [self.predict(s) for s in samples]
        requests = [self.build_request(s) for s in samples]
        responses = self.backend.generate_batch(requests)
        return [
            self.build_prediction(sample, request, response)
            for sample, request, response in zip(samples, requests, responses)
        ]