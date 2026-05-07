from __future__ import annotations

from cuactspot.adapters.base import BaseModelAdapter
from cuactspot.registry import ADAPTER_REGISTRY
from cuactspot.types import DatasetSample, ModelPrediction


@ADAPTER_REGISTRY.register("composable_adapter")
class ComposableModelAdapter(BaseModelAdapter):
    def predict(self, sample: DatasetSample) -> ModelPrediction:
        if self.backend is None:
            raise ValueError("ComposableModelAdapter requires a backend")
        request = self.build_request(sample)
        response = self.backend.generate(request)
        return self.build_prediction(sample, request, response)