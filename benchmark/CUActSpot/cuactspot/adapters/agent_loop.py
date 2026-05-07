from __future__ import annotations

from abc import abstractmethod

from cuactspot.adapters.base import BaseModelAdapter
from cuactspot.types import DatasetSample, ModelPrediction


class BaseAgentLoopAdapter(BaseModelAdapter):
    def predict(self, sample: DatasetSample) -> ModelPrediction:
        return self.run_agent_loop(sample)

    @abstractmethod
    def run_agent_loop(self, sample: DatasetSample) -> ModelPrediction:
        raise NotImplementedError