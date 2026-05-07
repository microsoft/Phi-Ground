from __future__ import annotations

from typing import Any, Dict, Optional

from cuactspot.backends.base import BaseGenerationBackend
from cuactspot.registry import BACKEND_REGISTRY
from cuactspot.types import BackendResponse, GenerationRequest


@BACKEND_REGISTRY.register("phi_ground_vllm")
class PhiGroundVLLMBackend(BaseGenerationBackend):
    def __init__(
        self,
        model: str = "microsoft/Phi-Ground",
        tensor_parallel_size: int = 1,
        trust_remote_code: bool = True,
        max_num_seqs: int = 200,
        gpu_memory_utilization: Optional[float] = None,
        max_model_len: Optional[int] = None,
        dtype: Optional[str] = None,
        download_dir: Optional[str] = None,
        revision: Optional[str] = None,
        tokenizer: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        from vllm import LLM, SamplingParams

        llm_kwargs: Dict[str, Any] = {
            "model": model,
            "tensor_parallel_size": tensor_parallel_size,
            "trust_remote_code": trust_remote_code,
            "max_num_seqs": max_num_seqs,
        }
        optional_kwargs = {
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_model_len": max_model_len,
            "dtype": dtype,
            "download_dir": download_dir,
            "revision": revision,
            "tokenizer": tokenizer,
        }
        llm_kwargs.update({key: value for key, value in optional_kwargs.items() if value is not None})
        llm_kwargs.update(kwargs)

        self.model_name = model
        self._sampling_params_cls = SamplingParams
        self._llm = LLM(**llm_kwargs)

    def generate(self, request: GenerationRequest) -> BackendResponse:
        if request.image is None:
            raise ValueError("PhiGroundVLLMBackend requires a prepared image")

        sampling_kwargs: Dict[str, Any] = {
            "temperature": 0.0,
            "max_tokens": 64,
        }
        sampling_kwargs.update(request.generation_kwargs)

        prompt_payload = {
            "prompt": request.prompt.user_prompt,
            "multi_modal_data": {
                "image": [request.image.payload],
            },
        }
        outputs = self._llm.generate(
            [prompt_payload],
            sampling_params=self._sampling_params_cls(**sampling_kwargs),
        )

        output_text = ""
        if outputs and outputs[0].outputs:
            output_text = outputs[0].outputs[0].text

        return BackendResponse(
            text=output_text,
            metadata={
                "backend": "phi_ground_vllm",
                "model": self.model_name,
                "sampling_kwargs": sampling_kwargs,
            },
        )

    def close(self) -> None:
        self._llm = None