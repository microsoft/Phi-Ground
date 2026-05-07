from __future__ import annotations

import os
import re
from typing import Any, Dict, Optional

from cuactspot.backends.base import BaseGenerationBackend
from cuactspot.registry import BACKEND_REGISTRY
from cuactspot.types import BackendResponse, GenerationRequest


@BACKEND_REGISTRY.register("chat_vllm")
class ChatVLLMBackend(BaseGenerationBackend):
    def __init__(
        self,
        model: str,
        tokenizer: Optional[str] = None,
        processor: Optional[str] = None,
        tensor_parallel_size: int = 1,
        trust_remote_code: bool = False,
        max_num_seqs: int = 64,
        gpu_memory_utilization: Optional[float] = None,
        max_model_len: Optional[int] = None,
        dtype: Optional[str] = None,
        download_dir: Optional[str] = None,
        revision: Optional[str] = None,
        tokenizer_mode: Optional[str] = None,
        apply_chat_template_with: str = "tokenizer",
        rewrite_media_placeholders: bool = False,
        multi_modal_key: str = "image",
        image_as_list: bool = True,
        **kwargs: Any,
    ) -> None:
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

        from transformers import AutoProcessor, AutoTokenizer
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
            "tokenizer_mode": tokenizer_mode,
        }
        llm_kwargs.update(
            {key: value for key, value in optional_kwargs.items() if value is not None}
        )
        llm_kwargs.update(kwargs)

        self.model_name = model
        self.apply_chat_template_with = apply_chat_template_with
        self.rewrite_media_placeholders = rewrite_media_placeholders
        self.multi_modal_key = multi_modal_key
        self.image_as_list = image_as_list
        self._sampling_params_cls = SamplingParams
        self._llm = LLM(**llm_kwargs)

        chat_template_target = processor or tokenizer or model
        if apply_chat_template_with == "processor":
            self._chat_template_client = AutoProcessor.from_pretrained(
                chat_template_target,
                trust_remote_code=trust_remote_code,
            )
        else:
            self._chat_template_client = AutoTokenizer.from_pretrained(
                chat_template_target,
                trust_remote_code=trust_remote_code,
            )

    def generate(self, request: GenerationRequest) -> BackendResponse:
        if request.image is None:
            raise ValueError("ChatVLLMBackend requires a prepared image")

        prompt_text = self._build_prompt_text(request)
        sampling_kwargs: Dict[str, Any] = {
            "temperature": 0.0,
            "max_tokens": 256,
        }
        sampling_kwargs.update(request.generation_kwargs)

        image_payload = request.image.payload
        multi_modal_data = {
            self.multi_modal_key: [image_payload] if self.image_as_list else image_payload
        }
        outputs = self._llm.generate(
            [{"prompt": prompt_text, "multi_modal_data": multi_modal_data}],
            sampling_params=self._sampling_params_cls(**sampling_kwargs),
        )

        output_text = ""
        if outputs and outputs[0].outputs:
            output_text = outputs[0].outputs[0].text

        return BackendResponse(
            text=output_text,
            raw=outputs,
            metadata={
                "backend": "chat_vllm",
                "model": self.model_name,
                "prompt_text": prompt_text,
                "sampling_kwargs": sampling_kwargs,
            },
        )

    def generate_batch(self, requests: list) -> list:
        """Batch many requests through vLLM's continuous batching for high throughput."""
        if not requests:
            return []

        prompts: list[Dict[str, Any]] = []
        prompt_texts: list[str] = []
        for request in requests:
            if request.image is None:
                raise ValueError("ChatVLLMBackend requires a prepared image")
            prompt_text = self._build_prompt_text(request)
            prompt_texts.append(prompt_text)
            image_payload = request.image.payload
            multi_modal_data = {
                self.multi_modal_key: (
                    [image_payload] if self.image_as_list else image_payload
                )
            }
            prompts.append({"prompt": prompt_text, "multi_modal_data": multi_modal_data})

        # Use the first request's generation_kwargs (they should all be identical
        # for a given run; per-sample sampling overrides aren't currently supported).
        sampling_kwargs: Dict[str, Any] = {"temperature": 0.0, "max_tokens": 256}
        sampling_kwargs.update(requests[0].generation_kwargs)
        sampling_params = self._sampling_params_cls(**sampling_kwargs)

        outputs = self._llm.generate(prompts, sampling_params=sampling_params)

        responses: list[BackendResponse] = []
        for prompt_text, output in zip(prompt_texts, outputs):
            text = output.outputs[0].text if output and output.outputs else ""
            responses.append(
                BackendResponse(
                    text=text,
                    raw=output,
                    metadata={
                        "backend": "chat_vllm",
                        "model": self.model_name,
                        "prompt_text": prompt_text,
                        "sampling_kwargs": sampling_kwargs,
                    },
                )
            )
        return responses

    def close(self) -> None:
        self._llm = None
        self._chat_template_client = None

    def _build_prompt_text(self, request: GenerationRequest) -> str:
        if not request.prompt.messages:
            return request.prompt.user_prompt

        apply_chat_template = getattr(self._chat_template_client, "apply_chat_template", None)
        if apply_chat_template is None:
            raise RuntimeError(
                f"{type(self._chat_template_client).__name__} does not expose apply_chat_template"
            )

        prompt_text = apply_chat_template(
            request.prompt.messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        if self.rewrite_media_placeholders:
            prompt_text = re.sub(
                r"<\|media_begin\|>.*?<\|media_end\|>",
                "<|vision_start|><|image_pad|><|vision_end|>",
                prompt_text,
                flags=re.S,
            )
        return prompt_text