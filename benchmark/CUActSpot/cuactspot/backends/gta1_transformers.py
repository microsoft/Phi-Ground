from __future__ import annotations

import re
from typing import Any, Dict, Optional

from cuactspot.backends.base import BaseGenerationBackend
from cuactspot.registry import BACKEND_REGISTRY
from cuactspot.types import BackendResponse, GenerationRequest
from cuactspot.utils.qwen import smart_resize

GTA1_SYSTEM_PROMPT = (
    "You are a GUI agent. You are given a task and a screenshot of the screen. "
    "You need to perform a series of pyautogui actions to complete the task."
)


@BACKEND_REGISTRY.register("gta1_transformers")
class GTA1TransformersBackend(BaseGenerationBackend):
    def __init__(
        self,
        model: str = "Salesforce/GTA1-7B",
        torch_dtype: str = "bfloat16",
        device_map: str = "auto",
        attn_implementation: Optional[str] = "flash_attention_2",
        max_new_tokens: int = 128,
        min_pixels: int = 3136,
        max_pixels: int = 4096 * 2160,
        default_system_prompt: Optional[str] = None,
        rewrite_media_placeholders: bool = True,
        apply_chat_template_with: str = "tokenizer",
        **kwargs: Any,
    ) -> None:
        import torch
        from transformers import (
            AutoConfig,
            AutoProcessor,
            AutoTokenizer,
            Qwen2_5_VLForConditionalGeneration,
        )

        model_kwargs: Dict[str, Any] = {
            "torch_dtype": self._resolve_torch_dtype(torch, torch_dtype),
            "device_map": device_map,
        }
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation
        model_kwargs.update(kwargs)

        self.model_name = model
        self.max_new_tokens = max_new_tokens
        self._default_system_prompt = default_system_prompt if default_system_prompt is not None else GTA1_SYSTEM_PROMPT
        self._rewrite_media = rewrite_media_placeholders
        self._apply_chat_template_with = apply_chat_template_with
        config = AutoConfig.from_pretrained(model, trust_remote_code=True)
        rope_scaling = getattr(config, "rope_scaling", None)
        if isinstance(rope_scaling, dict) and "mrope_section" not in rope_scaling:
            config.rope_scaling = {
                **rope_scaling,
                "mrope_section": [16, 24, 24],
            }
        self._tokenizer = AutoTokenizer.from_pretrained(
            model,
            trust_remote_code=True,
        )
        self._processor = AutoProcessor.from_pretrained(
            model,
            trust_remote_code=True,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model,
            config=config,
            **model_kwargs,
        )

    def generate(self, request: GenerationRequest) -> BackendResponse:
        if request.image is None:
            raise ValueError("GTA1TransformersBackend requires a prepared image")

        image = request.image.payload
        original_width = request.image.width or image.width
        original_height = request.image.height or image.height
        resized_height, resized_width = smart_resize(
            original_height,
            original_width,
            factor=self._processor.image_processor.patch_size
            * self._processor.image_processor.merge_size,
            min_pixels=self._processor.image_processor.min_pixels,
            max_pixels=self._processor.image_processor.max_pixels,
        )
        resized_image = image.resize((resized_width, resized_height))

        system_prompt = request.prompt.system_prompt or self._default_system_prompt
        messages = []
        if system_prompt:
            messages.append(
                {"role": "system", "content": system_prompt},
            )
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": resized_image},
                    {"type": "text", "text": request.prompt.user_prompt},
                ],
            },
        )

        chat_template_client = (
            self._processor
            if self._apply_chat_template_with == "processor"
            else self._tokenizer
        )
        text = chat_template_client.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        if self._rewrite_media:
            text = self._rewrite_media_placeholder(text)
        inputs = self._processor(
            text=[text],
            images=[resized_image],
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self._model.device)

        generation_kwargs: Dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": False,
            "temperature": 1.0,
            "use_cache": True,
        }
        generation_kwargs.update(request.generation_kwargs)
        generation_kwargs.pop("max_tokens", None)
        if generation_kwargs.get("temperature", 1.0) not in (0, 0.0, 1.0):
            generation_kwargs["do_sample"] = True

        output_ids = self._model.generate(**inputs, **generation_kwargs)
        generated_ids = [
            sample_output_ids[len(input_ids):]
            for input_ids, sample_output_ids in zip(inputs.input_ids, output_ids)
        ]
        output_text = self._tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )[0]

        return BackendResponse(
            text=output_text,
            raw=generated_ids,
            metadata={
                "backend": "gta1_transformers",
                "model": self.model_name,
                "resized_width": resized_width,
                "resized_height": resized_height,
                "generation_kwargs": generation_kwargs,
            },
        )

    def close(self) -> None:
        self._model = None
        self._processor = None
        self._tokenizer = None

    def _resolve_torch_dtype(self, torch_module, value: str):
        if hasattr(torch_module, value):
            return getattr(torch_module, value)
        return value

    def _rewrite_media_placeholder(self, text: str) -> str:
        rewritten, count = re.subn(
            r"<\|media_begin\|>.*?<\|media_end\|>",
            "<|vision_start|><|image_pad|><|vision_end|>",
            text,
            flags=re.S,
        )
        if count == 0:
            raise RuntimeError(
                "GTA1 prompt template did not contain the expected media placeholder block."
            )
        return rewritten