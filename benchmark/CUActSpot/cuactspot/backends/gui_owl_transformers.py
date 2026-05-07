from __future__ import annotations

from typing import Any, Dict, Optional

from cuactspot.backends.base import BaseGenerationBackend
from cuactspot.registry import BACKEND_REGISTRY
from cuactspot.types import BackendResponse, GenerationRequest


@BACKEND_REGISTRY.register("gui_owl_transformers")
class GUIOwlTransformersBackend(BaseGenerationBackend):
    """Backend for GUI-Owl 1.5 models based on Qwen3-VL."""

    def __init__(
        self,
        model: str = "mPLUG/GUI-Owl-1.5-8B-Instruct",
        torch_dtype: str = "bfloat16",
        device_map: str = "auto",
        attn_implementation: Optional[str] = "flash_attention_2",
        max_new_tokens: int = 2048,
        min_pixels: int = 196 * 32 * 32,
        max_pixels: int = 9800 * 32 * 32,
        default_system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        # Work around cuDNN 3D-conv init failures on some driver versions.
        if not torch.backends.cudnn.is_available() or not torch.backends.cudnn.enabled:
            pass
        else:
            try:
                torch.backends.cudnn.enabled = True
            except Exception:
                pass
            # Probe cuDNN with a small 3D conv; disable if it fails.
            if device_map != "cpu" and torch.cuda.is_available():
                try:
                    _t = torch.randn(1, 1, 1, 3, 3, device="cuda")
                    torch.nn.functional.conv3d(_t, torch.randn(1, 1, 1, 3, 3, device="cuda"), padding=1)
                except RuntimeError:
                    torch.backends.cudnn.enabled = False

        model_kwargs: Dict[str, Any] = {
            "torch_dtype": self._resolve_torch_dtype(torch, torch_dtype),
            "device_map": device_map,
        }
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation
        model_kwargs.update(kwargs)

        self.model_name = model
        self.max_new_tokens = max_new_tokens
        self._default_system_prompt = default_system_prompt
        self._processor = AutoProcessor.from_pretrained(
            model,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        self._model = Qwen3VLForConditionalGeneration.from_pretrained(
            model,
            **model_kwargs,
        )

    def generate(self, request: GenerationRequest) -> BackendResponse:
        if request.image is None:
            raise ValueError("GUIOwlTransformersBackend requires a prepared image")

        image = request.image.payload

        system_prompt = request.prompt.system_prompt or self._default_system_prompt
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": request.prompt.user_prompt},
                ],
            },
        )

        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
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
        output_text = self._processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return BackendResponse(
            text=output_text,
            raw=generated_ids,
            metadata={
                "backend": "gui_owl_transformers",
                "model": self.model_name,
                "generation_kwargs": generation_kwargs,
            },
        )

    def generate_batch(self, requests: list) -> list:
        """Batch generate for multiple requests at once to maximize GPU utilization."""
        from qwen_vl_utils import process_vision_info

        all_messages = []
        for request in requests:
            if request.image is None:
                raise ValueError("GUIOwlTransformersBackend requires a prepared image")
            image = request.image.payload
            system_prompt = request.prompt.system_prompt or self._default_system_prompt
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
            messages.append({
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": request.prompt.user_prompt},
                ],
            })
            all_messages.append(messages)

        texts = [
            self._processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in all_messages
        ]
        image_inputs, video_inputs = process_vision_info(all_messages)
        inputs = self._processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
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
        if requests:
            generation_kwargs.update(requests[0].generation_kwargs)
        generation_kwargs.pop("max_tokens", None)

        output_ids = self._model.generate(**inputs, **generation_kwargs)

        generated_ids_list = [
            out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)
        ]
        output_texts = self._processor.batch_decode(
            generated_ids_list,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        responses = []
        for text in output_texts:
            responses.append(BackendResponse(
                text=text,
                raw={},
                metadata={
                    "backend": "gui_owl_transformers",
                    "model": self.model_name,
                    "batched": True,
                },
            ))
        return responses

    def close(self) -> None:
        self._model = None
        self._processor = None

    def _resolve_torch_dtype(self, torch_module, value: str):
        if hasattr(torch_module, value):
            return getattr(torch_module, value)
        return value
