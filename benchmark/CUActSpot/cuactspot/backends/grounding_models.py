from __future__ import annotations

from typing import Any, Dict, Optional

from cuactspot.backends.base import BaseGenerationBackend
from cuactspot.registry import BACKEND_REGISTRY
from cuactspot.types import BackendResponse, GenerationRequest


@BACKEND_REGISTRY.register("seeclick_transformers")
class SeeClickTransformersBackend(BaseGenerationBackend):
    """Backend for SeeClick (Qwen-VL based)."""

    def __init__(
        self,
        model: str = "cckevinn/SeeClick",
        torch_dtype: str = "bfloat16",
        device_map: str = "auto",
        **kwargs: Any,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model
        dtype = getattr(torch, torch_dtype) if hasattr(torch, torch_dtype) else torch_dtype
        self._tokenizer = AutoTokenizer.from_pretrained(
            model, trust_remote_code=True,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            model,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
        ).eval()

    def generate(self, request: GenerationRequest) -> BackendResponse:
        if request.image is None:
            raise ValueError("SeeClickTransformersBackend requires a prepared image")

        image_path = str(request.image.source_path)
        prompt_text = request.prompt.user_prompt

        query = self._tokenizer.from_list_format([
            {"image": image_path},
            {"text": prompt_text},
        ])
        inputs = self._tokenizer(query, return_tensors="pt")
        inputs = inputs.to(self._model.device)

        output_ids = self._model.generate(**inputs, do_sample=False, max_new_tokens=256)
        generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
        output_text = self._tokenizer.decode(
            generated_ids[0], skip_special_tokens=True,
        ).strip()

        return BackendResponse(
            text=output_text,
            metadata={"backend": "seeclick_transformers", "model": self.model_name},
        )

    def close(self) -> None:
        self._model = None
        self._tokenizer = None


@BACKEND_REGISTRY.register("uground_v1_transformers")
class UGroundV1TransformersBackend(BaseGenerationBackend):
    """Backend for UGround-V1 (Qwen2-VL based)."""

    def __init__(
        self,
        model: str = "osunlp/UGround-V1-7B",
        torch_dtype: str = "bfloat16",
        device_map: str = "auto",
        attn_implementation: Optional[str] = "flash_attention_2",
        max_new_tokens: int = 128,
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 1280 * 28 * 28,
        **kwargs: Any,
    ) -> None:
        import torch
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        dtype = getattr(torch, torch_dtype) if hasattr(torch, torch_dtype) else torch_dtype
        model_kwargs: Dict[str, Any] = {
            "torch_dtype": dtype,
            "device_map": device_map,
        }
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation
        model_kwargs.update(kwargs)

        self.model_name = model
        self.max_new_tokens = max_new_tokens
        self._processor = AutoProcessor.from_pretrained(
            model,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        self._model = Qwen2VLForConditionalGeneration.from_pretrained(
            model, **model_kwargs,
        )

    def generate(self, request: GenerationRequest) -> BackendResponse:
        if request.image is None:
            raise ValueError("UGroundV1TransformersBackend requires a prepared image")

        from qwen_vl_utils import process_vision_info

        image = request.image.payload
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": request.prompt.user_prompt},
                ],
            }
        ]

        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self._processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to(self._model.device)

        output_ids = self._model.generate(
            **inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
        )
        generated_ids = [
            out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)
        ]
        output_text = self._processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True,
        )[0]

        return BackendResponse(
            text=output_text,
            metadata={"backend": "uground_v1_transformers", "model": self.model_name},
        )

    def close(self) -> None:
        self._model = None
        self._processor = None


@BACKEND_REGISTRY.register("os_atlas_4b_transformers")
class OSAtlas4BTransformersBackend(BaseGenerationBackend):
    """Backend for OS-Atlas-Base-4B (InternVL2-4B based)."""

    def __init__(
        self,
        model: str = "OS-Copilot/OS-Atlas-Base-4B",
        torch_dtype: str = "bfloat16",
        device_map: str = "auto",
        max_new_tokens: int = 256,
        **kwargs: Any,
    ) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        dtype = getattr(torch, torch_dtype) if hasattr(torch, torch_dtype) else torch_dtype
        self.model_name = model
        self.max_new_tokens = max_new_tokens
        self._model = AutoModel.from_pretrained(
            model,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
        ).eval()
        self._tokenizer = AutoTokenizer.from_pretrained(
            model, trust_remote_code=True,
        )

    def generate(self, request: GenerationRequest) -> BackendResponse:
        if request.image is None:
            raise ValueError("OSAtlas4BTransformersBackend requires a prepared image")

        image = request.image.payload
        prompt_text = request.prompt.user_prompt
        query = f"<image>\n{prompt_text}"

        generation_config = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": False,
        }

        output_text = self._model.chat(
            self._tokenizer, image, query, generation_config,
        )

        return BackendResponse(
            text=output_text,
            metadata={"backend": "os_atlas_4b_transformers", "model": self.model_name},
        )

    def close(self) -> None:
        self._model = None
        self._tokenizer = None


@BACKEND_REGISTRY.register("os_atlas_7b_transformers")
class OSAtlas7BTransformersBackend(BaseGenerationBackend):
    """Backend for OS-Atlas-Base-7B (Qwen2-VL based)."""

    def __init__(
        self,
        model: str = "OS-Copilot/OS-Atlas-Base-7B",
        torch_dtype: str = "bfloat16",
        device_map: str = "auto",
        attn_implementation: Optional[str] = "flash_attention_2",
        max_new_tokens: int = 256,
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 1344 * 28 * 28,
        **kwargs: Any,
    ) -> None:
        import torch
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        dtype = getattr(torch, torch_dtype) if hasattr(torch, torch_dtype) else torch_dtype
        model_kwargs: Dict[str, Any] = {
            "torch_dtype": dtype,
            "device_map": device_map,
        }
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation
        model_kwargs.update(kwargs)

        self.model_name = model
        self.max_new_tokens = max_new_tokens
        self._processor = AutoProcessor.from_pretrained(
            model,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        self._model = Qwen2VLForConditionalGeneration.from_pretrained(
            model, **model_kwargs,
        )

    def generate(self, request: GenerationRequest) -> BackendResponse:
        if request.image is None:
            raise ValueError("OSAtlas7BTransformersBackend requires a prepared image")

        from qwen_vl_utils import process_vision_info

        image = request.image.payload
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": request.prompt.user_prompt},
                ],
            }
        ]

        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self._processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to(self._model.device)

        output_ids = self._model.generate(
            **inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
        )
        generated_ids = [
            out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)
        ]
        output_text = self._processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True,
        )[0]

        return BackendResponse(
            text=output_text,
            metadata={"backend": "os_atlas_7b_transformers", "model": self.model_name},
        )

    def close(self) -> None:
        self._model = None
        self._processor = None
