from __future__ import annotations

import base64
from io import BytesIO
from typing import Any, Dict, Optional

from cuactspot.backends.base import BaseGenerationBackend
from cuactspot.registry import BACKEND_REGISTRY
from cuactspot.types import BackendResponse, GenerationRequest


@BACKEND_REGISTRY.register("opencua_transformers")
class OpenCUATransformersBackend(BaseGenerationBackend):
    def __init__(
        self,
        model: str = "xlangai/OpenCUA-7B",
        trust_remote_code: bool = True,
        torch_dtype: str = "auto",
        device_map: str = "auto",
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> None:
        import torch
        from transformers import AutoImageProcessor, AutoModel, AutoTokenizer

        model_kwargs: Dict[str, Any] = {
            "torch_dtype": self._resolve_torch_dtype(torch, torch_dtype),
            "device_map": device_map,
            "trust_remote_code": trust_remote_code,
        }
        model_kwargs.update(kwargs)

        self.model_name = model
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(
            model,
            trust_remote_code=trust_remote_code,
        )
        self._model = AutoModel.from_pretrained(model, **model_kwargs)
        self._image_processor = AutoImageProcessor.from_pretrained(
            model,
            trust_remote_code=trust_remote_code,
        )

    def generate(self, request: GenerationRequest) -> BackendResponse:
        if request.image is None:
            raise ValueError("OpenCUATransformersBackend requires a prepared image")

        image = request.image.payload
        device = self._model.device
        messages = self._build_messages(request)

        input_ids = self._tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
        image_info = self._image_processor.preprocess(images=[image])
        pixel_values = self._torch.tensor(image_info["pixel_values"]).to(
            dtype=self._torch.bfloat16,
            device=device,
        )
        image_grid_thw = self._torch.tensor(image_info["image_grid_thw"]).to(device)
        expanded_input_ids = self._expand_media_placeholders(
            input_ids,
            image_grid_thw,
        )
        input_ids_tensor = self._torch.tensor([expanded_input_ids]).to(device)
        attention_mask = self._torch.ones_like(input_ids_tensor, device=device)

        generation_kwargs: Dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
        }
        if self.temperature > 0:
            generation_kwargs["temperature"] = self.temperature
        generation_kwargs.update(request.generation_kwargs)
        if generation_kwargs.get("temperature", 0) == 0:
            generation_kwargs.pop("temperature", None)
        if generation_kwargs.get("temperature", None) not in (None, 0, 0.0, 1.0):
            generation_kwargs["do_sample"] = True

        generated_ids = self._model.generate(
            input_ids_tensor,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            attention_mask=attention_mask,
            **generation_kwargs,
        )
        prompt_length = input_ids_tensor.shape[1]
        generated_ids = generated_ids[:, prompt_length:]
        output_text = self._tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return BackendResponse(
            text=output_text,
            raw=generated_ids,
            metadata={
                "backend": "opencua_transformers",
                "model": self.model_name,
                "generation_kwargs": generation_kwargs,
            },
        )

    def close(self) -> None:
        self._model = None
        self._tokenizer = None
        self._image_processor = None

    def _build_messages(self, request: GenerationRequest) -> list[dict[str, Any]]:
        system_prompt = request.prompt.system_prompt or ""
        user_prompt = request.prompt.user_prompt
        return [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": self._to_data_uri(request.image.payload),
                    },
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]

    def _to_data_uri(self, image) -> str:
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{encoded}"

    def _expand_media_placeholders(
        self,
        input_ids: list[int],
        image_grid_thw,
    ) -> list[int]:
        placeholder_token_id = self._model.config.media_placeholder_token_id
        spatial_merge_size = getattr(
            getattr(self._model.config, "vision_config", None),
            "spatial_merge_size",
            2,
        )
        merge_factor = spatial_merge_size * spatial_merge_size
        feature_counts = [
            int(grid.prod().item()) // merge_factor
            for grid in image_grid_thw
        ]

        expanded: list[int] = []
        feature_index = 0
        for token_id in input_ids:
            if token_id != placeholder_token_id:
                expanded.append(token_id)
                continue

            if feature_index >= len(feature_counts):
                raise ValueError(
                    "OpenCUA prompt contains more media placeholders than provided images."
                )

            expanded.extend([placeholder_token_id] * feature_counts[feature_index])
            feature_index += 1

        if feature_index != len(feature_counts):
            raise ValueError(
                "OpenCUA prompt did not contain the expected number of media placeholders."
            )

        return expanded

    def _resolve_torch_dtype(self, torch_module, value: str):
        if value == "auto":
            return value
        if hasattr(torch_module, value):
            return getattr(torch_module, value)
        return value