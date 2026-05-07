from __future__ import annotations

from typing import Any, Dict, Optional

from cuactspot.backends.base import BaseGenerationBackend
from cuactspot.registry import BACKEND_REGISTRY
from cuactspot.types import BackendResponse, GenerationRequest


@BACKEND_REGISTRY.register("gui_actor_transformers")
class GUIActorTransformersBackend(BaseGenerationBackend):
    def __init__(
        self,
        model: str = "microsoft/GUI-Actor-7B-Qwen2.5-VL",
        backbone: str = "qwen25vl",
        torch_dtype: str = "bfloat16",
        device_map: str = "auto",
        attn_implementation: Optional[str] = "flash_attention_2",
        topk: int = 3,
        use_placeholder: bool = True,
        **kwargs: Any,
    ) -> None:
        import torch
        from transformers import AutoProcessor

        try:
            from gui_actor.constants import chat_template
            from gui_actor.inference import get_prediction_region_point, inference
            if backbone == "qwen2vl":
                from gui_actor.modeling import (
                    Qwen2VLForConditionalGenerationWithPointer as ModelClass,
                )
            else:
                from gui_actor.modeling_qwen25vl import (
                    Qwen2_5_VLForConditionalGenerationWithPointer as ModelClass,
                )
            from qwen_vl_utils import process_vision_info
        except ImportError as exc:
            raise RuntimeError(
                "GUI-Actor support requires the `gui-actor` package. Install it from https://github.com/microsoft/GUI-Actor."
            ) from exc

        model_kwargs: Dict[str, Any] = {
            "torch_dtype": self._resolve_torch_dtype(torch, torch_dtype),
            "device_map": device_map,
        }
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation
        model_kwargs.update(kwargs)

        self.model_name = model
        self.topk = topk
        self.use_placeholder = use_placeholder
        self._processor = AutoProcessor.from_pretrained(model)
        self._tokenizer = self._processor.tokenizer
        self._model = ModelClass.from_pretrained(model, **model_kwargs).eval()
        self._model.config.use_cache = False
        self._inference_fn = inference
        self._chat_template = chat_template
        self._get_prediction_region_point = get_prediction_region_point
        self._process_vision_info = process_vision_info

    def generate(self, request: GenerationRequest) -> BackendResponse:
        conversation = list(request.prompt.messages)
        if not conversation:
            raise ValueError("GUIActorTransformersBackend requires prompt.messages")

        if self.use_placeholder:
            prediction = self._run_placeholder_inference(conversation)
        else:
            prediction = self._inference_fn(
                conversation,
                self._model,
                self._tokenizer,
                self._processor,
                use_placeholder=self.use_placeholder,
                topk=self.topk,
            )
        topk_points = prediction.get("topk_points") or []
        output_text = ""
        if topk_points:
            point_x, point_y = topk_points[0]
            output_text = f"({point_x}, {point_y})"

        return BackendResponse(
            text=output_text,
            raw=prediction,
            metadata={
                "backend": "gui_actor_transformers",
                "model": self.model_name,
                "output_text": prediction.get("output_text"),
                "topk_points": topk_points,
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

    def _run_placeholder_inference(
        self,
        conversation: list[dict[str, Any]],
    ) -> dict[str, Any]:
        assistant_starter = (
            "<|im_start|>assistant<|recipient|>os\n"
            "pyautogui.click(<|pointer_start|><|pointer_pad|><|pointer_end|>)"
        )
        prediction = {
            "output_text": assistant_starter,
            "n_width": None,
            "n_height": None,
            "attn_scores": None,
            "topk_points": None,
            "topk_values": None,
            "topk_points_all": None,
        }

        text = self._processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=False,
            chat_template=self._chat_template,
        )
        text += assistant_starter

        image_inputs, video_inputs = self._process_vision_info(conversation)
        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self._model.device)

        outputs = self._model(
            **inputs,
            use_cache=False,
            return_dict=True,
            output_hidden_states=True,
        )
        output_fields = self._extract_output_fields(outputs)
        hidden_states = output_fields.get("hidden_states")
        if hidden_states is None:
            return prediction

        input_ids = inputs["input_ids"][0]
        pointer_pad_mask = input_ids == self._model.config.pointer_pad_token_id
        if not pointer_pad_mask.any():
            return prediction

        image_pad_token_id = self._tokenizer.encode("<|image_pad|>")[0]
        image_mask = input_ids == image_pad_token_id

        decoder_hidden_states = hidden_states[-1][0][pointer_pad_mask]
        image_embeds = hidden_states[0][0][image_mask]
        attn_scores, _ = self._model.multi_patch_pointer_head(
            image_embeds,
            decoder_hidden_states,
        )
        prediction["attn_scores"] = attn_scores.tolist()

        _, n_height, n_width = (
            inputs["image_grid_thw"][0] // self._model.visual.spatial_merge_size
        ).tolist()
        prediction["n_width"] = n_width
        prediction["n_height"] = n_height

        _, region_points, region_scores, region_points_all = (
            self._get_prediction_region_point(
                attn_scores,
                n_width,
                n_height,
                return_all_regions=True,
                rect_center=False,
            )
        )
        prediction["topk_points"] = region_points[: self.topk]
        prediction["topk_values"] = region_scores[: self.topk]
        prediction["topk_points_all"] = region_points_all[: self.topk]
        return prediction

    def _extract_output_fields(self, output: Any) -> dict[str, Any]:
        fields: dict[str, Any] = {
            "logits": getattr(output, "logits", None),
            "past_key_values": getattr(output, "past_key_values", None),
            "hidden_states": getattr(output, "hidden_states", None),
            "attentions": getattr(output, "attentions", None),
            "rope_deltas": getattr(output, "rope_deltas", None),
        }

        packed_fields = getattr(output, "lm_loss", None)
        if isinstance(packed_fields, dict):
            for key, value in packed_fields.items():
                if key in fields and fields[key] is None:
                    fields[key] = value

        return fields