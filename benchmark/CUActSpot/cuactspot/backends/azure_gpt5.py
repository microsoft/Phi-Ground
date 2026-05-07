from __future__ import annotations

import base64
import io
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from cuactspot.backends.base import BaseGenerationBackend
from cuactspot.registry import BACKEND_REGISTRY
from cuactspot.types import BackendResponse, GenerationRequest


@BACKEND_REGISTRY.register("azure_gpt5_responses")
class AzureGPT5ResponsesBackend(BaseGenerationBackend):
    """Backend for Azure OpenAI GPT-5.4 using the Responses API with computer-use tool.

    Authentication credentials must be supplied via the run config (or environment
    variables). All Azure-tenant-specific values default to placeholders and must be
    overridden before use.
    """

    def __init__(
        self,
        azure_endpoint: str = "<YOUR_AZURE_OPENAI_ENDPOINT>",
        api_version: str = "2025-04-01-preview",
        managed_identity_client_id: str = "<YOUR_MANAGED_IDENTITY_CLIENT_ID>",
        model: str = "gpt-5.4",
        display_width: int = 1440,
        display_height: int = 900,
        reasoning_effort: str = "high",
        reasoning_summary: str = "auto",
        max_retries: int = 3,
        retry_delay: float = 10.0,
        request_timeout: float = 120.0,
        max_screenshot_rounds: int = 5,
    ) -> None:
        import os

        # Allow environment-variable overrides for sensitive fields so users do not
        # have to commit credentials to a JSON config.
        self.azure_endpoint = os.environ.get(
            "CUACTSPOT_AZURE_ENDPOINT", azure_endpoint
        )
        self.api_version = api_version
        self.managed_identity_client_id = os.environ.get(
            "CUACTSPOT_AZURE_MI_CLIENT_ID", managed_identity_client_id
        )
        self.model = model
        self.display_width = display_width
        self.display_height = display_height
        self.reasoning_effort = reasoning_effort
        self.reasoning_summary = reasoning_summary
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.request_timeout = request_timeout
        self.max_screenshot_rounds = max_screenshot_rounds
        self._client = None

    def _get_client(self):
        if self._client is None:
            from azure.identity import (
                ManagedIdentityCredential,
                get_bearer_token_provider,
            )
            from openai import AzureOpenAI

            token_provider = get_bearer_token_provider(
                ManagedIdentityCredential(
                    client_id=self.managed_identity_client_id
                ),
                "https://cognitiveservices.azure.com/.default",
            )
            self._client = AzureOpenAI(
                api_version=self.api_version,
                azure_endpoint=self.azure_endpoint,
                azure_ad_token_provider=token_provider,
                timeout=self.request_timeout,
            )
        return self._client

    def generate(self, request: GenerationRequest) -> BackendResponse:
        client = self._get_client()
        # First turn: send instruction together with the image.
        messages = self._build_messages(request)

        model_defaults = {
            "model": self.model,
            "reasoning": {
                "effort": self.reasoning_effort,
                "summary": self.reasoning_summary,
            },
        }

        tools = [{"type": "computer"}]

        response = self._api_call(client, messages, tools, model_defaults)
        actions = self._extract_actions(response)

        # Loop: keep sending the screenshot whenever the model asks for one.
        screenshot_rounds = 0
        image_url: Optional[str] = None
        if request.image is not None:
            image_url = self._to_image_data_url(request)

        for _ in range(self.max_screenshot_rounds):
            if not self._is_screenshot_only(actions):
                break
            if image_url is None:
                break
            call_id = self._get_last_call_id(response)
            if call_id is None:
                break
            screenshot_rounds += 1
            follow_up_input = [
                {
                    "call_id": call_id,
                    "type": "computer_call_output",
                    "output": {
                        "type": "input_image",
                        "image_url": image_url,
                    },
                }
            ]
            response = self._api_call(
                client,
                follow_up_input,
                tools,
                model_defaults,
                previous_response_id=response.id,
            )
            actions = self._extract_actions(response)

        text_output = self._actions_to_text(actions)
        reasoning_text = self._extract_reasoning(response)

        return BackendResponse(
            text=text_output,
            raw={
                "actions": actions,
                "reasoning": reasoning_text,
                "response_id": response.id if hasattr(response, "id") else None,
                "screenshot_rounds": screenshot_rounds,
            },
            metadata={
                "backend": "azure_gpt5_responses",
                "model": self.model,
                "display_width": self.display_width,
                "display_height": self.display_height,
                "screenshot_rounds": screenshot_rounds,
            },
        )

    def _api_call(
        self,
        client,
        input_data,
        tools,
        model_defaults,
        previous_response_id: Optional[str] = None,
    ):
        last_error = None
        for attempt in range(self.max_retries):
            try:
                kwargs: Dict[str, Any] = {
                    **model_defaults,
                    "input": input_data,
                    "tools": tools,
                    "truncation": "auto",
                }
                if previous_response_id is not None:
                    kwargs["previous_response_id"] = previous_response_id
                return client.responses.create(**kwargs)
            except Exception as exc:
                last_error = exc
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    continue
                raise RuntimeError(
                    f"Azure GPT-5.4 API failed after {self.max_retries} attempts: {last_error}"
                ) from last_error

    def _is_screenshot_only(self, actions: list[dict[str, Any]]) -> bool:
        if not actions:
            return False
        return all(a.get("type") == "screenshot" for a in actions)

    def _get_last_call_id(self, response) -> Optional[str]:
        for item in reversed(response.output):
            if item.type == "computer_call":
                call_id = getattr(item, "call_id", None)
                if call_id is not None:
                    return call_id
        return None

    def _build_messages(self, request: GenerationRequest) -> list[dict[str, Any]]:
        """Build first-turn messages with instruction and image."""
        messages: list[dict[str, Any]] = []

        system_text = request.prompt.system_prompt
        if system_text:
            messages.append({"role": "system", "content": system_text})

        user_content: list[dict[str, Any]] = []

        if request.image is not None:
            image_url = self._to_image_data_url(request)
            user_content.append(
                {
                    "type": "input_image",
                    "image_url": image_url,
                }
            )

        user_content.append(
            {
                "type": "input_text",
                "text": request.prompt.user_prompt,
            }
        )

        messages.append({"role": "user", "content": user_content})
        return messages

    def _build_text_only_messages(self, request: GenerationRequest) -> list[dict[str, Any]]:
        """Build first-turn messages with instruction only (no image)."""
        messages: list[dict[str, Any]] = []

        system_text = request.prompt.system_prompt
        if system_text:
            messages.append({"role": "system", "content": system_text})

        user_content: list[dict[str, Any]] = [
            {
                "type": "input_text",
                "text": request.prompt.user_prompt,
            }
        ]

        messages.append({"role": "user", "content": user_content})
        return messages

    def _to_image_data_url(self, request: GenerationRequest) -> str:
        image = request.image
        payload = image.payload

        if isinstance(payload, str):
            if payload.startswith("data:"):
                return payload
            mime = image.metadata.get("mime_type", "image/jpeg")
            return f"data:{mime};base64,{payload}"

        if isinstance(payload, Path):
            with payload.open("rb") as f:
                encoded = base64.b64encode(f.read()).decode("utf-8")
            return f"data:image/png;base64,{encoded}"

        if isinstance(payload, bytes):
            encoded = base64.b64encode(payload).decode("utf-8")
            mime = image.metadata.get("mime_type", "image/jpeg")
            return f"data:{mime};base64,{encoded}"

        try:
            from PIL import Image as PILImage

            if isinstance(payload, PILImage.Image):
                buf = io.BytesIO()
                payload.save(buf, format="JPEG", quality=95)
                encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
                return f"data:image/jpeg;base64,{encoded}"
        except ImportError:
            pass

        raise TypeError(f"Unsupported image payload type: {type(payload)!r}")

    def _extract_actions(self, response) -> list[dict[str, Any]]:
        actions: list[dict[str, Any]] = []
        for item in response.output:
            if item.type == "computer_call":
                action = getattr(item, "action", None)
                if action is not None:
                    if isinstance(action, dict):
                        actions.append(action)
                    elif hasattr(action, "__dict__"):
                        actions.append(self._action_obj_to_dict(action))
                    else:
                        actions.append({"raw": str(action)})
                multi_actions = getattr(item, "actions", None)
                if multi_actions is not None:
                    for a in multi_actions:
                        if isinstance(a, dict):
                            actions.append(a)
                        elif hasattr(a, "__dict__"):
                            actions.append(self._action_obj_to_dict(a))
                        else:
                            actions.append({"raw": str(a)})
        return actions

    def _action_obj_to_dict(self, action_obj) -> dict[str, Any]:
        d: dict[str, Any] = {}
        if hasattr(action_obj, "type"):
            d["type"] = action_obj.type
        for attr in ("x", "y", "button", "startX", "startY", "endX", "endY",
                      "start_x", "start_y", "end_x", "end_y",
                      "scrollX", "scrollY", "scroll_x", "scroll_y",
                      "keys", "text", "url"):
            val = getattr(action_obj, attr, None)
            if val is not None:
                d[attr] = val
        # Extract drag path: list of coordinate dicts [{x, y}, ...]
        path = getattr(action_obj, "path", None)
        if path is not None:
            serialized_path = []
            for point in path:
                if isinstance(point, dict):
                    serialized_path.append(point)
                elif hasattr(point, "__dict__"):
                    pt = {}
                    for k in ("x", "y"):
                        v = getattr(point, k, None)
                        if v is not None:
                            pt[k] = v
                    serialized_path.append(pt)
                else:
                    serialized_path.append({"raw": str(point)})
            d["path"] = serialized_path
        return d

    def _extract_reasoning(self, response) -> str:
        texts: list[str] = []
        for item in response.output:
            if item.type == "reasoning":
                for summary in getattr(item, "summary", []):
                    texts.append(getattr(summary, "text", str(summary)))
            elif item.type == "output_text":
                texts.append(getattr(item, "text", str(item)))
        return "\n".join(texts)

    def _actions_to_text(self, actions: list[dict[str, Any]]) -> str:
        parts: list[str] = []
        for action in actions:
            action_type = action.get("type", "unknown")
            if action_type in ("click", "double_click", "right_click"):
                x = action.get("x", 0)
                y = action.get("y", 0)
                parts.append(f"{action_type}(x={x}, y={y})")
            elif action_type == "drag":
                path = action.get("path")
                if isinstance(path, list) and len(path) >= 2:
                    sx, sy = path[0].get("x", 0), path[0].get("y", 0)
                    ex, ey = path[-1].get("x", 0), path[-1].get("y", 0)
                    parts.append(f"drag(start_x={sx}, start_y={sy}, end_x={ex}, end_y={ey})")
                else:
                    sx = action.get("startX") or action.get("start_x", 0)
                    sy = action.get("startY") or action.get("start_y", 0)
                    ex = action.get("endX") or action.get("end_x", 0)
                    ey = action.get("endY") or action.get("end_y", 0)
                    parts.append(f"drag(start_x={sx}, start_y={sy}, end_x={ex}, end_y={ey})")
            elif action_type == "scroll":
                x = action.get("x", 0)
                y = action.get("y", 0)
                parts.append(f"scroll(x={x}, y={y})")
            else:
                parts.append(json.dumps(action, ensure_ascii=False))
        return "\n".join(parts) if parts else json.dumps(actions, ensure_ascii=False)

    def close(self) -> None:
        self._client = None
