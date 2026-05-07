from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.request import Request, urlopen

from cuactspot.backends.base import BaseGenerationBackend
from cuactspot.registry import BACKEND_REGISTRY
from cuactspot.types import BackendResponse, GenerationRequest, PreparedImage


@BACKEND_REGISTRY.register("openai_compatible_api")
class OpenAICompatibleAPIBackend(BaseGenerationBackend):
    def __init__(
        self,
        endpoint: str,
        model: str,
        api_key_env: str = "OPENAI_API_KEY",
        api_key: Optional[str] = None,
        timeout: float = 120.0,
        extra_headers: Optional[Dict[str, str]] = None,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.endpoint = endpoint
        self.model = model
        self.api_key_env = api_key_env
        self.api_key = api_key
        self.timeout = timeout
        self.extra_headers = dict(extra_headers or {})
        self.extra_body = dict(extra_body or {})

    def generate(self, request: GenerationRequest) -> BackendResponse:
        api_key = self.api_key or os.getenv(self.api_key_env)
        if not api_key:
            raise ValueError(
                f"Missing API key. Set {self.api_key_env} or pass api_key in the backend config."
            )

        payload = self._build_payload(request)
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            **self.extra_headers,
        }
        http_request = Request(
            self.endpoint,
            data=body,
            headers=headers,
            method="POST",
        )
        with urlopen(http_request, timeout=self.timeout) as response:
            raw_response = json.loads(response.read().decode("utf-8"))

        text = self._extract_text(raw_response)
        return BackendResponse(
            text=text,
            raw=raw_response,
            metadata={"backend": "openai_compatible_api"},
        )

    def _build_payload(self, request: GenerationRequest) -> dict[str, Any]:
        messages = list(request.prompt.messages)
        if not messages:
            if request.prompt.system_prompt:
                messages.append(
                    {
                        "role": "system",
                        "content": request.prompt.system_prompt,
                    }
                )

            user_content: list[dict[str, Any]] = [
                {"type": "text", "text": request.prompt.user_prompt}
            ]
            if request.image is not None:
                user_content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": self._to_image_url(request.image),
                        },
                    }
                )
            messages.append({"role": "user", "content": user_content})
        else:
            messages = self._normalize_messages(messages, request)

        return {
            "model": self.model,
            "messages": messages,
            **request.generation_kwargs,
            **self.extra_body,
        }

    def _normalize_messages(
        self,
        messages: list[dict[str, Any]],
        request: GenerationRequest,
    ) -> list[dict[str, Any]]:
        normalized_messages = []
        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                normalized_messages.append(message)
                continue

            normalized_content = []
            for item in content:
                if not isinstance(item, dict):
                    normalized_content.append(item)
                    continue

                item_type = item.get("type")
                if item_type == "image":
                    image_ref = item.get("image")
                    normalized_content.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": self._coerce_image_ref(image_ref, request.image),
                            },
                        }
                    )
                    continue

                if item_type == "image_url":
                    image_url = item.get("image_url")
                    if isinstance(image_url, dict):
                        image_url = image_url.get("url")
                    normalized_content.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": self._coerce_image_ref(image_url, request.image),
                            },
                        }
                    )
                    continue

                normalized_content.append(item)

            normalized_messages.append({**message, "content": normalized_content})

        return normalized_messages

    def _coerce_image_ref(
        self,
        image_ref: Any,
        image: Optional[PreparedImage],
    ) -> str:
        if isinstance(image_ref, dict):
            image_ref = image_ref.get("url")

        if isinstance(image_ref, Path):
            return self._path_to_data_url(image_ref)

        if isinstance(image_ref, str):
            if image_ref.startswith("data:") or image_ref.startswith("http"):
                return image_ref
            try:
                image_path = Path(image_ref)
                if image_path.exists():
                    return self._path_to_data_url(image_path)
            except OSError:
                pass

        if image is not None:
            return self._to_image_url(image)

        raise TypeError(f"Unsupported image reference type: {type(image_ref)!r}")

    def _to_image_url(self, image: PreparedImage) -> str:
        payload = image.payload
        if isinstance(payload, str):
            if payload.startswith("data:") or payload.startswith("http"):
                return payload
            try:
                payload_path = Path(payload)
                if payload_path.exists():
                    return self._path_to_data_url(payload_path)
            except OSError:
                pass
            mime_type = image.metadata.get("mime_type", "image/png")
            return f"data:{mime_type};base64,{payload}"
        if isinstance(payload, Path):
            return self._path_to_data_url(payload)
        if isinstance(payload, bytes):
            mime_type = image.metadata.get("mime_type", "image/png")
            encoded = base64.b64encode(payload).decode("utf-8")
            return f"data:{mime_type};base64,{encoded}"
        raise TypeError(f"Unsupported image payload type: {type(payload)!r}")

    def _path_to_data_url(self, image_path: Path) -> str:
        mime_type = "image/png"
        with image_path.open("rb") as file_obj:
            encoded = base64.b64encode(file_obj.read()).decode("utf-8")
        return f"data:{mime_type};base64,{encoded}"

    def _extract_text(self, payload: dict[str, Any]) -> str:
        choices = payload.get("choices", [])
        if not choices:
            raise ValueError(f"No choices returned from API: {payload}")
        message = choices[0].get("message", {})
        content = message.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_chunks = [item.get("text", "") for item in content if item.get("type") == "text"]
            return "\n".join(chunk for chunk in text_chunks if chunk)
        raise ValueError(f"Unsupported API response content: {content!r}")