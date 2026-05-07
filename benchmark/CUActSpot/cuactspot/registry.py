from __future__ import annotations

from typing import Callable, Generic, TypeVar

ItemT = TypeVar("ItemT")


class Registry(Generic[ItemT]):
    def __init__(self, label: str) -> None:
        self.label = label
        self._items: dict[str, ItemT] = {}

    def register(self, name: str) -> Callable[[ItemT], ItemT]:
        def decorator(item: ItemT) -> ItemT:
            if name in self._items:
                raise ValueError(f"{self.label} '{name}' is already registered")
            self._items[name] = item
            return item

        return decorator

    def get(self, name: str) -> ItemT:
        try:
            return self._items[name]
        except KeyError as exc:
            raise KeyError(f"Unknown {self.label}: {name}") from exc

    def has(self, name: str) -> bool:
        return name in self._items

    def items(self) -> dict[str, ItemT]:
        return dict(self._items)


ADAPTER_REGISTRY: Registry[type] = Registry("adapter")
BACKEND_REGISTRY: Registry[type] = Registry("backend")
PROMPT_BUILDER_REGISTRY: Registry[type] = Registry("prompt_builder")
IMAGE_PREPROCESSOR_REGISTRY: Registry[type] = Registry("image_preprocessor")
PARSER_REGISTRY: Registry[type] = Registry("parser")
TRANSFORMER_REGISTRY: Registry[type] = Registry("transformer")