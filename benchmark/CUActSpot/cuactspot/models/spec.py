from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

from cuactspot.config import ComponentConfig


@dataclass
class ModelSpec:
    name: str
    description: str = ""
    adapter: Optional[ComponentConfig] = None
    backend: Optional[ComponentConfig] = None
    prompt_builder: Optional[ComponentConfig] = None
    image_preprocessor: Optional[ComponentConfig] = None
    parser: Optional[ComponentConfig] = None
    transformer: Optional[ComponentConfig] = None
    system_prompt: Optional[str] = None
    generation: Dict[str, object] = field(default_factory=dict)
    metadata: Dict[str, object] = field(default_factory=dict)