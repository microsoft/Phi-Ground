from cuactspot.backends.api_backend import OpenAICompatibleAPIBackend
from cuactspot.backends.azure_gpt5 import AzureGPT5ResponsesBackend
from cuactspot.backends.chat_vllm import ChatVLLMBackend
from cuactspot.backends.grounding_models import (
    OSAtlas4BTransformersBackend,
    OSAtlas7BTransformersBackend,
    SeeClickTransformersBackend,
    UGroundV1TransformersBackend,
)
from cuactspot.backends.gta1_transformers import GTA1TransformersBackend
from cuactspot.backends.gui_actor_transformers import GUIActorTransformersBackend
from cuactspot.backends.gui_owl_transformers import GUIOwlTransformersBackend
from cuactspot.backends.mock_backend import StaticResponseBackend
from cuactspot.backends.opencua_transformers import OpenCUATransformersBackend
from cuactspot.backends.phi_ground_vllm import PhiGroundVLLMBackend
from cuactspot.backends.transformers_backend import TransformersBackend
from cuactspot.backends.vllm_backend import VLLMBackend

__all__ = [
    "AzureGPT5ResponsesBackend",
    "ChatVLLMBackend",
    "GTA1TransformersBackend",
    "GUIActorTransformersBackend",
    "GUIOwlTransformersBackend",
    "OpenAICompatibleAPIBackend",
    "OpenCUATransformersBackend",
    "OSAtlas4BTransformersBackend",
    "OSAtlas7BTransformersBackend",
    "PhiGroundVLLMBackend",
    "SeeClickTransformersBackend",
    "StaticResponseBackend",
    "TransformersBackend",
    "UGroundV1TransformersBackend",
    "VLLMBackend",
]