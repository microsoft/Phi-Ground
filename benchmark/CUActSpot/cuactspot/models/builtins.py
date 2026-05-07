from __future__ import annotations

from cuactspot.config import ComponentConfig
from cuactspot.components.prompting import UI_TARS_COMPUTER_USE_TEMPLATE
from cuactspot.models.registry import register_model_spec
from cuactspot.models.spec import ModelSpec

QWEN_ACTION_SYSTEM_PROMPT = (
    "You are a GUI agent. You are given a task and a screenshot of the screen. "
    "You need to perform a series of pyautogui actions to complete the task."
)

GUI_ACTOR_GROUNDING_SYSTEM_PROMPT = (
    "You are a GUI agent. Given a screenshot of the current GUI and a human instruction, "
    "your task is to locate the screen element that corresponds to the instruction. "
    "You should output a PyAutoGUI action that performs a click on the correct position. "
    "To indicate the click location, we will use some special tokens, which is used to refer to a visual patch later. "
    "For example, you can output: pyautogui.click(<your_special_token_here>)."
)

GTA1_ACTSPOT_SYSTEM_PROMPT = (
    "You are a GUI agent. You are given a task and a screenshot of the screen. "
    "Return the complete pyautogui action sequence needed to finish the task. "
    "If the task needs multiple points, output every action in order. "
    "For drag tasks, output the start point action and the end point action. "
    "For polygon or multi-click tasks, output one pyautogui.click(x=..., y=...) per point in order. "
    "Return code only."
)

OPENCUA_ACTSPOT_SYSTEM_PROMPT = (
    "You are a GUI agent. You are given a task and a screenshot of the screen. "
    "Return only executable pyautogui code that completes the task. "
    "If the task needs multiple points, output the full action sequence in order. "
    "For drag tasks, use pyautogui.moveTo(x=..., y=...) and pyautogui.dragTo(x=..., y=..., button='left'). "
    "For polygon or multi-click tasks, output one pyautogui.click(x=..., y=...) per point in order. "
    "Do not explain anything. Do not describe the image. Output code only."
)

UI_TARS_COMPUTER_USE_SYSTEM_PROMPT = (
    "You are a GUI agent. You are given a task and your action history, with screenshots. "
    "You need to perform the next action to complete the task.\n\n"
    "## Output Format\n"
    "```\nThought: ...\nAction: ...\n```\n\n"
    "## Action Space\n"
    "click(point='<point>x1 y1</point>')\n"
    "left_double(point='<point>x1 y1</point>')\n"
    "right_single(point='<point>x1 y1</point>')\n"
    "drag(start_point='<point>x1 y1</point>', end_point='<point>x2 y2</point>')\n"
    "hotkey(key='ctrl c')\n"
    "type(content='xxx')\n"
    "scroll(point='<point>x1 y1</point>', direction='down or up or right or left')\n"
    "wait()\n"
    "finished(content='xxx')\n\n"
    "## Note\n"
    "- Use English in `Thought` part.\n"
    "- Write a small plan and finally summarize your next action (with its target element) "
    "in one sentence in `Thought` part."
)

UI_TARS_GROUNDING_SYSTEM_PROMPT = (
    "You are a GUI agent. You are given a task and your action history, with screenshots. "
    "You need to perform the next action to complete the task. \n\n"
    "## Output Format\n\n"
    "Action: ...\n\n\n"
    "## Action Space\n"
    "click(point='<point>x1 y1</point>')\n\n"
    "## User Instruction\n"
    "{instruction}"
)


register_model_spec(
    ModelSpec(
        name="static_debug",
        description="Smoke-test preset that returns a fixed coordinate string.",
        adapter=ComponentConfig(target="composable_adapter"),
        backend=ComponentConfig(target="static_response"),
        prompt_builder=ComponentConfig(target="default_prompt_builder"),
        image_preprocessor=ComponentConfig(target="identity_image_preprocessor"),
        parser=ComponentConfig(target="regex_coordinate_parser"),
        transformer=ComponentConfig(target="identity_coordinate_transformer"),
    )
)

register_model_spec(
    ModelSpec(
        name="openai_compatible_api",
        description="Single-turn API preset for OpenAI-compatible multimodal chat endpoints.",
        adapter=ComponentConfig(target="composable_adapter"),
        backend=ComponentConfig(target="openai_compatible_api"),
        prompt_builder=ComponentConfig(target="default_prompt_builder"),
        image_preprocessor=ComponentConfig(target="base64_image_preprocessor"),
        parser=ComponentConfig(target="regex_coordinate_parser"),
        transformer=ComponentConfig(target="identity_coordinate_transformer"),
    )
)

register_model_spec(
    ModelSpec(
        name="vllm_template",
        description="Template preset for future model-specific vLLM adapters.",
        adapter=ComponentConfig(target="composable_adapter"),
        backend=ComponentConfig(target="vllm"),
        prompt_builder=ComponentConfig(target="default_prompt_builder"),
        image_preprocessor=ComponentConfig(target="identity_image_preprocessor"),
        parser=ComponentConfig(target="regex_coordinate_parser"),
        transformer=ComponentConfig(target="identity_coordinate_transformer"),
    )
)

register_model_spec(
    ModelSpec(
        name="transformers_template",
        description="Template preset for future model-specific Hugging Face adapters.",
        adapter=ComponentConfig(target="composable_adapter"),
        backend=ComponentConfig(target="transformers"),
        prompt_builder=ComponentConfig(target="default_prompt_builder"),
        image_preprocessor=ComponentConfig(target="identity_image_preprocessor"),
        parser=ComponentConfig(target="regex_coordinate_parser"),
        transformer=ComponentConfig(target="identity_coordinate_transformer"),
    )
)

register_model_spec(
    ModelSpec(
        name="phi_ground_vllm",
        description="Phi-Ground preset using the official fixed-resolution preprocessing and vLLM inference path.",
        adapter=ComponentConfig(target="composable_adapter"),
        backend=ComponentConfig(
            target="phi_ground_vllm",
            kwargs={
                "max_model_len": 4096,
            },
        ),
        prompt_builder=ComponentConfig(target="phi_ground_actspot_prompt_builder"),
        image_preprocessor=ComponentConfig(target="phi_ground_image_preprocessor"),
        parser=ComponentConfig(target="phi_ground_coordinate_parser"),
        transformer=ComponentConfig(target="phi_ground_relative_transformer"),
        generation={
            "temperature": 0.0,
            "max_tokens": 64,
        },
    )
)

register_model_spec(
    ModelSpec(
        name="phi_ground_4b_16c_vllm",
        description="Phi-Ground-4B-16C preset reusing the Phi-Ground pipeline with a 1680x1008 preprocessing canvas.",
        adapter=ComponentConfig(target="composable_adapter"),
        backend=ComponentConfig(
            target="phi_ground_vllm",
            kwargs={
                "max_model_len": 8192,
            },
        ),
        prompt_builder=ComponentConfig(target="phi_ground_actspot_prompt_builder"),
        image_preprocessor=ComponentConfig(
            target="phi_ground_image_preprocessor",
            kwargs={
                "target_width": 336 * 5,
                "target_height": 336 * 3,
            },
        ),
        parser=ComponentConfig(target="phi_ground_coordinate_parser"),
        transformer=ComponentConfig(target="phi_ground_relative_transformer"),
        generation={
            "temperature": 0.0,
            "max_tokens": 64,
        },
    )
)

register_model_spec(
    ModelSpec(
        name="opencua_7b_vllm",
        description="OpenCUA-7B grounding preset using direct vLLM inference with the official chat-template path.",
        adapter=ComponentConfig(target="composable_adapter"),
        backend=ComponentConfig(
            target="chat_vllm",
            kwargs={
                "model": "xlangai/OpenCUA-7B",
                "trust_remote_code": True,
                "rewrite_media_placeholders": True,
                "max_model_len": 8192,
            },
        ),
        prompt_builder=ComponentConfig(target="chat_messages_prompt_builder"),
        image_preprocessor=ComponentConfig(target="pil_image_preprocessor"),
        parser=ComponentConfig(target="action_coordinate_parser"),
        transformer=ComponentConfig(target="smart_resize_coordinate_transformer"),
        system_prompt=QWEN_ACTION_SYSTEM_PROMPT,
        generation={
            "temperature": 0.0,
            "max_tokens": 512,
        },
    )
)

register_model_spec(
    ModelSpec(
        name="opencua_7b_vllm_api",
        description="OpenCUA-7B grounding preset using the official vLLM server path through an OpenAI-compatible API.",
        adapter=ComponentConfig(target="composable_adapter"),
        backend=ComponentConfig(
            target="openai_compatible_api",
            kwargs={
                "endpoint": "http://127.0.0.1:8000/v1/chat/completions",
                "model": "opencua-7b",
                "api_key": "EMPTY",
                "timeout": 300.0,
            },
        ),
        prompt_builder=ComponentConfig(target="chat_messages_prompt_builder"),
        image_preprocessor=ComponentConfig(target="identity_image_preprocessor"),
        parser=ComponentConfig(target="action_coordinate_parser"),
        transformer=ComponentConfig(target="smart_resize_coordinate_transformer"),
        system_prompt=OPENCUA_ACTSPOT_SYSTEM_PROMPT,
        generation={
            "temperature": 0.0,
            "max_tokens": 512,
        },
    )
)

register_model_spec(
    ModelSpec(
        name="opencua_32b_vllm",
        description="OpenCUA-32B grounding preset using direct vLLM inference with tensor parallel size 4.",
        adapter=ComponentConfig(target="composable_adapter"),
        backend=ComponentConfig(
            target="chat_vllm",
            kwargs={
                "model": "xlangai/OpenCUA-32B",
                "trust_remote_code": True,
                "rewrite_media_placeholders": True,
                "tensor_parallel_size": 4,
                "max_model_len": 8192,
            },
        ),
        prompt_builder=ComponentConfig(target="chat_messages_prompt_builder"),
        image_preprocessor=ComponentConfig(target="pil_image_preprocessor"),
        parser=ComponentConfig(target="action_coordinate_parser"),
        transformer=ComponentConfig(target="smart_resize_coordinate_transformer"),
        system_prompt=OPENCUA_ACTSPOT_SYSTEM_PROMPT,
        generation={
            "temperature": 0.0,
            "max_tokens": 512,
        },
    )
)

register_model_spec(
    ModelSpec(
        name="opencua_32b_vllm_api",
        description="OpenCUA-32B grounding preset using the official vLLM server path through an OpenAI-compatible API.",
        adapter=ComponentConfig(target="composable_adapter"),
        backend=ComponentConfig(
            target="openai_compatible_api",
            kwargs={
                "endpoint": "http://127.0.0.1:8000/v1/chat/completions",
                "model": "opencua-32b",
                "api_key": "EMPTY",
                "timeout": 300.0,
            },
        ),
        prompt_builder=ComponentConfig(target="chat_messages_prompt_builder"),
        image_preprocessor=ComponentConfig(target="identity_image_preprocessor"),
        parser=ComponentConfig(target="action_coordinate_parser"),
        transformer=ComponentConfig(target="smart_resize_coordinate_transformer"),
        system_prompt=OPENCUA_ACTSPOT_SYSTEM_PROMPT,
        generation={
            "temperature": 0.0,
            "max_tokens": 512,
        },
    )
)

register_model_spec(
    ModelSpec(
        name="gta1_7b_vllm",
        description="GTA1-7B grounding preset using direct vLLM inference with the official slow tokenizer path.",
        adapter=ComponentConfig(target="composable_adapter"),
        backend=ComponentConfig(
            target="chat_vllm",
            kwargs={
                "model": "Salesforce/GTA1-7B",
                "trust_remote_code": True,
                "tokenizer_mode": "slow",
                "rewrite_media_placeholders": True,
                "max_model_len": 32768,
            },
        ),
        prompt_builder=ComponentConfig(target="chat_messages_prompt_builder"),
        image_preprocessor=ComponentConfig(target="pil_image_preprocessor"),
        parser=ComponentConfig(target="action_coordinate_parser"),
        transformer=ComponentConfig(target="smart_resize_coordinate_transformer"),
        system_prompt=QWEN_ACTION_SYSTEM_PROMPT,
        generation={
            "temperature": 0.0,
            "max_tokens": 512,
        },
    )
)

register_model_spec(
    ModelSpec(
        name="gta1_32b_vllm",
        description="GTA1-32B grounding preset using direct vLLM inference with tensor parallel size 4.",
        adapter=ComponentConfig(target="composable_adapter"),
        backend=ComponentConfig(
            target="chat_vllm",
            kwargs={
                "model": "Salesforce/GTA1-32B",
                "trust_remote_code": True,
                "tokenizer_mode": "slow",
                "rewrite_media_placeholders": True,
                "tensor_parallel_size": 4,
                "max_model_len": 32768,
            },
        ),
        prompt_builder=ComponentConfig(target="chat_messages_prompt_builder"),
        image_preprocessor=ComponentConfig(target="pil_image_preprocessor"),
        parser=ComponentConfig(target="action_coordinate_parser"),
        transformer=ComponentConfig(target="smart_resize_coordinate_transformer"),
        system_prompt=QWEN_ACTION_SYSTEM_PROMPT,
        generation={
            "temperature": 0.0,
            "max_tokens": 512,
        },
    )
)

register_model_spec(
    ModelSpec(
        name="gui_actor_7b_qwen25_transformers",
        description="GUI-Actor-7B Qwen2.5-VL preset using the official pointer-head inference path.",
        adapter=ComponentConfig(target="composable_adapter"),
        backend=ComponentConfig(
            target="gui_actor_transformers",
            kwargs={
                "model": "microsoft/GUI-Actor-7B-Qwen2.5-VL",
                "backbone": "qwen25vl",
                "topk": 3,
                "use_placeholder": True,
            },
        ),
        prompt_builder=ComponentConfig(target="chat_messages_prompt_builder"),
        image_preprocessor=ComponentConfig(target="identity_image_preprocessor"),
        parser=ComponentConfig(target="regex_coordinate_parser"),
        transformer=ComponentConfig(
            target="coordinate_space_transformer",
            kwargs={
                "normalized": True,
                "clip_to_image": True,
            },
        ),
        system_prompt=GUI_ACTOR_GROUNDING_SYSTEM_PROMPT,
    )
)

register_model_spec(
    ModelSpec(
        name="ui_tars_7b_api",
        description="UI-TARS-7B-SFT grounding preset through an OpenAI-compatible local vLLM server.",
        adapter=ComponentConfig(target="composable_adapter"),
        backend=ComponentConfig(
            target="openai_compatible_api",
            kwargs={
                "endpoint": "http://127.0.0.1:8000/v1/chat/completions",
                "model": "ui-tars-7b",
                "api_key": "EMPTY",
                "timeout": 300.0,
            },
        ),
        prompt_builder=ComponentConfig(target="chat_messages_prompt_builder"),
        image_preprocessor=ComponentConfig(target="identity_image_preprocessor"),
        parser=ComponentConfig(target="ui_tars_coordinate_parser"),
        transformer=ComponentConfig(
            target="smart_resize_coordinate_transformer",
            kwargs={
                "min_pixels": 3136,
                "max_pixels": 2116800,
            },
        ),
        system_prompt=UI_TARS_COMPUTER_USE_SYSTEM_PROMPT,
        generation={
            "temperature": 0.0,
            "max_tokens": 256,
            "frequency_penalty": 1,
        },
    )
)

register_model_spec(
    ModelSpec(
        name="ui_tars_7b_grounding_api",
        description="UI-TARS-7B-SFT with GROUNDING prompt through an OpenAI-compatible local vLLM server.",
        adapter=ComponentConfig(target="composable_adapter"),
        backend=ComponentConfig(
            target="openai_compatible_api",
            kwargs={
                "endpoint": "http://127.0.0.1:8000/v1/chat/completions",
                "model": "ui-tars-7b",
                "api_key": "EMPTY",
                "timeout": 300.0,
            },
        ),
        prompt_builder=ComponentConfig(
            target="default_prompt_builder",
            kwargs={
                "template": UI_TARS_GROUNDING_SYSTEM_PROMPT,
            },
        ),
        image_preprocessor=ComponentConfig(target="identity_image_preprocessor"),
        parser=ComponentConfig(target="ui_tars_coordinate_parser"),
        transformer=ComponentConfig(
            target="smart_resize_coordinate_transformer",
            kwargs={
                "min_pixels": 3136,
                "max_pixels": 2116800,
            },
        ),
        generation={
            "temperature": 0.0,
            "max_tokens": 256,
            "frequency_penalty": 1,
        },
    )
)

register_model_spec(
    ModelSpec(
        name="ui_tars_1_5_7b_api",
        description="UI-TARS-1.5-7B grounding preset through an OpenAI-compatible local vLLM server.",
        adapter=ComponentConfig(target="composable_adapter"),
        backend=ComponentConfig(
            target="openai_compatible_api",
            kwargs={
                "endpoint": "http://127.0.0.1:8000/v1/chat/completions",
                "model": "ui-tars-1.5-7b",
                "api_key": "EMPTY",
                "timeout": 300.0,
            },
        ),
        prompt_builder=ComponentConfig(target="chat_messages_prompt_builder"),
        image_preprocessor=ComponentConfig(target="identity_image_preprocessor"),
        parser=ComponentConfig(target="ui_tars_coordinate_parser"),
        transformer=ComponentConfig(target="smart_resize_coordinate_transformer"),
        system_prompt=UI_TARS_COMPUTER_USE_SYSTEM_PROMPT,
        generation={
            "temperature": 0.0,
            "max_tokens": 256,
            "frequency_penalty": 1,
        },
    )
)

register_model_spec(
    ModelSpec(
        name="ui_tars_api",
        description="UI-TARS action-space preset through an OpenAI-compatible local server.",
        adapter=ComponentConfig(target="composable_adapter"),
        backend=ComponentConfig(
            target="openai_compatible_api",
            kwargs={
                "endpoint": "http://127.0.0.1:8000/v1/chat/completions",
                "model": "ui-tars",
                "api_key": "EMPTY",
            },
        ),
        prompt_builder=ComponentConfig(target="chat_messages_prompt_builder"),
        image_preprocessor=ComponentConfig(target="identity_image_preprocessor"),
        parser=ComponentConfig(target="ui_tars_coordinate_parser"),
        transformer=ComponentConfig(target="smart_resize_coordinate_transformer"),
        system_prompt=UI_TARS_COMPUTER_USE_SYSTEM_PROMPT,
        generation={
            "temperature": 0.0,
            "max_tokens": 256,
            "frequency_penalty": 1,
        },
    )
)

register_model_spec(
    ModelSpec(
        name="opencua_7b_transformers",
        description="OpenCUA-7B grounding preset using the official direct Hugging Face inference path.",
        adapter=ComponentConfig(target="composable_adapter"),
        backend=ComponentConfig(
            target="opencua_transformers",
            kwargs={
                "model": "xlangai/OpenCUA-7B",
                "trust_remote_code": True,
            },
        ),
        prompt_builder=ComponentConfig(target="chat_messages_prompt_builder"),
        image_preprocessor=ComponentConfig(target="pil_image_preprocessor"),
        parser=ComponentConfig(target="action_coordinate_parser"),
        transformer=ComponentConfig(target="smart_resize_coordinate_transformer"),
        system_prompt=OPENCUA_ACTSPOT_SYSTEM_PROMPT,
        generation={
            "max_new_tokens": 128,
        },
    )
)

register_model_spec(
    ModelSpec(
        name="opencua_32b_transformers",
        description="OpenCUA-32B grounding preset using the official direct Hugging Face inference path.",
        adapter=ComponentConfig(target="composable_adapter"),
        backend=ComponentConfig(
            target="opencua_transformers",
            kwargs={
                "model": "xlangai/OpenCUA-32B",
                "trust_remote_code": True,
            },
        ),
        prompt_builder=ComponentConfig(target="chat_messages_prompt_builder"),
        image_preprocessor=ComponentConfig(target="pil_image_preprocessor"),
        parser=ComponentConfig(target="action_coordinate_parser"),
        transformer=ComponentConfig(target="smart_resize_coordinate_transformer"),
        system_prompt=OPENCUA_ACTSPOT_SYSTEM_PROMPT,
        generation={
            "max_new_tokens": 128,
        },
    )
)

register_model_spec(
    ModelSpec(
        name="gta1_7b_transformers",
        description="GTA1-7B grounding preset using the official direct Hugging Face inference path.",
        adapter=ComponentConfig(target="composable_adapter"),
        backend=ComponentConfig(
            target="gta1_transformers",
            kwargs={
                "model": "Salesforce/GTA1-7B",
            },
        ),
        prompt_builder=ComponentConfig(target="plain_task_prompt_builder"),
        image_preprocessor=ComponentConfig(target="pil_image_preprocessor"),
        parser=ComponentConfig(target="action_coordinate_parser"),
        transformer=ComponentConfig(target="smart_resize_coordinate_transformer"),
        system_prompt=GTA1_ACTSPOT_SYSTEM_PROMPT,
        generation={
            "temperature": 0.0,
            "max_new_tokens": 128,
        },
    )
)

register_model_spec(
    ModelSpec(
        name="gta1_32b_transformers",
        description="GTA1-32B grounding preset using the official direct Hugging Face inference path.",
        adapter=ComponentConfig(target="composable_adapter"),
        backend=ComponentConfig(
            target="gta1_transformers",
            kwargs={
                "model": "Salesforce/GTA1-32B",
            },
        ),
        prompt_builder=ComponentConfig(target="plain_task_prompt_builder"),
        image_preprocessor=ComponentConfig(target="pil_image_preprocessor"),
        parser=ComponentConfig(target="action_coordinate_parser"),
        transformer=ComponentConfig(target="smart_resize_coordinate_transformer"),
        system_prompt=GTA1_ACTSPOT_SYSTEM_PROMPT,
        generation={
            "temperature": 0.0,
            "max_new_tokens": 128,
        },
    )
)

UI_VENUS_GROUND_PROMPT_TEMPLATE = (
    "Outline the position corresponding to the instruction: {instruction}. "
    "The output should be only [x1,y1,x2,y2]."
)

register_model_spec(
    ModelSpec(
        name="ui_venus_ground_7b_api",
        description="UI-Venus-Ground-7B grounding preset through an OpenAI-compatible local vLLM server.",
        adapter=ComponentConfig(target="composable_adapter"),
        backend=ComponentConfig(
            target="openai_compatible_api",
            kwargs={
                "endpoint": "http://127.0.0.1:8000/v1/chat/completions",
                "model": "ui-venus-ground-7b",
                "api_key": "EMPTY",
                "timeout": 300.0,
            },
        ),
        prompt_builder=ComponentConfig(
            target="chat_messages_prompt_builder",
            kwargs={
                "user_template": UI_VENUS_GROUND_PROMPT_TEMPLATE,
            },
        ),
        image_preprocessor=ComponentConfig(target="identity_image_preprocessor"),
        parser=ComponentConfig(target="bbox_center_coordinate_parser"),
        transformer=ComponentConfig(target="smart_resize_coordinate_transformer"),
        generation={
            "temperature": 0.0,
            "max_tokens": 128,
        },
    )
)

MAI_UI_GROUNDING_SYSTEM_PROMPT = (
    "You are a GUI grounding agent. \n\n"
    "## Task\n"
    "Given a screenshot and the user's grounding instruction. "
    "Your task is to accurately locate a UI element based on the user's instructions.\n"
    "First, you should carefully examine the screenshot and analyze the user's instructions, "
    "translate the user's instruction into a effective reasoning process, "
    "and then provide the final coordinate.\n"
    "## Output Format\n"
    "Return a json object with a reasoning process in <grounding_think></grounding_think> tags, "
    "a [x,y] format coordinate within <answer></answer> XML tags:\n"
    "<grounding_think>...</grounding_think>\n"
    "<answer>\n"
    '{"coordinate": [x,y]}\n'
    "</answer>"
)

register_model_spec(
    ModelSpec(
        name="mai_ui_8b_api",
        description="MAI-UI-8B grounding preset through an OpenAI-compatible local vLLM server.",
        adapter=ComponentConfig(target="composable_adapter"),
        backend=ComponentConfig(
            target="openai_compatible_api",
            kwargs={
                "endpoint": "http://127.0.0.1:8000/v1/chat/completions",
                "model": "mai-ui-8b",
                "api_key": "EMPTY",
                "timeout": 300.0,
            },
        ),
        prompt_builder=ComponentConfig(target="chat_messages_prompt_builder"),
        image_preprocessor=ComponentConfig(target="identity_image_preprocessor"),
        parser=ComponentConfig(target="mai_ui_grounding_parser"),
        transformer=ComponentConfig(
            target="coordinate_space_transformer",
            kwargs={
                "source_width": 999,
                "source_height": 999,
                "clip_to_image": True,
            },
        ),
        system_prompt=MAI_UI_GROUNDING_SYSTEM_PROMPT,
        generation={
            "temperature": 0.0,
            "max_tokens": 2048,
        },
    )
)

register_model_spec(
    ModelSpec(
        name="mai_ui_2b_api",
        description="MAI-UI-2B grounding preset through an OpenAI-compatible local vLLM server.",
        adapter=ComponentConfig(target="composable_adapter"),
        backend=ComponentConfig(
            target="openai_compatible_api",
            kwargs={
                "endpoint": "http://127.0.0.1:8000/v1/chat/completions",
                "model": "mai-ui-2b",
                "api_key": "EMPTY",
                "timeout": 300.0,
            },
        ),
        prompt_builder=ComponentConfig(target="chat_messages_prompt_builder"),
        image_preprocessor=ComponentConfig(target="identity_image_preprocessor"),
        parser=ComponentConfig(target="mai_ui_grounding_parser"),
        transformer=ComponentConfig(
            target="coordinate_space_transformer",
            kwargs={
                "source_width": 999,
                "source_height": 999,
                "clip_to_image": True,
            },
        ),
        system_prompt=MAI_UI_GROUNDING_SYSTEM_PROMPT,
        generation={
            "temperature": 0.0,
            "max_tokens": 2048,
        },
    )
)

register_model_spec(
    ModelSpec(
        name="gta1_32b_api",
        description="GTA1-32B grounding preset through an OpenAI-compatible local vLLM server with tp=4.",
        adapter=ComponentConfig(target="composable_adapter"),
        backend=ComponentConfig(
            target="openai_compatible_api",
            kwargs={
                "endpoint": "http://127.0.0.1:8000/v1/chat/completions",
                "model": "gta1-32b",
                "api_key": "EMPTY",
                "timeout": 300.0,
            },
        ),
        prompt_builder=ComponentConfig(target="chat_messages_prompt_builder"),
        image_preprocessor=ComponentConfig(target="identity_image_preprocessor"),
        parser=ComponentConfig(target="action_coordinate_parser"),
        transformer=ComponentConfig(target="smart_resize_coordinate_transformer"),
        system_prompt=QWEN_ACTION_SYSTEM_PROMPT,
        generation={
            "temperature": 0.0,
            "max_tokens": 512,
        },
    )
)

register_model_spec(
    ModelSpec(
        name="ui_tars_72b_api",
        description="UI-TARS-72B-SFT grounding preset through an OpenAI-compatible local vLLM server with tp=4.",
        adapter=ComponentConfig(target="composable_adapter"),
        backend=ComponentConfig(
            target="openai_compatible_api",
            kwargs={
                "endpoint": "http://127.0.0.1:8000/v1/chat/completions",
                "model": "ui-tars-72b",
                "api_key": "EMPTY",
                "timeout": 300.0,
            },
        ),
        prompt_builder=ComponentConfig(target="chat_messages_prompt_builder"),
        image_preprocessor=ComponentConfig(target="identity_image_preprocessor"),
        parser=ComponentConfig(target="ui_tars_coordinate_parser"),
        transformer=ComponentConfig(
            target="smart_resize_coordinate_transformer",
            kwargs={
                "min_pixels": 3136,
                "max_pixels": 2116800,
            },
        ),
        system_prompt=UI_TARS_COMPUTER_USE_SYSTEM_PROMPT,
        generation={
            "temperature": 0.0,
            "max_tokens": 256,
            "frequency_penalty": 1,
        },
    )
)

register_model_spec(
    ModelSpec(
        name="ui_venus_ground_72b_api",
        description="UI-Venus-Ground-72B grounding preset through an OpenAI-compatible local vLLM server with tp=4.",
        adapter=ComponentConfig(target="composable_adapter"),
        backend=ComponentConfig(
            target="openai_compatible_api",
            kwargs={
                "endpoint": "http://127.0.0.1:8000/v1/chat/completions",
                "model": "ui-venus-ground-72b",
                "api_key": "EMPTY",
                "timeout": 300.0,
            },
        ),
        prompt_builder=ComponentConfig(
            target="chat_messages_prompt_builder",
            kwargs={
                "user_template": UI_VENUS_GROUND_PROMPT_TEMPLATE,
            },
        ),
        image_preprocessor=ComponentConfig(target="identity_image_preprocessor"),
        parser=ComponentConfig(target="bbox_center_coordinate_parser"),
        transformer=ComponentConfig(target="smart_resize_coordinate_transformer"),
        generation={
            "temperature": 0.0,
            "max_tokens": 128,
        },
    )
)

register_model_spec(
    ModelSpec(
        name="ui_venus_ground_72b_transformers",
        description="UI-Venus-Ground-72B grounding preset using the Hugging Face inference path with device_map=auto.",
        adapter=ComponentConfig(target="composable_adapter"),
        backend=ComponentConfig(
            target="gta1_transformers",
            kwargs={
                "model": "inclusionAI/UI-Venus-Ground-72B",
                "max_new_tokens": 128,
                "min_pixels": 2000000,
                "max_pixels": 4800000,
                "default_system_prompt": "",
                "rewrite_media_placeholders": False,
            },
        ),
        prompt_builder=ComponentConfig(
            target="default_prompt_builder",
            kwargs={
                "template": UI_VENUS_GROUND_PROMPT_TEMPLATE,
            },
        ),
        image_preprocessor=ComponentConfig(target="pil_image_preprocessor"),
        parser=ComponentConfig(target="bbox_center_coordinate_parser"),
        transformer=ComponentConfig(
            target="smart_resize_coordinate_transformer",
            kwargs={
                "min_pixels": 2000000,
                "max_pixels": 4800000,
            },
        ),
        generation={
            "temperature": 0.0,
            "max_new_tokens": 128,
        },
    )
)

GPT5_ACTSPOT_SYSTEM_PROMPT = (
    "You are a precise GUI grounding agent. You are given a screenshot and a mouse operation instruction.\n"
    "Analyze the screenshot carefully and perform the exact mouse action described in the instruction.\n"
    "- For single click tasks, click precisely on the target element or position.\n"
    "- For drag tasks, drag from the exact start position to the exact end position.\n"
    "- If the task requires multiple clicks or a sequence of operations, perform all of them in order.\n"
    "Perform all required actions immediately in a single response."
)

register_model_spec(
    ModelSpec(
        name="gpt5_4_azure",
        description="GPT-5.4 grounding preset using Azure OpenAI Responses API with computer-use tool.",
        adapter=ComponentConfig(target="composable_adapter"),
        backend=ComponentConfig(
            target="azure_gpt5_responses",
            kwargs={
                "azure_endpoint": "<YOUR_AZURE_OPENAI_ENDPOINT>",
                "api_version": "2025-04-01-preview",
                "managed_identity_client_id": "<YOUR_MANAGED_IDENTITY_CLIENT_ID>",
                "model": "gpt-5.4",
                "display_width": 1440,
                "display_height": 900,
                "reasoning_effort": "high",
                "reasoning_summary": "auto",
            },
        ),
        prompt_builder=ComponentConfig(target="plain_task_prompt_builder"),
        image_preprocessor=ComponentConfig(
            target="resize_base64_image_preprocessor",
            kwargs={
                "target_width": 1440,
                "target_height": 900,
            },
        ),
        parser=ComponentConfig(target="gpt5_computer_call_parser"),
        transformer=ComponentConfig(
            target="coordinate_space_transformer",
            kwargs={
                "source_width": 1229,
                "source_height": 768,
                "clip_to_image": True,
            },
        ),
        system_prompt=GPT5_ACTSPOT_SYSTEM_PROMPT,
    )
)

PHI_GROUND_ANYTHING_PROMPT_TEMPLATE = "<|user|> \n{instruction}<|image_1|> \n<|end|> \n<|assistant|>"

register_model_spec(
    ModelSpec(
        name="infigui_r1_3b_transformers",
        description="InfiGUI-R1-3B grounding preset using HuggingFace Transformers inference.",
        adapter=ComponentConfig(target="composable_adapter"),
        backend=ComponentConfig(
            target="gta1_transformers",
            kwargs={
                "model": "InfiX-ai/InfiGUI-R1-3B",
                "max_pixels": 4390400,
                "rewrite_media_placeholders": False,
                "apply_chat_template_with": "processor",
                "default_system_prompt": "",
            },
        ),
        prompt_builder=ComponentConfig(
            target="infigui_grounding_prompt_builder",
            kwargs={
                "max_pixels": 4390400,
            },
        ),
        image_preprocessor=ComponentConfig(target="pil_image_preprocessor"),
        parser=ComponentConfig(target="infigui_grounding_parser"),
        transformer=ComponentConfig(
            target="smart_resize_coordinate_transformer",
            kwargs={
                "max_pixels": 4390400,
            },
        ),
        generation={
            "temperature": 0.0,
            "max_new_tokens": 1024,
        },
    )
)

register_model_spec(
    ModelSpec(
        name="se_gui_7b_transformers",
        description="SE-GUI-7B grounding preset using HuggingFace Transformers inference (Qwen2.5-VL fine-tune).",
        adapter=ComponentConfig(target="composable_adapter"),
        backend=ComponentConfig(
            target="gta1_transformers",
            kwargs={
                "model": "XinBB/SE-GUI-7B",
                "max_new_tokens": 512,
                "rewrite_media_placeholders": False,
                "apply_chat_template_with": "processor",
            },
        ),
        prompt_builder=ComponentConfig(target="chat_messages_prompt_builder"),
        image_preprocessor=ComponentConfig(target="pil_image_preprocessor"),
        parser=ComponentConfig(target="action_coordinate_parser"),
        transformer=ComponentConfig(target="smart_resize_coordinate_transformer"),
        system_prompt=QWEN_ACTION_SYSTEM_PROMPT,
        generation={
            "temperature": 0.0,
            "max_new_tokens": 512,
        },
    )
)

register_model_spec(
    ModelSpec(
        name="gui_g2_7b_transformers",
        description="GUI-G2-7B grounding preset using HuggingFace Transformers (Qwen2.5-VL fine-tune, bbox output).",
        adapter=ComponentConfig(target="composable_adapter"),
        backend=ComponentConfig(
            target="gta1_transformers",
            kwargs={
                "model": "inclusionAI/GUI-G2-7B",
                "max_new_tokens": 128,
                "rewrite_media_placeholders": False,
                "apply_chat_template_with": "processor",
                "default_system_prompt": "",
            },
        ),
        prompt_builder=ComponentConfig(
            target="default_prompt_builder",
            kwargs={
                "template": UI_VENUS_GROUND_PROMPT_TEMPLATE,
            },
        ),
        image_preprocessor=ComponentConfig(target="pil_image_preprocessor"),
        parser=ComponentConfig(target="bbox_center_coordinate_parser"),
        transformer=ComponentConfig(target="smart_resize_coordinate_transformer"),
        generation={
            "temperature": 0.0,
            "max_new_tokens": 128,
        },
    )
)

INFIGUI_G1_SYSTEM_PROMPT = (
    "You FIRST think about the reasoning process as an internal monologue "
    "and then provide the final answer.\n"
    "The reasoning process MUST BE enclosed within <think> </think> tags."
)

register_model_spec(
    ModelSpec(
        name="infigui_g1_7b_transformers",
        description="InfiGUI-G1-7B grounding preset using HuggingFace Transformers with thinking output.",
        adapter=ComponentConfig(target="composable_adapter"),
        backend=ComponentConfig(
            target="gta1_transformers",
            kwargs={
                "model": "InfiX-ai/InfiGUI-G1-7B",
                "max_new_tokens": 512,
                "max_pixels": 4390400,
                "rewrite_media_placeholders": False,
                "apply_chat_template_with": "processor",
            },
        ),
        prompt_builder=ComponentConfig(
            target="infigui_g1_grounding_prompt_builder",
            kwargs={
                "max_pixels": 4390400,
            },
        ),
        image_preprocessor=ComponentConfig(target="pil_image_preprocessor"),
        parser=ComponentConfig(target="infigui_g1_grounding_parser"),
        transformer=ComponentConfig(
            target="smart_resize_coordinate_transformer",
            kwargs={
                "max_pixels": 4390400,
            },
        ),
        system_prompt=INFIGUI_G1_SYSTEM_PROMPT,
        generation={
            "temperature": 0.0,
            "max_new_tokens": 512,
        },
    )
)

GUI_OWL_SYSTEM_PROMPT = (
    "# Tools\n\n"
    "You may call one or more functions to assist with the user query.\n\n"
    "You are provided with function signatures within <tools></tools> XML tags:\n"
    "<tools>\n"
    '{"type": "function", "function": {"name": "computer_use", "description": "Use a mouse '
    "to interact with a computer.\\n"
    "* The screen's resolution is 1000x1000.\\n"
    "* Make sure to click any buttons, links, icons, etc with the cursor tip in the center "
    "of the element. Don't click boxes on their edges unless asked.\\n"
    "* don't use any other computer use tool like type, key, scroll, left_click_drag and so on.\\n"
    "* you can only use the left_click and mouse_move action to interact with the computer. "
    'if you can\'t find the element, you should terminate the task and report the failure.", '
    '"parameters": {"properties": {"action": {"description": "The action to perform. '
    "The available actions are:\\n"
    "* `mouse_move`: Move the cursor to a specified (x, y) pixel coordinate on the screen.\\n"
    "* `left_click`: Click the left mouse button with coordinate (x, y) pixel coordinate "
    'on the screen.", "enum": ["mouse_move", "left_click"], "type": "string"}, '
    '"coordinate": {"description": "(x, y): The x (pixels from the left edge) and y '
    "(pixels from the top edge) coordinates to move the mouse to. Required only by "
    '`action=mouse_move` and `action=left_click`.", "type": "array"}}, '
    '"required": ["action"], "type": "object"}}}\n'
    "</tools>\n\n"
    "For each function call, return a json object with function name and arguments "
    "within <tool_call></tool_call> XML tags:\n"
    "<tool_call>\n"
    '{"name": <function-name>, "arguments": <args-json-object>}\n'
    "</tool_call>\n"
)

register_model_spec(
    ModelSpec(
        name="gui_owl_8b_instruct_transformers",
        description="GUI-Owl-1.5-8B-Instruct grounding preset using HuggingFace Transformers.",
        adapter=ComponentConfig(target="composable_adapter"),
        backend=ComponentConfig(
            target="gui_owl_transformers",
            kwargs={
                "model": "mPLUG/GUI-Owl-1.5-8B-Instruct",
                "max_new_tokens": 2048,
            },
        ),
        prompt_builder=ComponentConfig(target="plain_task_prompt_builder"),
        image_preprocessor=ComponentConfig(target="pil_image_preprocessor"),
        parser=ComponentConfig(target="mai_ui_grounding_parser"),
        transformer=ComponentConfig(
            target="coordinate_space_transformer",
            kwargs={
                "source_width": 1000,
                "source_height": 1000,
                "clip_to_image": True,
            },
        ),
        system_prompt=GUI_OWL_SYSTEM_PROMPT,
        generation={
            "temperature": 0.0,
            "max_new_tokens": 2048,
        },
    )
)

register_model_spec(
    ModelSpec(
        name="gui_owl_8b_think_transformers",
        description="GUI-Owl-1.5-8B-Think grounding preset using HuggingFace Transformers.",
        adapter=ComponentConfig(target="composable_adapter"),
        backend=ComponentConfig(
            target="gui_owl_transformers",
            kwargs={
                "model": "mPLUG/GUI-Owl-1.5-8B-Think",
                "max_new_tokens": 4096,
            },
        ),
        prompt_builder=ComponentConfig(target="plain_task_prompt_builder"),
        image_preprocessor=ComponentConfig(target="pil_image_preprocessor"),
        parser=ComponentConfig(target="mai_ui_grounding_parser"),
        transformer=ComponentConfig(
            target="coordinate_space_transformer",
            kwargs={
                "source_width": 1000,
                "source_height": 1000,
                "clip_to_image": True,
            },
        ),
        system_prompt=GUI_OWL_SYSTEM_PROMPT,
        generation={
            "temperature": 0.0,
            "max_new_tokens": 4096,
        },
    )
)

register_model_spec(
    ModelSpec(
        name="gui_owl_8b_instruct_vllm",
        description="GUI-Owl-1.5-8B-Instruct grounding preset using vLLM (Qwen3-VL).",
        adapter=ComponentConfig(target="composable_adapter"),
        backend=ComponentConfig(
            target="chat_vllm",
            kwargs={
                "model": "mPLUG/GUI-Owl-1.5-8B-Instruct",
                "trust_remote_code": True,
                "max_model_len": 24576,
                "max_num_seqs": 64,
                "gpu_memory_utilization": 0.9,
                "apply_chat_template_with": "processor",
            },
        ),
        prompt_builder=ComponentConfig(
            target="chat_messages_prompt_builder",
            kwargs={"user_template": "{instruction}"},
        ),
        image_preprocessor=ComponentConfig(target="pil_image_preprocessor"),
        parser=ComponentConfig(target="mai_ui_grounding_parser"),
        transformer=ComponentConfig(
            target="coordinate_space_transformer",
            kwargs={
                "source_width": 1000,
                "source_height": 1000,
                "clip_to_image": True,
            },
        ),
        system_prompt=GUI_OWL_SYSTEM_PROMPT,
        generation={
            "temperature": 0.0,
            "max_tokens": 2048,
        },
    )
)

register_model_spec(
    ModelSpec(
        name="gui_owl_8b_think_vllm",
        description="GUI-Owl-1.5-8B-Think grounding preset using vLLM (Qwen3-VL).",
        adapter=ComponentConfig(target="composable_adapter"),
        backend=ComponentConfig(
            target="chat_vllm",
            kwargs={
                "model": "mPLUG/GUI-Owl-1.5-8B-Think",
                "trust_remote_code": True,
                "max_model_len": 24576,
                "max_num_seqs": 64,
                "gpu_memory_utilization": 0.9,
                "apply_chat_template_with": "processor",
            },
        ),
        prompt_builder=ComponentConfig(
            target="chat_messages_prompt_builder",
            kwargs={"user_template": "{instruction}"},
        ),
        image_preprocessor=ComponentConfig(target="pil_image_preprocessor"),
        parser=ComponentConfig(target="mai_ui_grounding_parser"),
        transformer=ComponentConfig(
            target="coordinate_space_transformer",
            kwargs={
                "source_width": 1000,
                "source_height": 1000,
                "clip_to_image": True,
            },
        ),
        system_prompt=GUI_OWL_SYSTEM_PROMPT,
        generation={
            "temperature": 0.0,
            "max_tokens": 4096,
        },
    )
)

# ---------------------------------------------------------------------------
# EvoCUA (meituan) – Qwen3-VL fine-tune, S2 tool-call format
# ---------------------------------------------------------------------------

_EVOCUA_S2_DESCRIPTION = (
    "Use a mouse and keyboard to interact with a computer, and take screenshots.\n"
    "* This is an interface to a desktop GUI. You must click on desktop icons to start applications.\n"
    "* Some applications may take time to start or process actions, so you may need to wait "
    "and take successive screenshots to see the results of your actions. E.g. if you click on "
    "Firefox and a window doesn't open, try wait and taking another screenshot.\n"
    "* The screen's resolution is 1000x1000.\n"
    "* Whenever you intend to move the cursor to click on an element like an icon, you should "
    "consult a screenshot to determine the coordinates of the element before moving the cursor.\n"
    "* If you tried clicking on a program or link but it failed to load even after waiting, "
    "try adjusting your cursor position so that the tip of the cursor visually falls on "
    "the element that you want to click.\n"
    "* Make sure to click any buttons, links, icons, etc with the cursor tip in the center "
    "of the element. Don't click boxes on their edges unless asked."
)

_EVOCUA_S2_ACTION_DESC = (
    '* `key`: Performs key down presses on the arguments passed in order, '
    'then performs key releases in reverse order.\n'
    '* `key_down`: Press and HOLD the specified key(s) down in order (no release).\n'
    '* `key_up`: Release the specified key(s) in reverse order.\n'
    '* `type`: Type a string of text on the keyboard.\n'
    '* `mouse_move`: Move the cursor to a specified (x, y) pixel coordinate on the screen.\n'
    '* `left_click`: Click the left mouse button at a specified (x, y) pixel coordinate on the screen.\n'
    '* `left_click_drag`: Click and drag the cursor to a specified (x, y) pixel coordinate on the screen.\n'
    '* `right_click`: Click the right mouse button at a specified (x, y) pixel coordinate on the screen.\n'
    '* `middle_click`: Click the middle mouse button at a specified (x, y) pixel coordinate on the screen.\n'
    '* `double_click`: Double-click the left mouse button at a specified (x, y) pixel coordinate on the screen.\n'
    '* `triple_click`: Triple-click the left mouse button at a specified (x, y) pixel coordinate on the screen.\n'
    '* `scroll`: Performs a scroll of the mouse scroll wheel.\n'
    '* `wait`: Wait specified seconds for the change to happen.\n'
    '* `terminate`: Terminate the current task and report its completion status.'
)

import json as _json

_EVOCUA_TOOLS_DEF = _json.dumps({
    "type": "function",
    "function": {
        "name_for_human": "computer_use",
        "name": "computer_use",
        "description": _EVOCUA_S2_DESCRIPTION,
        "parameters": {
            "properties": {
                "action": {
                    "description": _EVOCUA_S2_ACTION_DESC,
                    "enum": [
                        "key", "type", "mouse_move", "left_click",
                        "left_click_drag", "right_click", "middle_click",
                        "double_click", "triple_click", "scroll",
                        "wait", "terminate", "key_down", "key_up",
                    ],
                    "type": "string",
                },
                "keys": {"description": "Required only by `action=key`.", "type": "array"},
                "text": {"description": "Required only by `action=type`.", "type": "string"},
                "coordinate": {"description": "The x,y coordinates for mouse actions.", "type": "array"},
                "pixels": {"description": "The amount of scrolling.", "type": "number"},
                "time": {"description": "The seconds to wait.", "type": "number"},
                "status": {
                    "description": "The status of the task.",
                    "type": "string",
                    "enum": ["success", "failure"],
                },
            },
            "required": ["action"],
            "type": "object",
        },
        "args_format": "Format the arguments as a JSON object.",
    },
})

EVOCUA_S2_SYSTEM_PROMPT = (
    "# Tools\n\n"
    "You may call one or more functions to assist with the user query.\n\n"
    "You are provided with function signatures within <tools></tools> XML tags:\n"
    "<tools>\n"
    f"{_EVOCUA_TOOLS_DEF}\n"
    "</tools>\n\n"
    "For each function call, return a json object with function name and arguments "
    "within <tool_call></tool_call> XML tags:\n"
    "<tool_call>\n"
    '{"name": <function-name>, "arguments": <args-json-object>}\n'
    "</tool_call>\n\n"
    "# Response format\n\n"
    "Response format for every step:\n"
    "1) Action: a short imperative describing what to do in the UI.\n"
    "2) A single <tool_call>...</tool_call> block containing only the JSON: "
    '{"name": <function-name>, "arguments": <args-json-object>}.\n\n'
    "Rules:\n"
    "- Output exactly in the order: Action, <tool_call>.\n"
    "- Be brief: one sentence for Action.\n"
    "- Do not output anything else outside those parts.\n"
    "- If finishing, use action=terminate in the tool call."
)

# Grounding-only system prompt: forces left_click on the target element instead
# of letting the model pick `wait` / `terminate`. Used for benchmarks where the
# task is purely "where is element X?".
_EVOCUA_GROUND_TOOLS_DEF = _json.dumps({
    "type": "function",
    "function": {
        "name_for_human": "computer_use",
        "name": "computer_use",
        "description": (
            "Click on the UI element described in the user instruction. "
            "Always emit a single left_click action with the (x, y) coordinate "
            "of the target element on the provided screenshot."
        ),
        "parameters": {
            "properties": {
                "action": {
                    "description": "Must be `left_click`.",
                    "enum": ["left_click"],
                    "type": "string",
                },
                "coordinate": {
                    "description": "The [x, y] pixel coordinate to click.",
                    "type": "array",
                },
            },
            "required": ["action", "coordinate"],
            "type": "object",
        },
        "args_format": "Format the arguments as a JSON object.",
    },
})

EVOCUA_GROUNDING_SYSTEM_PROMPT = (
    "# Tools\n\n"
    "You are a GUI grounding model. Your job is to locate the UI element "
    "described by the user instruction on the given screenshot and click it.\n\n"
    "You must call exactly one function. The function signature is in <tools></tools>:\n"
    "<tools>\n"
    f"{_EVOCUA_GROUND_TOOLS_DEF}\n"
    "</tools>\n\n"
    "For the function call, return a json object with function name and arguments "
    "within <tool_call></tool_call> XML tags:\n"
    "<tool_call>\n"
    '{"name": "computer_use", "arguments": {"action": "left_click", "coordinate": [x, y]}}\n'
    "</tool_call>\n\n"
    "# Response format\n\n"
    "Response format for every step:\n"
    "1) Action: a short imperative describing what you are clicking.\n"
    "2) A single <tool_call>...</tool_call> block containing only the JSON.\n\n"
    "Rules:\n"
    "- Output exactly in the order: Action, <tool_call>.\n"
    "- Be brief: one sentence for Action.\n"
    "- The action MUST be `left_click`. Do NOT use `wait`, `terminate`, "
    "or any other action.\n"
    "- The coordinate MUST point at the visual center of the element you describe.\n"
    "- The screen coordinate space is 0..1000 on both axes."
)

register_model_spec(
    ModelSpec(
        name="evocua_8b_transformers",
        description="EvoCUA-8B-20260105 grounding preset using HuggingFace Transformers (Qwen3-VL).",
        adapter=ComponentConfig(target="composable_adapter"),
        backend=ComponentConfig(
            target="gui_owl_transformers",
            kwargs={
                "model": "meituan/EvoCUA-8B-20260105",
                "max_new_tokens": 4096,
            },
        ),
        prompt_builder=ComponentConfig(target="evocua_prompt_builder"),
        image_preprocessor=ComponentConfig(target="pil_image_preprocessor"),
        parser=ComponentConfig(target="mai_ui_grounding_parser"),
        transformer=ComponentConfig(
            target="coordinate_space_transformer",
            kwargs={
                "source_width": 1000,
                "source_height": 1000,
                "clip_to_image": True,
            },
        ),
        system_prompt=EVOCUA_S2_SYSTEM_PROMPT,
        generation={
            "temperature": 0.0,
            "max_new_tokens": 4096,
        },
    )
)

register_model_spec(
    ModelSpec(
        name="evocua_32b_transformers",
        description="EvoCUA-32B-20260105 grounding preset using HuggingFace Transformers (Qwen3-VL).",
        adapter=ComponentConfig(target="composable_adapter"),
        backend=ComponentConfig(
            target="gui_owl_transformers",
            kwargs={
                "model": "meituan/EvoCUA-32B-20260105",
                "max_new_tokens": 4096,
            },
        ),
        prompt_builder=ComponentConfig(target="evocua_prompt_builder"),
        image_preprocessor=ComponentConfig(target="pil_image_preprocessor"),
        parser=ComponentConfig(target="mai_ui_grounding_parser"),
        transformer=ComponentConfig(
            target="coordinate_space_transformer",
            kwargs={
                "source_width": 1000,
                "source_height": 1000,
                "clip_to_image": True,
            },
        ),
        system_prompt=EVOCUA_S2_SYSTEM_PROMPT,
        generation={
            "temperature": 0.0,
            "max_new_tokens": 4096,
        },
    )
)

_EVOCUA_VLLM_USER_TEMPLATE = (
    "Please generate the next move according to the UI screenshot, "
    "instruction and previous actions.\n\n"
    "Instruction: {instruction}\n\n"
    "Previous actions:\nNone"
)

register_model_spec(
    ModelSpec(
        name="evocua_8b_vllm",
        description="EvoCUA-8B-20260105 grounding preset using vLLM (Qwen3-VL).",
        adapter=ComponentConfig(target="composable_adapter"),
        backend=ComponentConfig(
            target="chat_vllm",
            kwargs={
                "model": "meituan/EvoCUA-8B-20260105",
                "trust_remote_code": True,
                "max_model_len": 24576,
                "max_num_seqs": 64,
                "gpu_memory_utilization": 0.9,
                "apply_chat_template_with": "processor",
            },
        ),
        prompt_builder=ComponentConfig(
            target="chat_messages_prompt_builder",
            kwargs={"user_template": _EVOCUA_VLLM_USER_TEMPLATE},
        ),
        image_preprocessor=ComponentConfig(target="pil_image_preprocessor"),
        parser=ComponentConfig(target="mai_ui_grounding_parser"),
        transformer=ComponentConfig(
            target="coordinate_space_transformer",
            kwargs={
                "source_width": 1000,
                "source_height": 1000,
                "clip_to_image": True,
            },
        ),
        system_prompt=EVOCUA_GROUNDING_SYSTEM_PROMPT,
        generation={
            "temperature": 0.0,
            "max_tokens": 4096,
        },
    )
)

register_model_spec(
    ModelSpec(
        name="evocua_32b_vllm",
        description="EvoCUA-32B-20260105 grounding preset using vLLM (Qwen3-VL).",
        adapter=ComponentConfig(target="composable_adapter"),
        backend=ComponentConfig(
            target="chat_vllm",
            kwargs={
                "model": "meituan/EvoCUA-32B-20260105",
                "trust_remote_code": True,
                "tensor_parallel_size": 4,
                "max_model_len": 16384,
                "max_num_seqs": 8,
                "gpu_memory_utilization": 0.85,
                "apply_chat_template_with": "processor",
            },
        ),
        prompt_builder=ComponentConfig(
            target="chat_messages_prompt_builder",
            kwargs={"user_template": _EVOCUA_VLLM_USER_TEMPLATE},
        ),
        image_preprocessor=ComponentConfig(target="pil_image_preprocessor"),
        parser=ComponentConfig(target="mai_ui_grounding_parser"),
        transformer=ComponentConfig(
            target="coordinate_space_transformer",
            kwargs={
                "source_width": 1000,
                "source_height": 1000,
                "clip_to_image": True,
            },
        ),
        system_prompt=EVOCUA_GROUNDING_SYSTEM_PROMPT,
        generation={
            "temperature": 0.0,
            "max_tokens": 4096,
        },
    )
)
# ---------------------------------------------------------------------------
# SeeClick – Qwen-VL grounding model, normalized 0-1 point output
# ---------------------------------------------------------------------------

SEECLICK_GROUNDING_PROMPT = (
    'In this UI screenshot, what is the position of the element '
    'corresponding to the command "{instruction}" (with point)?'
)

register_model_spec(
    ModelSpec(
        name="seeclick_transformers",
        description="SeeClick grounding preset using HuggingFace Transformers (Qwen-VL).",
        adapter=ComponentConfig(target="composable_adapter"),
        backend=ComponentConfig(
            target="seeclick_transformers",
            kwargs={"model": "cckevinn/SeeClick"},
        ),
        prompt_builder=ComponentConfig(
            target="default_prompt_builder",
            kwargs={"template": SEECLICK_GROUNDING_PROMPT},
        ),
        image_preprocessor=ComponentConfig(target="identity_image_preprocessor"),
        parser=ComponentConfig(target="seeclick_point_parser"),
        transformer=ComponentConfig(
            target="coordinate_space_transformer",
            kwargs={"normalized": True, "clip_to_image": True},
        ),
    )
)

# ---------------------------------------------------------------------------
# UGround-V1 – Qwen2-VL grounding model, 0-1000 relative point output
# ---------------------------------------------------------------------------

UGROUND_GROUNDING_PROMPT = (
    "Your task is to help the user identify the precise coordinates (x, y) "
    "of a specific area/element/object on the screen based on a description.\n"
    "Description: {instruction}\n"
    "Answer:"
)

register_model_spec(
    ModelSpec(
        name="uground_v1_7b_transformers",
        description="UGround-V1-7B grounding preset using HuggingFace Transformers (Qwen2-VL).",
        adapter=ComponentConfig(target="composable_adapter"),
        backend=ComponentConfig(
            target="uground_v1_transformers",
            kwargs={"model": "osunlp/UGround-V1-7B"},
        ),
        prompt_builder=ComponentConfig(
            target="default_prompt_builder",
            kwargs={"template": UGROUND_GROUNDING_PROMPT},
        ),
        image_preprocessor=ComponentConfig(target="pil_image_preprocessor"),
        parser=ComponentConfig(target="regex_coordinate_parser"),
        transformer=ComponentConfig(
            target="coordinate_space_transformer",
            kwargs={"source_width": 1000, "source_height": 1000, "clip_to_image": True},
        ),
    )
)

register_model_spec(
    ModelSpec(
        name="uground_v1_2b_transformers",
        description="UGround-V1-2B grounding preset using HuggingFace Transformers (Qwen2-VL).",
        adapter=ComponentConfig(target="composable_adapter"),
        backend=ComponentConfig(
            target="uground_v1_transformers",
            kwargs={"model": "osunlp/UGround-V1-2B"},
        ),
        prompt_builder=ComponentConfig(
            target="default_prompt_builder",
            kwargs={"template": UGROUND_GROUNDING_PROMPT},
        ),
        image_preprocessor=ComponentConfig(target="pil_image_preprocessor"),
        parser=ComponentConfig(target="regex_coordinate_parser"),
        transformer=ComponentConfig(
            target="coordinate_space_transformer",
            kwargs={"source_width": 1000, "source_height": 1000, "clip_to_image": True},
        ),
    )
)

# ---------------------------------------------------------------------------
# OS-Atlas – InternVL2 (4B) and Qwen2-VL (7B) grounding models
# ---------------------------------------------------------------------------

OS_ATLAS_4B_GROUNDING_PROMPT = (
    "In the screenshot of this web page, please give me the coordinates of "
    "the element I want to click on according to my instructions "
    "(with point).\n{instruction}"
)

OS_ATLAS_7B_GROUNDING_PROMPT = (
    "In this UI screenshot, what is the position of the element "
    'corresponding to the command "{instruction}" (with bbox)?'
)

register_model_spec(
    ModelSpec(
        name="os_atlas_4b_transformers",
        description="OS-Atlas-Base-4B grounding preset using HuggingFace Transformers (InternVL2).",
        adapter=ComponentConfig(target="composable_adapter"),
        backend=ComponentConfig(
            target="os_atlas_4b_transformers",
            kwargs={"model": "OS-Copilot/OS-Atlas-Base-4B"},
        ),
        prompt_builder=ComponentConfig(
            target="default_prompt_builder",
            kwargs={"template": OS_ATLAS_4B_GROUNDING_PROMPT},
        ),
        image_preprocessor=ComponentConfig(target="pil_image_preprocessor"),
        parser=ComponentConfig(target="regex_coordinate_parser"),
        transformer=ComponentConfig(
            target="coordinate_space_transformer",
            kwargs={"source_width": 1000, "source_height": 1000, "clip_to_image": True},
        ),
    )
)

# vLLM preset for OS-Atlas-Base-4B (InternVL2-4B backbone). NOTE: OS-Atlas-Base-4B
# uses non-standard tokenizer additions (e.g. <|begin_of_box|>/<|end_of_box|>) and
# the vanilla `internvl_chat` chat template may not be a perfect match for the
# fine-tune's expected formatting. The preset below is provided for completeness
# but typically requires a separate environment with the InternVL2-tuned vLLM
# build (or a custom chat template) to reproduce the published numbers.
register_model_spec(
    ModelSpec(
        name="os_atlas_4b_vllm",
        description=(
            "OS-Atlas-Base-4B grounding preset using vLLM (InternVL2). "
            "May require a separate environment / custom chat template."
        ),
        adapter=ComponentConfig(target="composable_adapter"),
        backend=ComponentConfig(
            target="chat_vllm",
            kwargs={
                "model": "OS-Copilot/OS-Atlas-Base-4B",
                "trust_remote_code": True,
                "max_model_len": 8192,
                "max_num_seqs": 32,
                "gpu_memory_utilization": 0.85,
                "apply_chat_template_with": "tokenizer",
            },
        ),
        prompt_builder=ComponentConfig(
            target="chat_messages_prompt_builder",
            kwargs={"user_template": OS_ATLAS_4B_GROUNDING_PROMPT},
        ),
        image_preprocessor=ComponentConfig(target="pil_image_preprocessor"),
        parser=ComponentConfig(target="regex_coordinate_parser"),
        transformer=ComponentConfig(
            target="coordinate_space_transformer",
            kwargs={"source_width": 1000, "source_height": 1000, "clip_to_image": True},
        ),
        generation={
            "temperature": 0.0,
            "max_tokens": 256,
        },
    )
)

register_model_spec(
    ModelSpec(
        name="os_atlas_7b_transformers",
        description="OS-Atlas-Base-7B grounding preset using HuggingFace Transformers (Qwen2-VL).",
        adapter=ComponentConfig(target="composable_adapter"),
        backend=ComponentConfig(
            target="os_atlas_7b_transformers",
            kwargs={"model": "OS-Copilot/OS-Atlas-Base-7B"},
        ),
        prompt_builder=ComponentConfig(
            target="default_prompt_builder",
            kwargs={"template": OS_ATLAS_7B_GROUNDING_PROMPT},
        ),
        image_preprocessor=ComponentConfig(target="pil_image_preprocessor"),
        parser=ComponentConfig(target="os_atlas_bbox_parser"),
        transformer=ComponentConfig(
            target="coordinate_space_transformer",
            kwargs={"source_width": 1000, "source_height": 1000, "clip_to_image": True},
        ),
    )
)
