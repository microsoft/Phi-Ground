from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from cuactspot.registry import PROMPT_BUILDER_REGISTRY
from cuactspot.types import DatasetSample, PromptSpec

DEFAULT_PROMPT_TEMPLATE = """You are given an image-based mouse operation task.
Task: {task}
Modality: {modality}

Return only the coordinates needed for the action.
Examples:
- click: [(x, y)]
- drag: [(x1, y1), (x2, y2)]
"""

UI_TARS_COMPUTER_USE_TEMPLATE = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task.

## Output Format
```
Thought: ...
Action: ...
```

## Action Space
click(point='<point>x1 y1</point>')
left_double(point='<point>x1 y1</point>')
right_single(point='<point>x1 y1</point>')
drag(start_point='<point>x1 y1</point>', end_point='<point>x2 y2</point>')
hotkey(key='ctrl c')
type(content='xxx')
scroll(point='<point>x1 y1</point>', direction='down or up or right or left')
wait()
finished(content='xxx')

## Note
- Use English in `Thought` part.
- Write a small plan and finally summarize your next action in one sentence in `Thought` part.

## User Instruction
{instruction}
"""

PHI_GROUND_ACTSPOT_TEMPLATE = """<|user|>
The instruction:
{instruction}

Identify the coordinate point or points needed to complete the instruction in the image.
Return each required point in order using relative coordinates multiplied by 1000.
Use the format <point>x, y</point> for each point and output only the points.
<|image_1|>
<|end|>
<|assistant|>"""

PHI_GROUND_SSP_TEMPLATE = """<|user|>
The description of the element: 
{instruction}

Locate the above described element in the image. The output should be bounding box using relative coordinates multiplying 1000.
<|image_1|>
<|end|>
<|assistant|>"""

UI_TARS_GROUNDING_TEMPLATE = (
    "You are a GUI agent. You are given a task and your action history, with screenshots. "
    "You need to perform the next action to complete the task. \n\n"
    "## Output Format\n\n"
    "Action: ...\n\n\n"
    "## Action Space\n"
    "click(point='<point>x1 y1</point>')\n\n"
    "## User Instruction\n"
    "{instruction}"
)


class BasePromptBuilder(ABC):
    @abstractmethod
    def build(
        self,
        sample: DatasetSample,
        system_prompt: Optional[str] = None,
    ) -> PromptSpec:
        raise NotImplementedError


@PROMPT_BUILDER_REGISTRY.register("default_prompt_builder")
class DefaultPromptBuilder(BasePromptBuilder):
    def __init__(self, template: Optional[str] = None) -> None:
        self.template = template or DEFAULT_PROMPT_TEMPLATE

    def build(
        self,
        sample: DatasetSample,
        system_prompt: Optional[str] = None,
    ) -> PromptSpec:
        user_prompt = self.template.format(
            instruction=sample.task_text,
            task=sample.task_text,
            modality=sample.modality or "unknown",
            sample_id=sample.sample_id,
        )
        return PromptSpec(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
        )


@PROMPT_BUILDER_REGISTRY.register("plain_task_prompt_builder")
class PlainTaskPromptBuilder(BasePromptBuilder):
    def build(
        self,
        sample: DatasetSample,
        system_prompt: Optional[str] = None,
    ) -> PromptSpec:
        return PromptSpec(
            user_prompt=sample.task_text,
            system_prompt=system_prompt,
        )


@PROMPT_BUILDER_REGISTRY.register("chat_messages_prompt_builder")
class ChatMessagesPromptBuilder(BasePromptBuilder):
    def __init__(
        self,
        user_template: str = "{instruction}",
        include_image: bool = True,
        image_content_type: str = "image",
        image_field_name: str = "image",
        image_first: bool = True,
    ) -> None:
        self.user_template = user_template
        self.include_image = include_image
        self.image_content_type = image_content_type
        self.image_field_name = image_field_name
        self.image_first = image_first

    def build(
        self,
        sample: DatasetSample,
        system_prompt: Optional[str] = None,
    ) -> PromptSpec:
        user_prompt = self.user_template.format(
            instruction=sample.task_text,
            task=sample.task_text,
            modality=sample.modality or "unknown",
            sample_id=sample.sample_id,
        )
        messages = []
        if system_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_prompt}],
                }
            )

        image_block = {
            "type": self.image_content_type,
            self.image_field_name: str(sample.image_path),
        }
        text_block = {"type": "text", "text": user_prompt}

        user_content: list = []
        if self.include_image and self.image_first:
            user_content.append(image_block)
            user_content.append(text_block)
        elif self.include_image and not self.image_first:
            user_content.append(text_block)
            user_content.append(image_block)
        else:
            user_content.append(text_block)
        messages.append({"role": "user", "content": user_content})
        return PromptSpec(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            messages=messages,
        )


@PROMPT_BUILDER_REGISTRY.register("phi_ground_actspot_prompt_builder")
class PhiGroundActSpotPromptBuilder(BasePromptBuilder):
    def __init__(self, template: Optional[str] = None) -> None:
        self.template = template or PHI_GROUND_ACTSPOT_TEMPLATE

    def build(
        self,
        sample: DatasetSample,
        system_prompt: Optional[str] = None,
    ) -> PromptSpec:
        return PromptSpec(
            user_prompt=self.template.format(instruction=sample.task_text),
            system_prompt=system_prompt,
        )


INFIGUI_GROUNDING_SYSTEM_PROMPT = (
    "You FIRST think about the reasoning process as an internal monologue "
    "and then provide the final answer.\n"
    "The reasoning process MUST BE enclosed within <think> </think> tags."
)

INFIGUI_GROUNDING_USER_TEMPLATE = (
    "The screen's resolution is {width}x{height}.\n"
    'Point to the UI element most relevant to "{instruction}", '
    "output its coordinates using JSON format:\n"
    "```json\n"
    "[\n"
    '    {{"point_2d": [x, y], "label": "object name/description"}}\n'
    "]```"
)


@PROMPT_BUILDER_REGISTRY.register("infigui_grounding_prompt_builder")
class InfiGUIGroundingPromptBuilder(BasePromptBuilder):
    def __init__(
        self,
        max_pixels: int = 4390400,
        min_pixels: int = 3136,
        factor: int = 28,
    ) -> None:
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.factor = factor

    def build(
        self,
        sample: DatasetSample,
        system_prompt: Optional[str] = None,
    ) -> PromptSpec:
        from cuactspot.utils.qwen import smart_resize

        if sample.image_size is not None:
            orig_w, orig_h = sample.image_size
            new_h, new_w = smart_resize(
                orig_h,
                orig_w,
                factor=self.factor,
                min_pixels=self.min_pixels,
                max_pixels=self.max_pixels,
            )
        else:
            new_w, new_h = 1024, 768

        user_text = INFIGUI_GROUNDING_USER_TEMPLATE.format(
            width=new_w,
            height=new_h,
            instruction=sample.task_text,
        )

        sys_text = system_prompt or INFIGUI_GROUNDING_SYSTEM_PROMPT

        messages = [
            {"role": "system", "content": sys_text},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(sample.image_path)},
                    {"type": "text", "text": user_text},
                ],
            },
        ]

        return PromptSpec(
            user_prompt=user_text,
            system_prompt=sys_text,
            messages=messages,
        )


INFIGUI_G1_GROUNDING_USER_TEMPLATE = (
    'The screen\'s resolution is {width}x{height}.\n'
    'Locate the UI element(s) for "{instruction}", '
    'output the coordinates using JSON format: '
    '[{{"point_2d": [x, y]}}, ...]'
)


@PROMPT_BUILDER_REGISTRY.register("infigui_g1_grounding_prompt_builder")
class InfiGUIGroundingG1PromptBuilder(BasePromptBuilder):
    def __init__(
        self,
        max_pixels: int = 4390400,
        min_pixels: int = 3136,
        factor: int = 28,
    ) -> None:
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.factor = factor

    def build(
        self,
        sample: DatasetSample,
        system_prompt: Optional[str] = None,
    ) -> PromptSpec:
        from cuactspot.utils.qwen import smart_resize

        if sample.image_size is not None:
            orig_w, orig_h = sample.image_size
            new_h, new_w = smart_resize(
                orig_h,
                orig_w,
                factor=self.factor,
                min_pixels=self.min_pixels,
                max_pixels=self.max_pixels,
            )
        else:
            new_w, new_h = 1024, 768

        user_text = INFIGUI_G1_GROUNDING_USER_TEMPLATE.format(
            width=new_w,
            height=new_h,
            instruction=sample.task_text,
        )

        sys_text = system_prompt

        messages = []
        if sys_text:
            messages.append({"role": "system", "content": sys_text})
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(sample.image_path)},
                    {"type": "text", "text": user_text},
                ],
            },
        )

        return PromptSpec(
            user_prompt=user_text,
            system_prompt=sys_text,
            messages=messages,
        )


EVOCUA_USER_TEMPLATE = (
    "Please generate the next move according to the UI screenshot, "
    "instruction and previous actions.\n\n"
    "Instruction: {instruction}\n\n"
    "Previous actions:\nNone"
)


@PROMPT_BUILDER_REGISTRY.register("evocua_prompt_builder")
class EvoCUAPromptBuilder(BasePromptBuilder):
    def build(
        self,
        sample: DatasetSample,
        system_prompt: Optional[str] = None,
    ) -> PromptSpec:
        user_text = EVOCUA_USER_TEMPLATE.format(instruction=sample.task_text)
        return PromptSpec(
            user_prompt=user_text,
            system_prompt=system_prompt,
        )