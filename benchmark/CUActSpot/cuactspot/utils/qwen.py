from __future__ import annotations

import math


def smart_resize(
    height: int,
    width: int,
    factor: int = 28,
    min_pixels: int = 3136,
    max_pixels: int = 12845056,
    max_ratio: int = 200,
) -> tuple[int, int]:
    external = _load_external_smart_resize()
    if external is not None:
        return external(
            height,
            width,
            factor=factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )

    if height < factor or width < factor:
        raise ValueError("Image dimensions must be at least one factor step")

    ratio = max(height, width) / float(min(height, width))
    if ratio > max_ratio:
        raise ValueError(
            f"Image aspect ratio {ratio:.2f} exceeds the supported max ratio {max_ratio}"
        )

    resized_height = max(factor, _round_by_factor(height, factor))
    resized_width = max(factor, _round_by_factor(width, factor))

    current_pixels = resized_height * resized_width
    if current_pixels > max_pixels:
        beta = math.sqrt((height * width) / float(max_pixels))
        resized_height = max(factor, _floor_by_factor(height / beta, factor))
        resized_width = max(factor, _floor_by_factor(width / beta, factor))
    elif current_pixels < min_pixels:
        beta = math.sqrt(min_pixels / float(height * width))
        resized_height = max(factor, _ceil_by_factor(height * beta, factor))
        resized_width = max(factor, _ceil_by_factor(width * beta, factor))

    return int(resized_height), int(resized_width)


def _load_external_smart_resize():
    try:
        from qwen_vl_utils import smart_resize as external_smart_resize

        return external_smart_resize
    except ImportError:
        pass

    try:
        from transformers.models.qwen2_vl.image_processing_qwen2_vl_fast import (
            smart_resize as external_smart_resize,
        )

        return external_smart_resize
    except ImportError:
        pass

    try:
        from transformers.models.qwen2_vl.image_processing_qwen2_vl import (
            smart_resize as external_smart_resize,
        )

        return external_smart_resize
    except ImportError:
        return None


def _round_by_factor(number: float, factor: int) -> int:
    return int(round(number / factor) * factor)


def _ceil_by_factor(number: float, factor: int) -> int:
    return int(math.ceil(number / factor) * factor)


def _floor_by_factor(number: float, factor: int) -> int:
    return int(math.floor(number / factor) * factor)