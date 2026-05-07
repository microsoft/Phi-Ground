from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from cuactspot.registry import TRANSFORMER_REGISTRY
from cuactspot.types import BackendResponse, Coordinate, DatasetSample, GenerationRequest
from cuactspot.utils.qwen import smart_resize


class BaseCoordinateTransformer(ABC):
    @abstractmethod
    def transform(
        self,
        coordinates: list[Coordinate],
        sample: DatasetSample,
        request: Optional[GenerationRequest] = None,
        response: Optional[BackendResponse] = None,
    ) -> list[Coordinate]:
        raise NotImplementedError


@TRANSFORMER_REGISTRY.register("identity_coordinate_transformer")
class IdentityCoordinateTransformer(BaseCoordinateTransformer):
    def transform(
        self,
        coordinates: list[Coordinate],
        sample: DatasetSample,
        request: Optional[GenerationRequest] = None,
        response: Optional[BackendResponse] = None,
    ) -> list[Coordinate]:
        return list(coordinates)


@TRANSFORMER_REGISTRY.register("coordinate_space_transformer")
class CoordinateSpaceTransformer(BaseCoordinateTransformer):
    def __init__(
        self,
        source_width: Optional[float] = None,
        source_height: Optional[float] = None,
        normalized: bool = False,
        clip_to_image: bool = False,
        round_digits: Optional[int] = 2,
    ) -> None:
        self.source_width = source_width
        self.source_height = source_height
        self.normalized = normalized
        self.clip_to_image = clip_to_image
        self.round_digits = round_digits

    def transform(
        self,
        coordinates: list[Coordinate],
        sample: DatasetSample,
        request: Optional[GenerationRequest] = None,
        response: Optional[BackendResponse] = None,
    ) -> list[Coordinate]:
        if sample.image_size is None:
            return list(coordinates)

        width, height = sample.image_size
        transformed: list[Coordinate] = []
        for x, y in coordinates:
            new_x, new_y = self._transform_point(x, y, width, height)
            if self.clip_to_image:
                new_x = min(max(new_x, 0.0), float(width))
                new_y = min(max(new_y, 0.0), float(height))
            if self.round_digits is not None:
                new_x = round(new_x, self.round_digits)
                new_y = round(new_y, self.round_digits)
            transformed.append((new_x, new_y))
        return transformed

    def _transform_point(
        self,
        x: float,
        y: float,
        target_width: int,
        target_height: int,
    ) -> Coordinate:
        if self.normalized:
            return (x * target_width, y * target_height)
        if self.source_width and self.source_height:
            return (
                x * target_width / self.source_width,
                y * target_height / self.source_height,
            )
        return (x, y)


@TRANSFORMER_REGISTRY.register("phi_ground_relative_transformer")
class PhiGroundRelativeCoordinateTransformer(BaseCoordinateTransformer):
    def __init__(
        self,
        scale: float = 1000.0,
        clip_to_image: bool = True,
        round_digits: Optional[int] = 2,
    ) -> None:
        self.scale = scale
        self.clip_to_image = clip_to_image
        self.round_digits = round_digits

    def transform(
        self,
        coordinates: list[Coordinate],
        sample: DatasetSample,
        request: Optional[GenerationRequest] = None,
        response: Optional[BackendResponse] = None,
    ) -> list[Coordinate]:
        if request is None or request.image is None:
            return list(coordinates)

        metadata = request.image.metadata
        reshape_ratio = metadata.get("reshape_ratio")
        target_width = metadata.get("target_width") or request.image.width
        target_height = metadata.get("target_height") or request.image.height
        if not reshape_ratio or not target_width or not target_height:
            return list(coordinates)

        image_width = None if sample.image_size is None else sample.image_size[0]
        image_height = None if sample.image_size is None else sample.image_size[1]

        transformed: list[Coordinate] = []
        for x, y in coordinates:
            new_x = (float(x) / self.scale) * float(target_width) / float(reshape_ratio)
            new_y = (float(y) / self.scale) * float(target_height) / float(reshape_ratio)
            if self.clip_to_image and image_width is not None and image_height is not None:
                new_x = min(max(new_x, 0.0), float(image_width))
                new_y = min(max(new_y, 0.0), float(image_height))
            if self.round_digits is not None:
                new_x = round(new_x, self.round_digits)
                new_y = round(new_y, self.round_digits)
            transformed.append((new_x, new_y))
        return transformed


@TRANSFORMER_REGISTRY.register("smart_resize_coordinate_transformer")
class SmartResizeCoordinateTransformer(BaseCoordinateTransformer):
    def __init__(
        self,
        factor: int = 28,
        min_pixels: int = 3136,
        max_pixels: int = 12845056,
        clip_to_image: bool = True,
        round_digits: Optional[int] = 2,
    ) -> None:
        self.factor = factor
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.clip_to_image = clip_to_image
        self.round_digits = round_digits

    def transform(
        self,
        coordinates: list[Coordinate],
        sample: DatasetSample,
        request: Optional[GenerationRequest] = None,
        response: Optional[BackendResponse] = None,
    ) -> list[Coordinate]:
        if sample.image_size is None:
            return list(coordinates)

        image_width, image_height = sample.image_size
        resized_height, resized_width = smart_resize(
            image_height,
            image_width,
            factor=self.factor,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )

        transformed: list[Coordinate] = []
        for x, y in coordinates:
            new_x = float(x) * float(image_width) / float(resized_width)
            new_y = float(y) * float(image_height) / float(resized_height)
            if self.clip_to_image:
                new_x = min(max(new_x, 0.0), float(image_width))
                new_y = min(max(new_y, 0.0), float(image_height))
            if self.round_digits is not None:
                new_x = round(new_x, self.round_digits)
                new_y = round(new_y, self.round_digits)
            transformed.append((new_x, new_y))
        return transformed


@TRANSFORMER_REGISTRY.register("aspect_ratio_pad_coordinate_transformer")
class AspectRatioPadCoordinateTransformer(BaseCoordinateTransformer):
    """Transform coordinates from a padded canvas back to original image space.

    The model outputs coordinates in ``source_width x source_height`` space,
    but the image was aspect-ratio-resized and top-left pasted on that canvas.
    This transformer uses the ``resized_width``/``resized_height`` from the
    image metadata to map coordinates to original image pixels.

    When ``model_display_width``/``model_display_height`` are set, the model
    outputs in a different coordinate space than the canvas.  Coordinates are
    first scaled from model-display space to canvas space before the
    pad-aware mapping is applied.
    """

    def __init__(
        self,
        source_width: float = 1440,
        source_height: float = 900,
        model_display_width: Optional[float] = None,
        model_display_height: Optional[float] = None,
        clip_to_image: bool = True,
        round_digits: Optional[int] = 2,
    ) -> None:
        self.source_width = source_width
        self.source_height = source_height
        self.model_display_width = model_display_width
        self.model_display_height = model_display_height
        self.clip_to_image = clip_to_image
        self.round_digits = round_digits

    def transform(
        self,
        coordinates: list[Coordinate],
        sample: DatasetSample,
        request: Optional[GenerationRequest] = None,
        response: Optional[BackendResponse] = None,
    ) -> list[Coordinate]:
        if sample.image_size is None:
            return list(coordinates)

        image_width, image_height = sample.image_size

        resized_w = self.source_width
        resized_h = self.source_height
        if request is not None and request.image is not None:
            meta = request.image.metadata
            resized_w = meta.get("resized_width", self.source_width)
            resized_h = meta.get("resized_height", self.source_height)

        # Scale resized dims from canvas space to model-display space.
        if self.model_display_width is not None and self.source_width:
            resized_w = resized_w * self.model_display_width / self.source_width
        if self.model_display_height is not None and self.source_height:
            resized_h = resized_h * self.model_display_height / self.source_height

        transformed: list[Coordinate] = []
        for x, y in coordinates:
            new_x = float(x) * float(image_width) / float(resized_w)
            new_y = float(y) * float(image_height) / float(resized_h)
            if self.clip_to_image:
                new_x = min(max(new_x, 0.0), float(image_width))
                new_y = min(max(new_y, 0.0), float(image_height))
            if self.round_digits is not None:
                new_x = round(new_x, self.round_digits)
                new_y = round(new_y, self.round_digits)
            transformed.append((new_x, new_y))
        return transformed