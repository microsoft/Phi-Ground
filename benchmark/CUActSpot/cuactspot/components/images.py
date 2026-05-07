from __future__ import annotations

import base64
import mimetypes
from abc import ABC, abstractmethod
from typing import Tuple

from cuactspot.registry import IMAGE_PREPROCESSOR_REGISTRY
from cuactspot.types import DatasetSample, PreparedImage


class BaseImagePreprocessor(ABC):
    @abstractmethod
    def process(self, sample: DatasetSample) -> PreparedImage:
        raise NotImplementedError


@IMAGE_PREPROCESSOR_REGISTRY.register("identity_image_preprocessor")
class IdentityImagePreprocessor(BaseImagePreprocessor):
    def process(self, sample: DatasetSample) -> PreparedImage:
        width = None if sample.image_size is None else sample.image_size[0]
        height = None if sample.image_size is None else sample.image_size[1]
        return PreparedImage(
            source_path=sample.image_path,
            payload=sample.image_path,
            width=width,
            height=height,
        )


@IMAGE_PREPROCESSOR_REGISTRY.register("base64_image_preprocessor")
class Base64ImagePreprocessor(BaseImagePreprocessor):
    def process(self, sample: DatasetSample) -> PreparedImage:
        mime_type, _ = mimetypes.guess_type(sample.image_path.name)
        with sample.image_path.open("rb") as file_obj:
            payload = base64.b64encode(file_obj.read()).decode("utf-8")
        width = None if sample.image_size is None else sample.image_size[0]
        height = None if sample.image_size is None else sample.image_size[1]
        return PreparedImage(
            source_path=sample.image_path,
            payload=payload,
            width=width,
            height=height,
            metadata={"mime_type": mime_type or "application/octet-stream"},
        )


@IMAGE_PREPROCESSOR_REGISTRY.register("pil_image_preprocessor")
class PILImagePreprocessor(BaseImagePreprocessor):
    def __init__(self, mode: str = "RGB") -> None:
        self.mode = mode

    def process(self, sample: DatasetSample) -> PreparedImage:
        from PIL import Image

        with Image.open(sample.image_path) as image:
            processed = image.convert(self.mode)

        width = None if sample.image_size is None else sample.image_size[0]
        height = None if sample.image_size is None else sample.image_size[1]
        return PreparedImage(
            source_path=sample.image_path,
            payload=processed,
            width=width,
            height=height,
            metadata={"mode": self.mode},
        )


@IMAGE_PREPROCESSOR_REGISTRY.register("phi_ground_image_preprocessor")
class PhiGroundImagePreprocessor(BaseImagePreprocessor):
    def __init__(
        self,
        target_width: int = 336 * 3,
        target_height: int = 336 * 2,
        background_color: Tuple[int, int, int] = (255, 255, 255),
    ) -> None:
        self.target_width = target_width
        self.target_height = target_height
        self.background_color = background_color

    def process(self, sample: DatasetSample) -> PreparedImage:
        from PIL import Image

        with Image.open(sample.image_path) as image:
            image = image.convert("RGB")
            processed_image, reshape_ratio, resized_size = self._process_image(image)
            original_width, original_height = image.size

        return PreparedImage(
            source_path=sample.image_path,
            payload=processed_image,
            width=self.target_width,
            height=self.target_height,
            metadata={
                "reshape_ratio": reshape_ratio,
                "target_width": self.target_width,
                "target_height": self.target_height,
                "resized_width": resized_size[0],
                "resized_height": resized_size[1],
                "original_width": original_width,
                "original_height": original_height,
            },
        )

    def _process_image(self, image):
        from PIL import Image

        img_ratio = image.width / float(image.height)
        target_ratio = self.target_width / float(self.target_height)

        if img_ratio > target_ratio:
            new_width = self.target_width
            new_height = int(new_width / img_ratio)
        else:
            new_height = self.target_height
            new_width = int(new_height * img_ratio)

        reshape_ratio = new_width / float(image.width)
        resized_image = image.resize((new_width, new_height), Image.LANCZOS)
        canvas = Image.new(
            "RGB",
            (self.target_width, self.target_height),
            self.background_color,
        )
        canvas.paste(resized_image, (0, 0))
        return canvas, reshape_ratio, (new_width, new_height)


@IMAGE_PREPROCESSOR_REGISTRY.register("resize_base64_image_preprocessor")
class ResizeBase64ImagePreprocessor(BaseImagePreprocessor):
    """Resize image to a fixed resolution and encode as base64 JPEG."""

    def __init__(
        self,
        target_width: int = 1440,
        target_height: int = 900,
        quality: int = 95,
    ) -> None:
        self.target_width = target_width
        self.target_height = target_height
        self.quality = quality

    def process(self, sample: DatasetSample) -> PreparedImage:
        import io
        from PIL import Image

        with Image.open(sample.image_path) as image:
            image = image.convert("RGB")
            original_width, original_height = image.size
            resized = image.resize(
                (self.target_width, self.target_height), Image.LANCZOS
            )

        buf = io.BytesIO()
        resized.save(buf, format="JPEG", quality=self.quality)
        payload = base64.b64encode(buf.getvalue()).decode("utf-8")

        return PreparedImage(
            source_path=sample.image_path,
            payload=payload,
            width=self.target_width,
            height=self.target_height,
            metadata={
                "mime_type": "image/jpeg",
                "original_width": original_width,
                "original_height": original_height,
                "target_width": self.target_width,
                "target_height": self.target_height,
            },
        )


@IMAGE_PREPROCESSOR_REGISTRY.register("aspect_ratio_pad_base64_image_preprocessor")
class AspectRatioPadBase64ImagePreprocessor(BaseImagePreprocessor):
    """Resize image keeping aspect ratio, pad to target size, encode as base64 JPEG."""

    def __init__(
        self,
        target_width: int = 1440,
        target_height: int = 900,
        quality: int = 95,
        background_color: tuple = (255, 255, 255),
    ) -> None:
        self.target_width = target_width
        self.target_height = target_height
        self.quality = quality
        self.background_color = tuple(background_color)

    def process(self, sample: DatasetSample) -> PreparedImage:
        import io
        from PIL import Image

        with Image.open(sample.image_path) as image:
            image = image.convert("RGB")
            original_width, original_height = image.size

            img_ratio = original_width / float(original_height)
            target_ratio = self.target_width / float(self.target_height)

            if img_ratio > target_ratio:
                new_width = self.target_width
                new_height = int(new_width / img_ratio)
            else:
                new_height = self.target_height
                new_width = int(new_height * img_ratio)

            resized = image.resize((new_width, new_height), Image.LANCZOS)
            canvas = Image.new("RGB", (self.target_width, self.target_height), self.background_color)
            canvas.paste(resized, (0, 0))

        buf = io.BytesIO()
        canvas.save(buf, format="JPEG", quality=self.quality)
        payload = base64.b64encode(buf.getvalue()).decode("utf-8")

        return PreparedImage(
            source_path=sample.image_path,
            payload=payload,
            width=self.target_width,
            height=self.target_height,
            metadata={
                "mime_type": "image/jpeg",
                "original_width": original_width,
                "original_height": original_height,
                "target_width": self.target_width,
                "target_height": self.target_height,
                "resized_width": new_width,
                "resized_height": new_height,
                "aspect_ratio_preserved": True,
            },
        )


@IMAGE_PREPROCESSOR_REGISTRY.register("original_base64_image_preprocessor")
class OriginalBase64ImagePreprocessor(BaseImagePreprocessor):
    """Encode original image as base64 JPEG without resizing."""

    def __init__(self, quality: int = 95) -> None:
        self.quality = quality

    def process(self, sample: DatasetSample) -> PreparedImage:
        import io
        from PIL import Image

        with Image.open(sample.image_path) as image:
            image = image.convert("RGB")
            original_width, original_height = image.size

        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=self.quality)
        payload = base64.b64encode(buf.getvalue()).decode("utf-8")

        return PreparedImage(
            source_path=sample.image_path,
            payload=payload,
            width=original_width,
            height=original_height,
            metadata={
                "mime_type": "image/jpeg",
                "original_width": original_width,
                "original_height": original_height,
            },
        )


@IMAGE_PREPROCESSOR_REGISTRY.register("adaptive_resize_base64_image_preprocessor")
class AdaptiveResizeBase64ImagePreprocessor(BaseImagePreprocessor):
    """Force-resize when aspect ratio is close to target; resize+pad otherwise."""

    def __init__(
        self,
        target_width: int = 1440,
        target_height: int = 900,
        ratio_threshold: float = 0.2,
        quality: int = 95,
        background_color: tuple = (255, 255, 255),
    ) -> None:
        self.target_width = target_width
        self.target_height = target_height
        self.ratio_threshold = ratio_threshold
        self.quality = quality
        self.background_color = background_color

    def process(self, sample: DatasetSample) -> PreparedImage:
        import io
        from PIL import Image

        with Image.open(sample.image_path) as image:
            image = image.convert("RGB")
            original_width, original_height = image.size

        img_ratio = original_width / float(original_height)
        target_ratio = self.target_width / float(self.target_height)
        ratio_diff = abs(img_ratio - target_ratio) / target_ratio

        if ratio_diff <= self.ratio_threshold:
            resized = image.resize(
                (self.target_width, self.target_height), Image.LANCZOS
            )
            result_image = resized
            resized_width = self.target_width
            resized_height = self.target_height
            used_pad = False
        else:
            if img_ratio > target_ratio:
                new_width = self.target_width
                new_height = int(new_width / img_ratio)
            else:
                new_height = self.target_height
                new_width = int(new_height * img_ratio)
            resized = image.resize((new_width, new_height), Image.LANCZOS)
            canvas = Image.new(
                "RGB",
                (self.target_width, self.target_height),
                self.background_color,
            )
            canvas.paste(resized, (0, 0))
            result_image = canvas
            resized_width = new_width
            resized_height = new_height
            used_pad = True

        buf = io.BytesIO()
        result_image.save(buf, format="JPEG", quality=self.quality)
        payload = base64.b64encode(buf.getvalue()).decode("utf-8")

        return PreparedImage(
            source_path=sample.image_path,
            payload=payload,
            width=self.target_width,
            height=self.target_height,
            metadata={
                "mime_type": "image/jpeg",
                "original_width": original_width,
                "original_height": original_height,
                "target_width": self.target_width,
                "target_height": self.target_height,
                "resized_width": resized_width,
                "resized_height": resized_height,
                "used_pad": used_pad,
            },
        )