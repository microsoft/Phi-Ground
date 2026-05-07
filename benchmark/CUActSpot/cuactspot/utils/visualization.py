from __future__ import annotations

from pathlib import Path
from typing import Iterable

from cuactspot.types import Coordinate, DatasetSample


def draw_coordinates(
    sample: DatasetSample,
    coordinates: Iterable[Coordinate],
    output_path: Path,
    radius: int = 8,
    outline_width: int = 3,
) -> None:
    from PIL import Image, ImageDraw

    with Image.open(sample.image_path).convert("RGB") as image:
        draw = ImageDraw.Draw(image)
        for x, y in coordinates:
            bounding_box = [
                (x - radius, y - radius),
                (x + radius, y + radius),
            ]
            draw.ellipse(bounding_box, outline="red", width=outline_width)
        image.save(output_path)