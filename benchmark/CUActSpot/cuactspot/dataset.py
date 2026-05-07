from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple, Union

from cuactspot.config import DatasetConfig
from cuactspot.types import DatasetSample, PolygonAnnotation


def load_samples(config: DatasetConfig) -> list[DatasetSample]:
    data_dir = Path(config.data_dir)
    json_files = sorted(data_dir.glob("*.json"))
    selected_ids = set(config.sample_ids)
    include_modalities = {item.lower() for item in config.include_modalities}
    exclude_modalities = {item.lower() for item in config.exclude_modalities}

    samples: list[DatasetSample] = []
    for json_path in json_files:
        sample = load_sample(json_path, config.image_extensions)
        if selected_ids and sample.sample_id not in selected_ids:
            continue
        modality = (sample.modality or "").lower()
        if include_modalities and modality not in include_modalities:
            continue
        if exclude_modalities and modality in exclude_modalities:
            continue
        samples.append(sample)

    limit = config.limit
    if limit is not None:
        return samples[:limit]
    return samples


def load_sample(
    json_path: Union[str, Path],
    image_extensions: Optional[List[str]] = None,
) -> DatasetSample:
    path = Path(json_path)
    with path.open("r", encoding="utf-8") as file_obj:
        payload = json.load(file_obj)

    sample_id = payload.get("image_id") or path.stem
    modality = payload.get("modality") or sample_id.split("-", maxsplit=1)[0]
    polygons = [
        PolygonAnnotation(
            polygon_id=item.get("id", ""),
            polygon_type=item.get("type", ""),
            points=[(float(x), float(y)) for x, y in item.get("points", [])],
            rank=str(item.get("rank", "")),
            visible=bool(item.get("visible", True)),
        )
        for item in payload.get("polygons", [])
    ]
    image_path = _resolve_image_path(path.parent, sample_id, path.stem, image_extensions)
    image_size = _probe_image_size(image_path)

    return DatasetSample(
        sample_id=sample_id,
        json_path=path,
        image_path=image_path,
        task=payload.get("task"),
        task_en=payload.get("task_en"),
        modality=modality,
        polygons=polygons,
        raw_annotation=payload,
        image_size=image_size,
        metadata={
            "masks": payload.get("masks", []),
        },
    )


def _resolve_image_path(
    data_dir: Path,
    image_id: str,
    sample_stem: str,
    image_extensions: Optional[List[str]],
) -> Path:
    extensions = image_extensions or [".png", ".jpg", ".jpeg", ".webp"]
    for stem in (image_id, sample_stem):
        for suffix in extensions:
            candidate = data_dir / f"{stem}{suffix}"
            if candidate.exists():
                return candidate
    raise FileNotFoundError(f"No image found for sample {sample_stem} in {data_dir}")


def _probe_image_size(image_path: Path) -> Optional[Tuple[int, int]]:
    try:
        from PIL import Image
    except ImportError:
        return None

    with Image.open(image_path) as image:
        return image.size