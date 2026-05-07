from __future__ import annotations

from typing import Dict, List

from cuactspot.types import SampleEvaluationResult

REPORT_MODALITIES = ["Canvas", "GUI", "Image", "Table", "Text"]


def infer_modality(sample_id: str) -> str:
    return sample_id.split("-", 1)[0]


def build_report(results: List[SampleEvaluationResult]) -> Dict[str, object]:
    return {
        "details": [item.to_dict() for item in results],
        "overall": build_overall_metrics(results),
    }


def build_overall_metrics(
    results: List[SampleEvaluationResult],
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}

    for modality in REPORT_MODALITIES:
        modality_results = [
            item for item in results if infer_modality(item.sample_id) == modality
        ]
        if not modality_results:
            metrics[modality] = 0.0
            continue
        acc = sum(1 for item in modality_results if item.passed) / float(
            len(modality_results)
        )
        acc = round(acc, 6)
        metrics[modality] = acc

    metrics["overall"] = round(
        sum(metrics[modality] for modality in REPORT_MODALITIES)
        / float(len(REPORT_MODALITIES)),
        6,
    )
    return metrics