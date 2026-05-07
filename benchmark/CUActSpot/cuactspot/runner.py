from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter
from typing import Optional

from cuactspot.config import RunConfig
from cuactspot.dataset import load_samples
from cuactspot.evaluator import CoordinateEvaluator
from cuactspot.factory import build_model_adapter
from cuactspot.reporting import build_report
from cuactspot.types import DatasetSample, ModelPrediction, SampleEvaluationResult
from cuactspot.utils.io import ensure_directory, resolve_output_path, write_json
from cuactspot.utils.serialize import make_jsonable
from cuactspot.utils.visualization import draw_coordinates

_BENCHMARK_ACTSPOT = "actspot"

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm is optional at import time
    tqdm = None


class EvaluationRunner:
    def __init__(self, config: RunConfig) -> None:
        self.config = config
        self._benchmark = config.dataset.benchmark or _BENCHMARK_ACTSPOT
        if self._benchmark != _BENCHMARK_ACTSPOT:
            raise ValueError(
                f"Unsupported benchmark '{self._benchmark}'. "
                "This release only supports the 'actspot' benchmark (CUActSpot)."
            )
        self.evaluator = CoordinateEvaluator()

    def run(self) -> dict[str, object]:
        samples = load_samples(self.config.dataset)
        if self.config.runtime.max_samples is not None:
            samples = samples[: self.config.runtime.max_samples]

        output_dir = Path(self.config.output.output_dir)
        if output_dir.exists() and any(output_dir.iterdir()) and not self.config.output.overwrite:
            raise FileExistsError(
                f"Output directory already contains files: {output_dir}"
            )
        ensure_directory(output_dir)
        visualize_dir = self._resolve_visualize_dir()
        if visualize_dir is not None:
            ensure_directory(visualize_dir)

        model = build_model_adapter(self.config.model)

        batch_size = self.config.runtime.batch_size
        num_workers = self.config.runtime.num_workers
        if batch_size > 1 and hasattr(model, "predict_batch"):
            results = self._run_batched(samples, model, visualize_dir, batch_size)
        elif num_workers > 1:
            results = self._run_parallel(samples, model, visualize_dir, num_workers)
        else:
            results = self._run_sequential(samples, model, visualize_dir)

        model.close()

        if self._benchmark == _BENCHMARK_SSP:
            from cuactspot.screenspot_pro import build_screenspot_pro_report
            report = build_screenspot_pro_report(results)
        elif self._benchmark == _BENCHMARK_UV:
            from cuactspot.ui_vision import build_ui_vision_report
            report = build_ui_vision_report(results)
        else:
            report = build_report(results)
        report_path = resolve_output_path(output_dir, self.config.output.report_file)
        write_json(report_path, report)
        return report

    def _run_sequential(
        self,
        samples: list[DatasetSample],
        model,
        visualize_dir: Optional[Path],
    ) -> list[SampleEvaluationResult]:
        results: list[SampleEvaluationResult] = []
        progress = (
            tqdm(
                samples,
                total=len(samples),
                desc=f"Evaluating {self.config.model.name}",
                dynamic_ncols=True,
            )
            if tqdm is not None
            else None
        )
        sample_iterator = progress if progress is not None else samples
        try:
            for sample in sample_iterator:
                result = self._evaluate_single(sample, model, visualize_dir)
                results.append(result)
        finally:
            if progress is not None:
                progress.close()
        return results

    def _run_batched(
        self,
        samples: list[DatasetSample],
        model,
        visualize_dir: Optional[Path],
        batch_size: int,
    ) -> list[SampleEvaluationResult]:
        results: list[SampleEvaluationResult] = []
        progress = (
            tqdm(
                total=len(samples),
                desc=f"Evaluating {self.config.model.name} (bs={batch_size})",
                dynamic_ncols=True,
            )
            if tqdm is not None
            else None
        )
        try:
            for i in range(0, len(samples), batch_size):
                batch = samples[i : i + batch_size]
                batch_results = self._evaluate_batch(batch, model, visualize_dir)
                results.extend(batch_results)
                if progress is not None:
                    progress.update(len(batch))
        finally:
            if progress is not None:
                progress.close()
        return results

    def _evaluate_batch(
        self,
        batch: list[DatasetSample],
        model,
        visualize_dir: Optional[Path],
    ) -> list[SampleEvaluationResult]:
        start_time = perf_counter()
        batch_results: list[SampleEvaluationResult] = []
        try:
            predictions = model.predict_batch(batch)
            duration_sec = perf_counter() - start_time
            per_sample_duration = duration_sec / len(batch)
            for sample, prediction in zip(batch, predictions):
                passed, error_message = self.evaluator.evaluate(
                    sample, prediction.coordinates,
                )
                result = _result_from_prediction(
                    sample=sample,
                    sample_id=sample.sample_id,
                    task=sample.task_text,
                    modality=sample.modality,
                    passed=passed,
                    error_message=error_message,
                    duration_sec=per_sample_duration,
                    prediction=prediction,
                )
                if visualize_dir is not None:
                    draw_coordinates(
                        sample,
                        prediction.coordinates,
                        visualize_dir / f"{sample.sample_id}_out.png",
                    )
                batch_results.append(result)
        except Exception as exc:
            if not self.config.runtime.continue_on_error:
                raise
            duration_sec = perf_counter() - start_time
            for sample in batch:
                batch_results.append(SampleEvaluationResult(
                    sample_id=sample.sample_id,
                    passed=False,
                    error_message=str(exc),
                    coordinates=[],
                    raw_output="",
                    task=sample.task_text,
                    modality=sample.modality,
                    duration_sec=duration_sec / len(batch),
                    model_input={},
                    metadata={"exception_type": type(exc).__name__, **sample.metadata},
                ))
        return batch_results

    def _run_parallel(
        self,
        samples: list[DatasetSample],
        model,
        visualize_dir: Optional[Path],
        num_workers: int,
    ) -> list[SampleEvaluationResult]:
        results_map: dict[str, SampleEvaluationResult] = {}
        progress = (
            tqdm(
                total=len(samples),
                desc=f"Evaluating {self.config.model.name} ({num_workers}w)",
                dynamic_ncols=True,
            )
            if tqdm is not None
            else None
        )

        try:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                future_to_sample = {
                    executor.submit(
                        self._evaluate_single, sample, model, visualize_dir
                    ): sample
                    for sample in samples
                }
                for future in as_completed(future_to_sample):
                    sample = future_to_sample[future]
                    try:
                        result = future.result()
                    except Exception as exc:
                        result = SampleEvaluationResult(
                            sample_id=sample.sample_id,
                            passed=False,
                            error_message=str(exc),
                            coordinates=[],
                            raw_output="",
                            task=sample.task_text,
                            modality=sample.modality,
                            duration_sec=0.0,
                            model_input={},
                            metadata={"exception_type": type(exc).__name__, **sample.metadata},
                        )
                    results_map[sample.sample_id] = result
                    if progress is not None:
                        progress.update(1)
        finally:
            if progress is not None:
                progress.close()

        return [results_map[s.sample_id] for s in samples if s.sample_id in results_map]

    def _evaluate_single(
        self,
        sample: DatasetSample,
        model,
        visualize_dir: Optional[Path],
    ) -> SampleEvaluationResult:
        best_of_n = self.config.runtime.best_of_n
        if best_of_n > 1:
            return self._evaluate_best_of_n(sample, model, visualize_dir, best_of_n)
        return self._evaluate_once(sample, model, visualize_dir)

    def _evaluate_once(
        self,
        sample: DatasetSample,
        model,
        visualize_dir: Optional[Path],
    ) -> SampleEvaluationResult:
        start_time = perf_counter()
        try:
            prediction = model.predict(sample)
            passed, error_message = self.evaluator.evaluate(
                sample,
                prediction.coordinates,
            )
            duration_sec = perf_counter() - start_time
            result = _result_from_prediction(
                sample=sample,
                sample_id=sample.sample_id,
                task=sample.task_text,
                modality=sample.modality,
                passed=passed,
                error_message=error_message,
                duration_sec=duration_sec,
                prediction=prediction,
            )
            if visualize_dir is not None:
                draw_coordinates(
                    sample,
                    prediction.coordinates,
                    visualize_dir / f"{sample.sample_id}_out.png",
                )
            return result
        except Exception as exc:
            if not self.config.runtime.continue_on_error:
                raise
            duration_sec = perf_counter() - start_time
            return SampleEvaluationResult(
                sample_id=sample.sample_id,
                passed=False,
                error_message=str(exc),
                coordinates=[],
                raw_output="",
                task=sample.task_text,
                modality=sample.modality,
                duration_sec=duration_sec,
                model_input={},
                metadata={"exception_type": type(exc).__name__, **sample.metadata},
            )

    def _resolve_visualize_dir(self) -> Optional[Path]:
        if not self.config.output.visualize_dir:
            return None
        return Path(self.config.output.visualize_dir)

    def _evaluate_best_of_n(
        self,
        sample: DatasetSample,
        model,
        visualize_dir: Optional[Path],
        n: int,
    ) -> SampleEvaluationResult:
        temperature = self.config.runtime.best_of_temperature
        orig_gen_kwargs = dict(model.generation_kwargs)
        model.generation_kwargs["temperature"] = temperature

        best_result: Optional[SampleEvaluationResult] = None
        trial_outputs: list[str] = []
        start_time = perf_counter()
        try:
            for trial in range(n):
                try:
                    prediction = model.predict(sample)
                    passed, error_message = self.evaluator.evaluate(
                        sample,
                        prediction.coordinates,
                    )
                    trial_outputs.append(prediction.raw_output)
                    result = _result_from_prediction(
                        sample=sample,
                        sample_id=sample.sample_id,
                        task=sample.task_text,
                        modality=sample.modality,
                        passed=passed,
                        error_message=error_message,
                        duration_sec=perf_counter() - start_time,
                        prediction=prediction,
                    )
                    if passed:
                        result.metadata["best_of_n"] = n
                        result.metadata["best_of_trial"] = trial + 1
                        result.metadata["best_of_temperature"] = temperature
                        result.metadata["trial_outputs"] = trial_outputs
                        if visualize_dir is not None:
                            draw_coordinates(
                                sample,
                                prediction.coordinates,
                                visualize_dir / f"{sample.sample_id}_out.png",
                            )
                        return result
                    if best_result is None:
                        best_result = result
                except Exception as exc:
                    if not self.config.runtime.continue_on_error:
                        raise
                    trial_outputs.append(f"[error: {exc}]")
                    if best_result is None:
                        best_result = SampleEvaluationResult(
                            sample_id=sample.sample_id,
                            passed=False,
                            error_message=str(exc),
                            coordinates=[],
                            raw_output="",
                            task=sample.task_text,
                            modality=sample.modality,
                            duration_sec=perf_counter() - start_time,
                            model_input={},
                            metadata={"exception_type": type(exc).__name__},
                        )
        finally:
            model.generation_kwargs = orig_gen_kwargs

        assert best_result is not None
        best_result.duration_sec = perf_counter() - start_time
        best_result.metadata["best_of_n"] = n
        best_result.metadata["best_of_trial"] = n
        best_result.metadata["best_of_temperature"] = temperature
        best_result.metadata["trial_outputs"] = trial_outputs
        return best_result


def _result_from_prediction(
    sample,
    sample_id: str,
    task: str,
    modality: Optional[str],
    passed: bool,
    error_message: str,
    duration_sec: float,
    prediction: ModelPrediction,
) -> SampleEvaluationResult:
    return SampleEvaluationResult(
        sample_id=sample_id,
        passed=passed,
        error_message=error_message,
        coordinates=prediction.coordinates,
        raw_output=prediction.raw_output,
        task=task,
        modality=modality,
        duration_sec=duration_sec,
        model_input=make_jsonable(
            {
                "sample_id": sample.sample_id,
                "image_path": str(sample.image_path),
                **prediction.request_metadata,
            }
        ),
        metadata={
            "request_metadata": prediction.request_metadata,
            "response_metadata": prediction.response_metadata,
            "prediction_metadata": prediction.metadata,
            **sample.metadata,
        },
    )