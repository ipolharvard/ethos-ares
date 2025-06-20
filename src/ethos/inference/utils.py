import time
from collections.abc import Generator, Sequence
from pathlib import Path
from queue import Queue

import polars as pl
import torch as th
from loguru import logger
from torch.nn import functional as F

from ..datasets import (
    CriticalOutcomeAtTriageDataset,
    DrgPredictionDataset,
    EdReattendenceDataset,
    HospitalAdmissionAtTriageDataset,
    HospitalMortalityDataset,
    ICUAdmissionDataset,
    ICUMortalityDataset,
    ICUReadmissionDataset,
    ReadmissionDataset,
    SofaPredictionDataset,
)
from ..datasets.base import InferenceDataset
from ..datasets.extensions import create_single_trajectory_dataset
from .constants import Task


def get_dataset_cls(task: Task) -> type[InferenceDataset]:
    # TODO: Make extensions automatically, not as separate tasks
    match task:
        case Task.HOSPITAL_MORTALITY | Task.HOSPITAL_MORTALITY_SINGLE:
            dataset_cls = HospitalMortalityDataset
            if task == Task.HOSPITAL_MORTALITY_SINGLE:
                return create_single_trajectory_dataset(dataset_cls)
            return dataset_cls
        case Task.ICU_MORTALITY:
            return ICUMortalityDataset
        case Task.READMISSION:
            return ReadmissionDataset
        case Task.DRG_PREDICTION:
            return DrgPredictionDataset
        case Task.SOFA_PREDICTION:
            return SofaPredictionDataset
        case Task.ICU_READMISSION:
            return ICUReadmissionDataset
        case Task.ICU_ADMISSION | Task.ICU_ADMISSION_SINGLE:
            dataset_cls = ICUAdmissionDataset
            match task:
                case Task.ICU_ADMISSION_SINGLE:
                    return create_single_trajectory_dataset(dataset_cls)
                case _:
                    return dataset_cls
        case Task.ED_HOSPITALIZATION:
            return HospitalAdmissionAtTriageDataset
        case Task.ED_CRITICAL_OUTCOME:
            return CriticalOutcomeAtTriageDataset
        case Task.ED_REPRESENTATION:
            return EdReattendenceDataset
        case _:
            raise ValueError(f"Unknown task: {task}, available are {', '.join(Task)}")


def evaluate_dataset_subset(dataset: InferenceDataset, subset_size: int | float) -> tuple[int, str]:
    if subset_size is None:
        n_samples = len(dataset)
        token_suffix = ""
    elif subset_size <= 0:
        raise ValueError(f"Subset must be a positive number, got {subset_size}")
    elif subset_size < 1:
        n_samples = round(subset_size * len(dataset))
        token_suffix = f"_subset{subset_size}"
    else:
        n_samples = int(subset_size)
        token_suffix = f"_subset{n_samples}"
        if n_samples > len(dataset):
            raise ValueError(f"Subset size ({n_samples}) is larger than the dataset size")
    return n_samples, token_suffix


def producer(subsets: Sequence, queue: Queue, num_proc: int):
    for subset in subsets:
        queue.put(subset)

    # Poison pills
    for _ in range(num_proc):
        queue.put(None)


def create_loader(queue: Queue, dataset) -> Generator[tuple[th.Tensor, dict], None, None]:
    while True:
        indices = queue.get()
        if indices is None:
            break
        yield from (dataset[i] for i in indices)


def get_token_time(tokens: Sequence, vocab) -> th.Tensor:
    """Returns time in microseconds."""
    if isinstance(tokens, th.Tensor):
        tokens = tokens.view(-1).cpu()

    def decode(token):
        try:
            return vocab.decode(token)
        except KeyError:
            return None

    interval_estimates = vocab.interval_estimates["mean"]
    return th.tensor(
        [
            interval_estimates.get(t, 0) if t is not None else th.inf
            for t in (decode(t) for t in tokens)
        ]
    )


@th.no_grad()
def get_next_token(
    model,
    x: th.Tensor,
    ctx: th.Tensor | None = None,
    return_probs: bool = False,
    top_k: int | None = None,
    temperature: float = 1.0,
):
    if ctx is not None:
        logits = model(ctx, decoder_input_ids=x).logits
    else:
        logits = model(x).logits
    logits = logits[:, -1, :]
    logits /= temperature

    if top_k is not None:
        v, _ = th.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = -float("Inf")

    probs = F.softmax(logits, dim=-1)
    next_token = th.multinomial(probs, num_samples=1)
    if return_probs:
        return next_token, probs
    return next_token


def format_big_number(num: int) -> str:
    for factor, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "k")):
        if num >= factor:
            value = num / factor
            if value < 10:
                if suffix == "k":
                    return f"{num:.0f}"
                return f"{value:.2f}{suffix}"
            return f"{value:.1f}{suffix}"
    return f"{num:.0f}"


def write_results_to_parquet(
    output_dir: Path, results: list[dict], completed_sample_num: int | None = None
):
    output_dir.mkdir(exist_ok=True, parents=True)
    out_fn = "samples"
    if completed_sample_num is not None:
        out_fn += f"_[{completed_sample_num - len(results)}-{completed_sample_num})"
    out_fp = output_dir / out_fn

    pl.from_dicts(results, infer_schema_length=None).with_columns(
        pl.col("^.*token_time$").cast(pl.Duration),
        (
            pl.col("generated_tokens").cast(pl.List(pl.UInt16))
            if "generated_tokens" in results[0]
            else "expected"
        ),
    ).write_parquet(out_fp.with_suffix(".parquet"), use_pyarrow=True)


def wait_for_workers(output_dir: str | Path, sleep_time: int = 2):
    time_slept = 0
    output_dir = Path(output_dir)
    while any(output_dir.glob("*.lock")):
        time.sleep(sleep_time)
        time_slept += sleep_time
        if time_slept > 30:
            logger.warning(f"Waiting for: {list(output_dir.glob('*lock'))}")
