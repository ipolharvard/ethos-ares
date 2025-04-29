from contextlib import nullcontext
from functools import partial
from pathlib import Path

import polars as pl
import torch as th
from transformers import EncoderDecoderModel

from .constants import MAPPINGS_DIR
from .model import GPT2LMNoBiasModel


def setup_torch(device, dtype, seed=42):
    th.manual_seed(seed)
    device_type = "cuda" if "cuda" in device else "cpu"
    if dtype == "bfloat16" and device_type == "cuda" and not th.cuda.is_bf16_supported():
        print("WARNING: bfloat16 is not supported on this device, using float16 instead")
        dtype = "float16"
    if device_type == "cuda":
        th.cuda.manual_seed(seed)
        th.backends.cuda.matmul.allow_tf32 = True
        th.backends.cudnn.allow_tf32 = True
    ptdtype = {"float32": th.float32, "bfloat16": th.bfloat16, "float16": th.float16}[dtype]
    ctx = (
        nullcontext()
        if device_type == "cpu"
        else th.amp.autocast(device_type=device_type, dtype=ptdtype)
    )
    return ctx


def load_model_checkpoint(checkpoint_fp: str | Path, **kwargs) -> tuple[
    GPT2LMNoBiasModel | EncoderDecoderModel, dict]:
    """Load model from a checkpoint file, **kwargs are passed to `torch.load`."""
    checkpoint = th.load(checkpoint_fp, weights_only=False, **kwargs)

    if checkpoint["model_config"].is_encoder_decoder:
        model = EncoderDecoderModel(checkpoint["model_config"])
    else:
        model = GPT2LMNoBiasModel(checkpoint["model_config"])

    model.load_state_dict(checkpoint["model"])

    del checkpoint["model"]
    del checkpoint["model_config"]
    return model, checkpoint


def get_mimic_hf_patients(mimic_dir: str | Path) -> list[str]:
    if not (mimic_dir / "hosp").is_dir():
        raise FileNotFoundError(f"Expected, but not found: {mimic_dir / 'hosp'}")
    read_fn = pl.scan_parquet
    if not (mimic_dir / "hosp" / "diagnoses_icd.parquet").is_file():
        read_fn = partial(pl.scan_csv, infer_schema_length=None)
    return (
        (
            read_fn(mimic_dir / "hosp" / "diagnoses_icd.*")
            .filter(
                pl.col("icd_code").is_in(
                    read_fn(mimic_dir / "hosp" / "d_icd_diagnoses.*")
                    .select("icd_code")
                    .filter(pl.col("icd_code").str.slice(0, 3).is_in(["428", "I50"]))
                    .collect()
                    .to_series()
                )
            )
            .select(pl.col("subject_id").unique())
            .collect()
        )
        .to_series()
        .to_list()
    )


def get_mimic_sepsis_icu_stays() -> pl.DataFrame:
    return pl.read_csv(MAPPINGS_DIR / "mimic-iv_derived.csv.gz").filter("sepsis")
