"""ED-BENCHMARK paper: https://www.nature.com/articles/s41597-022-01782-9"""

from abc import ABC
from datetime import timedelta
from pathlib import Path

import torch as th

from ..constants import SpecialToken as ST
from .base import InferenceDataset


class _InferenceAtTriageDataset(InferenceDataset, ABC):
    """The base class for datasets that generate patient timelines ending at the last token related
    to triage at the ED."""

    outcome_indices: th.Tensor

    def __init__(self, input_dir: str | Path, n_positions: int = 2048, **kwargs):
        super().__init__(input_dir, n_positions, **kwargs)
        self.ed_reg_indices = self._get_indices_of_stokens(ST.ED_ADMISSION)
        self.start_indices = self._move_indices_to_last_same_time(self.ed_reg_indices)

    def __len__(self) -> int:
        return len(self.start_indices)

    def __getitem__(self, idx) -> tuple[th.Tensor, dict]:
        ed_reg_idx = self.ed_reg_indices[idx]
        start_idx = self.start_indices[idx]
        outcome_idx = self.outcome_indices[idx]

        ed_hadm_id = self._get_hadm_id(ed_reg_idx)
        return super().__getitem__(start_idx), {
            "true_token_dist": (outcome_idx - start_idx).item(),
            "true_token_time": (self.times[outcome_idx] - self.times[start_idx]).item(),
            "patient_id": self.patient_id_at_idx[start_idx].item(),
            "hadm_id": ed_hadm_id,
            "prediction_time": self.times[start_idx].item(),
            "data_idx": start_idx.item(),
        }


class HospitalAdmissionAtTriageDataset(_InferenceAtTriageDataset):
    """Generates patient timelines ending at the last token corresponding to the result of the
    patient's triage. The target variable indicates whether the patient was admitted to the
    hospital.

    Reference from the ED-BENCHMARK paper:
        "The hospitalization outcome is defined as an inpatient care site admission immediately
        following an ED visit. Patients who transitioned to ED observation were not considered
        hospitalized unless they were eventually admitted to the hospital. As hospital beds are
        limited, this outcome indicates resource utilization and may facilitate resource allocation
        efforts. The hospitalization outcome also reflects patient acuity to a limited extent, as
        hospitalized patients represent a broad spectrum of disease severity."
    """

    def __init__(self, input_dir: str | Path, n_positions: int = 2048, **kwargs):
        super().__init__(input_dir, n_positions, **kwargs)
        self.stop_stokens = [ST.ADMISSION] + self.stop_stokens
        adm_indices = self._get_indices_of_stokens(ST.ADMISSION)
        # TODO: Explain why fill with 0. It doesn't make sense but it does the job.
        self.outcome_indices = self._match(adm_indices, self.ed_reg_indices, fill_unmatched=0)

    def __getitem__(self, idx) -> tuple[th.Tensor, dict]:
        x, y = super().__getitem__(idx)
        expected = y["hadm_id"] is not None
        return x, {"expected": expected, **y}


class CriticalOutcomeAtTriageDataset(_InferenceAtTriageDataset):
    """Generates patient timelines ending at the last token corresponding to the result of the
    patient's triage. The target variable indicates whether the patient was admitted to the ICU or
    died.

    Reference from the ED-BENCHMARK paper:
        "The critical outcome is compositely defined as either inpatient mortality or transfer to an
        ICU within 12 hours. This outcome represents critically ill patients who require ED
        resources urgently and may suffer from poorer health outcomes if care is delayed. Predicting
        the critical outcome at ED triage may enable physicians to allocate ED resources efficiently
        and intervene on high-risk patients promptly."
    """

    time_limit = timedelta(hours=12)

    def __init__(self, input_dir: str | Path, n_positions: int = 2048, **kwargs):
        super().__init__(input_dir, n_positions, **kwargs)
        self.stop_stokens = [ST.ICU_ADMISSION, ST.DISCHARGE] + self.stop_stokens
        stop_stoken_indices = self._get_indices_of_stokens(self.stop_stokens)
        self.outcome_indices = self._match(
            stop_stoken_indices, self.ed_reg_indices, fill_unmatched=0
        )
        self.dth_indices = self._get_indices_of_stokens(ST.DEATH)

    def __getitem__(self, idx) -> tuple[th.Tensor, dict]:
        x, y = super().__getitem__(idx)
        outcome_idx = self.outcome_indices[idx]
        return x, {"expected": self.vocab.decode(self.tokens[outcome_idx]), **y}


class EdReattendenceDataset(InferenceDataset):
    """Generates patient timelines ending at the token corresponding to the result of the patient's
    ED discharge.

    Notes:
        - ED visits resulting in hospital admission are excluded, as the focus is on cases where
          patients were incorrectly not admitted to the hospital. This differs from the approach
          outlined in the ED-BENCHMARK paper.
        - If ``expected`` is set to False, the ``true_token_time`` represents the observation time
        from the start token.

    Reference from the ED-BENCHMARK paper:
        "The ED reattendance outcome refers to a patient’s return visit to the ED within 72 hours
        after their previous discharge. It is a widely used indicator of the quality of care and
        patient safety, believed to reflect cases where patients may not have been adequately
        triaged during their initial emergency visit."
    """

    time_limit = timedelta(hours=72)

    def __init__(self, input_dir: str | Path, n_positions: int = 2048, **kwargs):
        super().__init__(input_dir, n_positions, **kwargs)
        self.stop_stokens = [ST.ED_ADMISSION] + self.stop_stokens

        ed_out_indices = self._get_indices_of_stokens(ST.ED_DISCHARGE)
        ed_out_indices = th.tensor(
            [idx for idx in ed_out_indices if self._get_hadm_id(idx) is None]
        )

        self.start_indices = self._move_indices_to_last_same_time(ed_out_indices)

        ed_reg_or_end_indices = self._get_indices_of_stokens([ST.ED_ADMISSION, ST.TIMELINE_END])
        self.outcome_indices = self._match(ed_reg_or_end_indices, ed_out_indices)

    def __len__(self) -> int:
        return len(self.start_indices)

    def __getitem__(self, idx) -> tuple[th.Tensor, dict]:
        start_idx = self.start_indices[idx]
        outcome_idx = self.outcome_indices[idx]

        return super().__getitem__(start_idx), {
            "expected": self.vocab.decode(self.tokens[outcome_idx].item()) == ST.ED_ADMISSION,
            "true_token_dist": (outcome_idx - start_idx).item(),
            "true_token_time": (self.times[outcome_idx] - self.times[start_idx]).item(),
            "patient_id": self.patient_id_at_idx[start_idx].item(),
            "prediction_time": self.times[start_idx].item(),
            "data_idx": start_idx.item(),
        }
