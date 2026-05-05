# app/state.py
from pydantic import BaseModel, Field
from typing import Optional, Annotated
from uuid import uuid4
import operator

def merge_dicts(a: dict, b: dict) -> dict:
    """Merge two dicts, b values override a."""
    return {**a, **b}

REQUIRED_FIELDS = [
    "headache_frequency", "headache_severity", "vision_changes", "seizures",
    "memory_loss", "speech_difficulty", "balance_issues", "nausea_vomiting",
    "symptom_duration", "prior_brain_conditions", "family_history_cancer",
    "current_medications"
]

def merge_checklist(a, b) -> dict:
    """Merge two checklist dicts — True values lock in and never revert."""
    if not isinstance(a, dict):
        a = {f: False for f in REQUIRED_FIELDS}
    if not isinstance(b, dict):
        b = {}
    return {**a, **{k: (a.get(k, False) or v) for k, v in b.items()}}

class PatientState(BaseModel):
    patient_id: str = Field(default_factory=lambda: str(uuid4()))
    audio_input: Optional[str] = None
    transcription: Optional[str] = None
    symptoms: Annotated[list[str], operator.add] = Field(default_factory=list)
    answers: Annotated[dict, merge_dicts] = Field(default_factory=dict)
    symptom_checklist:  Annotated[dict, merge_checklist]    = Field(default_factory=lambda: {f: False for f in REQUIRED_FIELDS})
    field_values: Annotated[dict, merge_dicts] = Field(default_factory=dict)
    missing_fields: list[str] = Field(default_factory=list)
    followup_question: Optional[str] = None
    doctor_approved: bool = False
    mri_requested: bool = False
    mri_recommendation: Optional[bool] = None
    mri_recommendation_reason: Optional[str] = None
    mri_result: Optional[dict] = None
    summary: Optional[dict] = None
    doctor_review_required: bool = True  # Always true
    doctor_confirmed_diagnosis: Optional[str] = None
    diagnosis_corrected: Optional[bool] = None
    final_diagnosis: Optional[str] = None
    doctor_notes: Optional[str] = None
    status: str = "pending"