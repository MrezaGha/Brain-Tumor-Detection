# app/schemas.py
from pydantic import BaseModel, Field, field_validator

from app.state import REQUIRED_FIELDS

# class FieldValue(BaseModel):
#     field: str = Field(description=f"Must be one of: {REQUIRED_FIELDS}")
#     value: str = Field(description="Short plain-text answer the patient gave.")

class ExtractedSymptoms(BaseModel):
    symptoms: list[str] = Field(
        default_factory=list,
        description="List of neurological symptoms the patient explicitly stated. "
            "Only extract NEW symptoms from the latest response — do NOT repeat symptoms "
            "already listed in the conversation history."
    )
    fields_covered: list[str] = Field(
        default_factory=list,
        description=(
           f"List of fields the patient addressed in this response (even to deny). "
            f"Only include fields from this list: {REQUIRED_FIELDS}"
        )
    )
    field_values: list[str] = Field(
        default_factory=list,
        description=(
            "The patient's short answer for each field in fields_covered, "
            "in the SAME ORDER as fields_covered. "
            "E.g. if fields_covered=['headache_frequency','vision_changes'] "
            "then field_values=['twice a week','none']. "
            "Must have the same length as fields_covered."
        )
    )
    # missing_fields: list[str] = Field(
    #     default = [],
    #     description=(
    #         "Neurological fields not mentioned by the patient after reviewing the FULL symptom history. "
    #         "Only include fields the patient has NOT yet addressed. "
    #         "Check for these and add any that are missing: "
    #         "headache_frequency, headache_severity, vision_changes, "
    #         "seizures, memory_loss, speech_difficulty, balance_issues, "
    #         "nausea_vomiting, symptom_duration, prior_brain_conditions, "
    #         "family_history_cancer, current_medications. "
            
    #     )
    # )
    followup_question: str | None = Field(
        default=None,
        description=(
            "One single follow-up question targeting the most critical unchecked field"
            # "Do NOT ask about anything already covered in the symptom history. "
            "If all fields are covered, return null."
        )
    )
    mri_recommendation: bool | None = Field(
        default=None,
        description=(
            "Whether the backend recommends requesting an MRI based on symptom severity. "
            "Use null if there is not enough information yet."
        ),
    )
    mri_recommendation_reason: str | None = Field(
        default=None,
        description="Short reason for the MRI recommendation, or null if unavailable.",
    )

    @field_validator("symptoms")
    def symptoms_not_empty(cls, v):
        # if not v:
        #     raise ValueError("No symptoms could be extracted from audio")
        if not v:
            return []
        return [s.lower().strip() for s in v]  # normalize casing

    @field_validator("fields_covered")
    def fields_must_be_required(cls, v):
        if not v:
            return []
        return [f.strip() for f in v if isinstance(f, str) and f.strip() in REQUIRED_FIELDS]

    @field_validator("field_values")
    def field_values_are_clean_strings(cls, v):
        if not v:
            return []
        return [str(value).strip() for value in v]

    @field_validator("mri_recommendation_reason")
    def reason_is_clean(cls, v):
        return v.strip() if isinstance(v, str) and v.strip() else None
