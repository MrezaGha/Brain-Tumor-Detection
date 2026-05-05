# app/nodes/approval.py
from langgraph.types import interrupt, Command
from app.state import PatientState


def approval_node(state: PatientState) -> dict:
    # Graph pauses here — doctor sees this payload and decides
    decision = interrupt({
        "patient_id":        state.patient_id,
        "symptoms":          state.symptoms,
        # "answers":           state.answers,
        "field_values":      state.field_values,
        "mri_requested":     state.mri_requested,
        "mri_recommendation": state.mri_recommendation,
        "mri_recommendation_reason": state.mri_recommendation_reason,
        "summary":           state.summary,
        "action":            "Review patient summary. Awaiting doctor approval."
    })

    return {
        "doctor_approved": True,
        "status": "approved",
        "mri_recommendation": state.mri_recommendation,
        "mri_recommendation_reason": state.mri_recommendation_reason,
    }
