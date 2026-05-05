"""
app/nodes/summary.py

LangGraph node - Builds a structured summary dict and sets doctor_review_required based on the MRI prediction (or lack of one).

doctor_review_required always required.
"""

from app.state import PatientState


def summary_node(state: PatientState) -> dict:
    """
    LangGraph node function.

    Reads: symptoms, field_values, mri_result, doctor_approved
    Writes: summary (dict), doctor_review_required (bool)
    """
    mri = state.mri_result  # dict or None

    structured_summary = {
        "patient_id": state.patient_id,

        # Step 1 outputs
        "symptoms": state.symptoms,
        "field_values": state.field_values,
        "symptom_checklist": state.symptom_checklist,

        # MRI request decision and backend recommendation
        "mri_requested": state.mri_requested,
        "mri_recommendation": state.mri_recommendation,
        "mri_recommendation_reason": state.mri_recommendation_reason,

        # Step 3 output (None if no MRI was taken)
        "mri_result": {
            "predicted_class": mri.get("class"),
            "confidence":      mri.get("confidence"),
            "probabilities":   mri.get("probabilities"),
        } if mri else None,

        # Always requires doctor sign-off
        "doctor_review_required":  True,
        "doctor_review_completed": False,   # filled by doctor_review_node
        "final_diagnosis":         state.final_diagnosis,
        "doctor_notes":            state.doctor_notes,
    }

    return {
        "summary":                structured_summary,
        "doctor_review_required": True,
        "mri_recommendation":      state.mri_recommendation,
        "mri_recommendation_reason": state.mri_recommendation_reason,
    }
