"""
app/nodes/doctor_review.py

LangGraph node – always runs after summary_node.

Uses interrupt() to pause the graph and present the AI prediction
to the doctor. The doctor either confirms the diagnosis or provides
a corrected one via POST /patient/{patient_id}/doctor-review.

When the route resumes the graph with Command(resume={...}), this
node receives the doctor's input, updates the summary, and marks
the review as complete.

doctor_review input schema:
    {
        "confirmed_diagnosis": str,   # doctor's final call — must be one of:
                                      # "glioma" | "meningioma" |
                                      # "notumor" | "pituitary"
        "notes": str | None           # optional free-text remarks
    }
"""

from langgraph.types import interrupt

from app.state import PatientState

VALID_DIAGNOSES = {"glioma", "meningioma", "notumor", "pituitary"}


def doctor_review_node(state: PatientState) -> dict:
    """
    LangGraph node function.

    1. Interrupts graph execution and surfaces the AI prediction to the doctor (via the interrupt payload).
    2. Resumes when the doctor submits their review.
    3. Compares confirmed_diagnosis to the AI prediction and sets diagnosis_corrected = True if they differ.
    4. Merges the doctor's input back into the summary dict.
    """
    predicted = (
        state.mri_result.get("class") if state.mri_result else None
    )

    # Pause here until doctor submits review
    review = interrupt({
        "action":               "doctor_diagnosis_review",
        "patient_id":           state.patient_id,
        "predicted_diagnosis":  predicted,
        "confidence":           state.mri_result.get("confidence") if state.mri_result else None,
        "probabilities":        state.mri_result.get("probabilities") if state.mri_result else None,
        "symptoms":             state.symptoms,
        "field_values":         state.field_values,
        "instructions": (
            "Confirm the predicted diagnosis or provide a corrected one. "
            f"Valid values: {sorted(VALID_DIAGNOSES)}. "
            "Optionally include free-text notes."
        ),
    })
    # Graph resumes here with doctor's input

    confirmed_diagnosis: str | None = review.get("confirmed_diagnosis") or predicted
    if isinstance(confirmed_diagnosis, str):
        confirmed_diagnosis = confirmed_diagnosis.strip().lower() or predicted
    if confirmed_diagnosis and confirmed_diagnosis not in VALID_DIAGNOSES:
        print(f"Invalid doctor diagnosis '{confirmed_diagnosis}', falling back to prediction '{predicted}'")
        confirmed_diagnosis = predicted

    doctor_notes:        str | None = review.get("notes")

    diagnosis_corrected: bool = (
        confirmed_diagnosis != predicted
        if (confirmed_diagnosis and predicted)
        else False
    )

    final_diagnosis: str | None = confirmed_diagnosis or predicted

    # Merge doctor's review into the existing summary dict
    updated_summary = dict(state.summary or {})
    updated_summary.update({
        "final_diagnosis":         final_diagnosis,
        "doctor_confirmed_diagnosis": confirmed_diagnosis,
        "diagnosis_corrected":     diagnosis_corrected,
        "doctor_notes":            doctor_notes,
        "doctor_review_completed": True,
    })

    return {
        "doctor_confirmed_diagnosis": confirmed_diagnosis,
        "diagnosis_corrected":        diagnosis_corrected,
        "final_diagnosis":            final_diagnosis,
        "doctor_notes":               doctor_notes,
        "summary":                    updated_summary,
        "status":                     "reviewed",
        "mri_recommendation":         state.mri_recommendation,
        "mri_recommendation_reason":   state.mri_recommendation_reason,
    }
