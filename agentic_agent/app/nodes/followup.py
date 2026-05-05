# app/nodes/followup.py
from langgraph.types import interrupt
from app.state import PatientState

def followup_node(state: PatientState) -> dict:
    missing = state.missing_fields
    question = state.followup_question

    if not missing or not question:
        print("Symptom intake complete — routing to doctor approval")
        return {
            "status": "ready_for_approval",
            "mri_recommendation": state.mri_recommendation,
            "mri_recommendation_reason": state.mri_recommendation_reason,
        }

    print(f"Follow-up: {question}")
    # re-trigger audio collection or text input here
    interrupt(question)
    return {
        "status": "awaiting_followup",
        "mri_recommendation": state.mri_recommendation,
        "mri_recommendation_reason": state.mri_recommendation_reason,
    }