"""
app/nodes/save_results.py

LangGraph node: Persist the full PatientState to disk.

Always the final node in the graph (runs whether or not an MRI was taken).
Writes app/results/<patient_id>.json so the doctor UI / EHR can
retrieve the complete record by patient_id.
"""

import json
from pathlib import Path

from app.state import PatientState

RESULTS_DIR = Path("app/results")


def save_results_node(state: PatientState) -> dict:
    """
    LangGraph node function.

    Serializes the entire PatientState and writes it to: app/results/<patient_id>.json

    Returns:
        { "status": "complete" | existing status }
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    out_path = RESULTS_DIR / f"{state.patient_id}.json"

    with open(out_path, "w") as f:
        json.dump(state.model_dump(), f, indent=2, default=str)

    # Set "pending" => "complete"; preserve any other status already set
    final_status = state.status if state.status != "pending" else "complete"

    return {
        "status": final_status,
        "mri_recommendation": state.mri_recommendation,
        "mri_recommendation_reason": state.mri_recommendation_reason,
    }
