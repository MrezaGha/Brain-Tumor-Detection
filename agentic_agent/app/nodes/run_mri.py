"""
app/nodes/run_mri.py

LangGraph node: Run the MRI classifier.

Flow:
  1. Checks whether app/scans/<patient_id>.<ext> exists on disk.
  2. If the scan has NOT been uploaded yet → calls interrupt(), which pauses the graph. The API route saves the file and then resumes.
  3. Once the scan is present → calls predict_mri() and writes the full result dict into state.mri_result.
"""

from pathlib import Path
from typing import Optional

from langgraph.types import interrupt

from app.state import PatientState
from app.tools.mri_classifier import predict_mri

SCANS_DIR       = Path("app/scans")
CHECKPOINT_PATH = "app/models/efficientnet_best.pt"
SUPPORTED_EXT   = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]


def _find_scan(patient_id: str) -> Optional[Path]:
    """Return the scan path if it exists on disk, else None."""
    for ext in SUPPORTED_EXT:
        p = SCANS_DIR / f"{patient_id}{ext}"
        if p.is_file():
            return p
    return None


def run_mri_node(state: PatientState) -> dict:
    """
    LangGraph node function.

    Called automatically by graph.py after doctor_approval when
    state.mri_requested is True.

    Returns a partial-state dict:
        { "mri_result": <prediction dict>, "status": "complete" }
    """
    scan_path = _find_scan(state.patient_id)
    print(f"Scan path: {scan_path}")

    if scan_path is None:
        # Pause graph here – the API route will resume once scan is uploaded
        interrupt({
            "action":     "upload_mri_scan",
            "patient_id": state.patient_id,
            "message":    (
                f"MRI scan not found. Upload file as "
                f"app/scans/{state.patient_id}.<ext> and call "
                f"POST /patient/{state.patient_id}/mri-scan to resume."
            ),
        })

    result = predict_mri(
        image_path=str(scan_path),
        checkpoint_path=CHECKPOINT_PATH,
    )
    print(f"MRI image classified: {result}")
    return {
        "mri_result": result,   # full dict: class, confidence, probabilities
        "status":     "complete",
        "mri_recommendation": state.mri_recommendation,
        "mri_recommendation_reason": state.mri_recommendation_reason,
    }