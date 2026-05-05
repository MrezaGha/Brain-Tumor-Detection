"""
api/routes.py — All HTTP endpoints bridging the UI to the LangGraph backend.

Endpoints
---------
POST  /api/patient/start                  Start session; receives first audio blob
POST  /api/patient/{id}/audio             Submit follow-up audio; resumes graph
GET   /api/patient/{id}/status            Return lightweight status + followup_question
POST  /api/patient/{id}/upload-mri        Upload MRI scan; resumes graph at run_mri
GET   /api/patient/{id}/result            Return full PatientState as JSON
POST  /api/patient/{id}/approve           Doctor approves — resumes approval interrupt
PUT   /api/patient/{id}/update            Update summary + doctor notes
"""

import uuid
import shutil
import tempfile
from pathlib import Path
from typing import Optional
import asyncio
from functools import partial
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from langgraph.types import Command
from pydantic import BaseModel

from app.state import REQUIRED_FIELDS
from app.graph import graph

router = APIRouter()

SCANS_DIR  = Path("app/scans");  SCANS_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_DIR  = Path("app/audio");  AUDIO_DIR.mkdir(parents=True, exist_ok=True)


# ── Request models ───────────────────────────────────────────────────────────

class UpdateRequest(BaseModel):
    corrected_class: Optional[str] = None
    doctor_note: Optional[str] = None
    diagnosis_corrected: Optional[bool] = None


class DoctorReviewRequest(BaseModel):
    confirmed_diagnosis: Optional[str] = None
    notes: Optional[str] = None


# ── Helpers ──────────────────────────────────────────────────────────────────

def _config(patient_id: str) -> dict:
    return {"configurable": {"thread_id": patient_id}}

def _safe_invoke(config, payload):
    """Invoke graph; if it hits an interrupt that's expected, return current state."""
    try:
        return graph.invoke(payload, config)
    except Exception as e:
        print(f"graph.invoke() EXCEPTION: {e}")
        return graph.get_state(config).values

async def _async_invoke(config, payload):
    """Run blocking graph.invoke() in a thread so FastAPI stays responsive."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,                          # uses default ThreadPoolExecutor
        partial(_safe_invoke, config, payload)
    )


DISPLAY_CLASS = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "pituitary": "Pituitary Tumor",
    "notumor": "No Tumor",
    "no_tumor": "No Tumor",
    "no tumor": "No Tumor",
}


FIELD_LABELS = {
    "headache_frequency": "Headache Frequency",
    "headache_severity": "Headache Severity",
    "vision_changes": "Vision Changes",
    "seizures": "Seizures",
    "memory_loss": "Memory Loss",
    "speech_difficulty": "Speech Difficulty",
    "balance_issues": "Balance Issues",
    "nausea_vomiting": "Nausea / Vomiting",
    "symptom_duration": "Symptom Duration",
    "prior_brain_conditions": "Prior Brain Conditions",
    "family_history_cancer": "Family History of Cancer",
    "current_medications": "Current Medications",
}


def _display_class(label: Optional[str]) -> str:
    if not label:
        return "Pending"
    key = str(label).strip().lower()
    return DISPLAY_CLASS.get(key, key.replace("_", " ").title())


def _format_confidence(value) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "N/A"


def _summary_mri_result(summary: Optional[dict]) -> dict:
    if not isinstance(summary, dict):
        return {}
    mri = summary.get("mri_result")
    if isinstance(mri, dict):
        return mri
    return {
        "class": summary.get("predicted_class"),
        "confidence": summary.get("confidence"),
        "probabilities": summary.get("probabilities"),
    }


def _normalize_prediction(state: dict) -> dict:
    raw = state.get("mri_result")
    if not isinstance(raw, dict):
        raw = _summary_mri_result(state.get("summary"))
    raw = raw if isinstance(raw, dict) else {}

    raw_class = raw.get("class") or raw.get("predicted_class")
    confidence = raw.get("confidence")
    try:
        confidence = float(confidence) if confidence is not None else None
    except (TypeError, ValueError):
        confidence = None

    probabilities = raw.get("probabilities") or {}
    if not isinstance(probabilities, dict):
        probabilities = {}

    return {
        "class": raw_class,
        "display_class": _display_class(raw_class),
        "confidence": confidence,
        "confidence_pct": _format_confidence(confidence),
        "probabilities": probabilities,
    }


def _is_low_priority_value(value) -> bool:
    text = str(value or "").strip().lower()
    return text in {"", "none", "no", "denied", "not reported", "n/a", "na", "negative"}


def _field_priority(field: str, value: str, covered: bool) -> str:
    if not covered or _is_low_priority_value(value):
        return "low"
    if field in {"prior_brain_conditions", "family_history_cancer", "current_medications"}:
        return "medium"
    return "high"


def _build_patient_intake(state: dict) -> list[dict]:
    field_values = state.get("field_values") or {}
    checklist = state.get("symptom_checklist") or {}
    if not isinstance(field_values, dict):
        field_values = {}
    if not isinstance(checklist, dict):
        checklist = {}

    rows = []
    for index, field in enumerate(REQUIRED_FIELDS):
        covered = bool(checklist.get(field))
        raw_value = field_values.get(field)
        if raw_value is None or str(raw_value).strip() == "":
            value = "Mentioned, details not captured" if covered else "Not reported"
        else:
            value = str(raw_value).strip()
        priority = _field_priority(field, value, covered)
        rows.append({
            "field": field,
            "label": FIELD_LABELS.get(field, field.replace("_", " ").title()),
            "value": value,
            "covered": covered,
            "priority": priority,
            "_order": index,
        })

    rank = {"high": 0, "medium": 1, "low": 2}
    rows.sort(key=lambda row: (rank.get(row["priority"], 2), row["_order"]))
    for row in rows:
        row.pop("_order", None)
    return rows


def _state_summary(state: dict) -> dict:
    """Return the fields the UI polls for — lightweight, no binary data."""
    return {
        "patient_id": state.get("patient_id"),
        "symptoms": state.get("symptoms", []),
        "symptom_checklist": state.get("symptom_checklist", {f: False for f in REQUIRED_FIELDS}),
        # "answers":  state.get("answers", {}),
        "field_values": state.get("field_values", {}),
        "patient_intake": _build_patient_intake(state),
        "missing_fields": state.get("missing_fields", []),
        "followup_question": state.get("followup_question"),
        "transcription": state.get("transcription"),
        "doctor_approved": state.get("doctor_approved", False),
        "mri_requested": state.get("mri_requested", False),
        "mri_recommendation": state.get("mri_recommendation"),
        "mri_recommendation_reason": state.get("mri_recommendation_reason"),
        "mri_result": state.get("mri_result"),
        "ai_prediction": _normalize_prediction(state),
        "summary": state.get("summary"),
        "doctor_review_required": state.get("doctor_review_required", True),
        "doctor_confirmed_diagnosis": state.get("doctor_confirmed_diagnosis"),
        "diagnosis_corrected": state.get("diagnosis_corrected"),
        "final_diagnosis": state.get("final_diagnosis"),
        "doctor_notes": state.get("doctor_notes"),
        "status": state.get("status", "pending")

    }


# ── Routes ───────────────────────────────────────────────────────────────────

@router.post("/patient/start")
async def start_workflow(file: UploadFile = File(...)):
    """
    Receives the FIRST audio recording from the UI.
    Saves the audio, initialises PatientState, and runs the graph through
    extract_symptoms + followup. If missing_fields remain the graph pauses
    and followup_question is returned so the UI can prompt the next recording.
    """
    patient_id = str(uuid.uuid4())[:8]
    config = _config(patient_id)

    # Save audio
    suffix = Path(file.filename).suffix or ".webm"
    audio_path = AUDIO_DIR / f"{patient_id}_0{suffix}"
    with open(audio_path, "wb") as out:
        shutil.copyfileobj(file.file, out)

    initial_state = {
        "patient_id": patient_id,
        "audio_input": str(audio_path),
        "symptoms": [],
        # "answers": {},
        "field_values": {},
        "symptom_checklist": {f: False for f in REQUIRED_FIELDS},
        "missing_fields": [],
        "doctor_approved": False,
        "mri_requested": False,
        "mri_recommendation": None,
        "mri_recommendation_reason": None,
        "status": "pending"
    }

    print(f"Invoking graph for patient {patient_id}, audio: {audio_path}")
    # result = _safe_invoke(config, initial_state)
    result = await _async_invoke(config, initial_state)
    print(f"Graph invoke complete for {patient_id}")

    return JSONResponse({"patient_id": patient_id, "state": _state_summary(result)})


@router.post("/patient/{patient_id}/audio")
async def submit_audio(patient_id: str, file: UploadFile = File(...)):
    """
    Receives a FOLLOW-UP audio recording (answer to followup_question).
    Saves the audio and resumes the graph so the followup node can process it.
    The graph will loop back if more missing_fields remain, or advance to
    doctor_approval / MRI when the symptom picture is complete.
    """
    config = _config(patient_id)
    snapshot = graph.get_state(config)
    if not snapshot:
        raise HTTPException(404, "Patient session not found. Call /start first.")

    # Count existing audio files for this patient to number them sequentially
    existing = list(AUDIO_DIR.glob(f"{patient_id}_*"))
    n = len(existing)
    suffix = Path(file.filename).suffix or ".webm"
    audio_path = AUDIO_DIR / f"{patient_id}_{n}{suffix}"
    with open(audio_path, "wb") as out:
        shutil.copyfileobj(file.file, out)

    # Resume graph with the new audio path
    # result = _safe_invoke(config, {"audio_input": str(audio_path)})
    result = await _async_invoke(config, {"audio_input": str(audio_path)})
    return JSONResponse({"patient_id": patient_id, "state": _state_summary(result)})


@router.get("/patient/{patient_id}/status")
async def get_status(patient_id: str):
    """Lightweight poll endpoint — returns status + followup_question only."""
    config = _config(patient_id)
    snapshot = graph.get_state(config)
    if not snapshot:
        raise HTTPException(404, "Patient session not found.")
    return JSONResponse({"patient_id": patient_id, "state": _state_summary(snapshot.values)})

@router.put("/patient/{patient_id}/set-mri")
async def set_mri_requested(patient_id: str, body: dict):
    """Doctor overrides MRI recommendation — body: {'mri_requested': true/false}"""
    config = _config(patient_id)
    graph.update_state(config, {"mri_requested": body.get("mri_requested", False)})
    return JSONResponse({"mri_requested": body.get("mri_requested")})

@router.post("/patient/{patient_id}/upload-mri")
async def upload_mri(patient_id: str, file: UploadFile = File(...)):
    """Save MRI scan and resume graph at run_mri node."""
    config = _config(patient_id)
    snapshot = graph.get_state(config)
    if not snapshot:
        raise HTTPException(404, "Patient session not found. Start the workflow first.")

    suffix = Path(file.filename).suffix or ".jpg"
    scan_path = SCANS_DIR / f"{patient_id}{suffix}"
    with open(scan_path, "wb") as out:
        shutil.copyfileobj(file.file, out)
    print(f"MRI saved: {scan_path} — exists: {scan_path.exists()}")  # ← add this

    if not scan_path.exists():
        raise HTTPException(500, "MRI upload failed; saved file was not found.")

    result = await _async_invoke(config, Command(resume=True))
    return JSONResponse({"patient_id": patient_id, "state": _state_summary(result)})


@router.get("/patient/{patient_id}/result")
async def get_result(patient_id: str):
    """Return full PatientState."""
    config = _config(patient_id)
    snapshot = graph.get_state(config)
    if not snapshot:
        raise HTTPException(404, "Patient session not found.")
    return JSONResponse({"patient_id": patient_id, "state": snapshot.values})


@router.post("/patient/{patient_id}/approve")
async def approve_result(patient_id: str):
    """Doctor clicks Approve — resumes the approval interrupt."""
    config = _config(patient_id)
    snapshot = graph.get_state(config)
    if not snapshot:
        raise HTTPException(404, "Patient session not found.")
    # result = _safe_invoke(config, Command(resume=True))
    result = await _async_invoke(config, Command(resume=True))
    return JSONResponse({"patient_id": patient_id, "approved": True,
                         "state": _state_summary(result)})


@router.put("/patient/{patient_id}/update")
async def update_summary(patient_id: str, body: UpdateRequest):
    """Doctor edits summary fields + adds note. Writes to both summary dict and top-level state."""
    config = _config(patient_id)
    snapshot = graph.get_state(config)
    if not snapshot:
        raise HTTPException(404, "Patient session not found.")

    current = snapshot.values
    current_summary = dict(current.get("summary") or {})
    top_level_patch = {}

    if body.corrected_class is not None:
        current_summary["corrected_class"] = body.corrected_class
        current_summary["diagnosis_corrected"] = True
        top_level_patch["final_diagnosis"] = body.corrected_class
        top_level_patch["doctor_confirmed_diagnosis"] = body.corrected_class

    if body.doctor_note is not None:
        current_summary["doctor_note"] = body.doctor_note
        top_level_patch["doctor_notes"] = body.doctor_note   # matches state.py field

    if body.diagnosis_corrected is not None:
        current_summary["diagnosis_corrected"] = body.diagnosis_corrected

    graph.update_state(config, {"summary": current_summary, **top_level_patch})

    return JSONResponse({
        "patient_id": patient_id,
        "summary": current_summary,
        "message": "Summary updated successfully.",
    })


@router.post("/patient/{patient_id}/doctor-review")
async def doctor_review(patient_id: str, body: DoctorReviewRequest):
    """Final doctor review; resumes the doctor_review_node interrupt."""
    config = _config(patient_id)
    snapshot = graph.get_state(config)
    if not snapshot:
        raise HTTPException(404, "Patient session not found.")

    prediction = _normalize_prediction(snapshot.values)
    confirmed = body.confirmed_diagnosis or prediction["class"]
    result = await _async_invoke(config, Command(resume={
        "confirmed_diagnosis": confirmed,
        "notes": body.notes,
    }))
    return JSONResponse({"patient_id": patient_id, "state": _state_summary(result)})
