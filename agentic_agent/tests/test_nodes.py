# tests/test_nodes.py
from app.nodes.extract_symptoms import extract_symptoms
from app.nodes.followup import followup_node
from app.nodes.approval import approval_node

# --- test extract_symptoms ---
def test_extract_symptoms_fills_state():
    fake_state = {
        "patient_id": "P-001",
        "audio_input": "I have chest pain and shortness of breath"
    }
    result = extract_symptoms(fake_state)
    assert "symptoms" in result
    assert isinstance(result["symptoms"], list)
    assert len(result["symptoms"]) > 0

# --- test followup ---
def test_followup_skips_when_no_missing_fields():
    fake_state = {
        "missing_fields": [],
        "followup_question": None
    }
    result = followup_node(fake_state)
    assert result["status"] == "ready_for_approval"

# --- test approval ---
def test_approval_sets_approved_true():
    # Mock the interrupt — monkeypatch it to return "approve"
    from unittest.mock import patch
    fake_state = {
        "patient_id": "P-001",
        "symptoms": ["chest pain"],
        "answers": {},
        "mri_requested": False,
        "summary": None
    }
    with patch("app.nodes.approval.interrupt", return_value="approve"):
        result = approval_node(fake_state)
    assert result["doctor_approved"] == True
    assert result["status"] == "approved"