# tests/test_integration.py
import pytest
from dotenv import load_dotenv
from app.state import PatientState
from app.nodes.extract_symptoms import extract_symptoms

load_dotenv()  # loads your real API key from .env

def test_extract_symptoms_real_api():
    fake_state = PatientState(
        patient_id= "P-001",
        transcription= "I've been having really bad headaches every morning for the past 3 weeks."
    )

    result = extract_symptoms(fake_state)

    print("\n--- REAL API RESULT ---")
    print(f"Symptoms:          {result['symptoms']}")
    print(f"Missing fields:    {result['missing_fields']}")
    print(f"Followup question: {result['followup_question']}")

    assert len(result["symptoms"]) > 0