# app/nodes/extract_symptoms.py

from langchain_openai import ChatOpenAI
from app.state import PatientState, REQUIRED_FIELDS, merge_checklist
from app.schemas import ExtractedSymptoms
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-5.4-mini"),    #gpt-4o-mini
    api_key=os.getenv("OPENAI_KEY"),
)


def _state_get(state: PatientState, key: str, default=None):
    if isinstance(state, dict):
        return state.get(key, default)
    return getattr(state, key, default)

def extract_symptoms(state: PatientState) -> dict:
    checklist = _state_get(state, "symptom_checklist", {f: False for f in REQUIRED_FIELDS})

    print(not isinstance(checklist, dict))
    if not isinstance(checklist, dict):
        checklist = {f: False for f in REQUIRED_FIELDS}

    uncovered = [f for f, done in checklist.items() if not done]
    covered   = [f for f, done in checklist.items() if done]
    existing_field_values = _state_get(state, "field_values", {}) or {}
    existing_symptoms = _state_get(state, "symptoms", []) or []
    transcription = _state_get(state, "transcription", "") or _state_get(state, "audio_input", "") or ""
    print(f"Covered: {covered}")
    print(f"Still needed: {uncovered}")

    # history = ""
    # if state.symptoms:
    #     history = f"\nDo NOT extract these symptoms that are already collected: {', '.join(state.symptoms)}"

    prompt = f"""
You are SynapsAgent's medical intake extractor for a demo app.

Latest patient speech:
{transcription}

Already captured symptoms:
{existing_symptoms}

Already covered fields. Do not ask about these again unless the patient provides new detail:
{covered}

Existing field values:
{existing_field_values}

Still needed fields:
{uncovered}

Required fields:
{REQUIRED_FIELDS}

Extract only new neurological symptoms from the latest speech.
For every required field the patient addressed in this latest speech, include that field in fields_covered.
field_values must be concise plain text in the same order as fields_covered.
Capture denials clearly as "none" or "denied" when the patient denies a field.
Do not invent details. If all fields are covered, followup_question must be null.
If fields remain missing, ask one concise follow-up question for the most important missing field.
Recommend mri_recommendation yes/no when enough severity information exists; otherwise use null.
Prefer recommending MRI when there are focal neurologic symptoms, seizures, progressive symptoms,
significant memory/speech changes, vision changes, balance issues, or persistent/worsening headaches.
"""

    raw = llm.with_structured_output(ExtractedSymptoms).invoke(prompt)
    new_symptoms = [s for s in raw.symptoms if s and s not in existing_symptoms]
    print(f"New symptoms: {raw.symptoms}")
    print(f"fields_covered: {raw.fields_covered}")
    print(f"field_values:   {raw.field_values}")

    # Zip fields_covered + field_values into a dict
    # Pad or trim field_values to match fields_covered length safely
    values = list(raw.field_values or [])
    if len(values) < len(raw.fields_covered):
        values.extend([""] * (len(raw.fields_covered) - len(values)))

    field_values_dict = {}
    for i, field in enumerate(raw.fields_covered or []):
        if field not in REQUIRED_FIELDS:
            continue
        value = values[i].strip() if i < len(values) and isinstance(values[i], str) else ""
        field_values_dict[field] = value or "Mentioned, details not captured"

    print(f"field_values_dict: {field_values_dict}")

    # Derive covered fields — keys of field_values_dict
    checklist_patch   = {f: True for f in field_values_dict}
    updated_checklist = merge_checklist(checklist, checklist_patch)
    missing           = [f for f, done in updated_checklist.items() if not done]

    print(f"all_covered: {set(field_values_dict.keys())}")
    print(f"missing: {missing}")

    # Preserve mri_recommendation once set
    if raw.mri_recommendation is not None:
        mri_rec = raw.mri_recommendation
        mri_reason = raw.mri_recommendation_reason
    else:
        mri_rec = state.mri_recommendation
        mri_reason = state.mri_recommendation_reason

    print(raw.mri_recommendation)
    print(raw.mri_recommendation_reason)
    print(f"{mri_rec}, {mri_reason}")

    return {
        "symptoms": new_symptoms,
        "symptom_checklist": updated_checklist,
        "field_values":      field_values_dict,
        "missing_fields": missing,
        "followup_question": None if not missing else raw.followup_question,
        "mri_recommendation": mri_rec,
        "mri_recommendation_reason": mri_reason,
        "mri_requested": (
            raw.mri_recommendation
            if raw.mri_recommendation is not None
            else _state_get(state, "mri_requested", False)
        ),
        # "answers":{f: "denied" for f in raw.fields_covered if f not in (state.symptoms or [])}
    }
    

# prompt = "Hi Doctor, I'm here because I've been having headaches and some neurological symptoms that I wanted to get checked out. I've been getting headaches about twice a week for a month now. The pain is usually around 6 out of 10 and it feels throbbing. I don't have any vision issues or nausea or vomiting problems. Neither do I have balanced problems but I did notice I have some memory issues and speech trouble. I've been forgetting words and losing track of conversations but also having trouble finding words and trouble getting sentences out."
# prompt = "I have no headaches, just a stomachache."
# fake_state = PatientState(
#         patient_id= "P-001",
#         transcription= prompt)

# result = extract_symptoms(fake_state)
