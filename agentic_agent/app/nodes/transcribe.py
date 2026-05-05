# app/nodes/transcribe.py
import whisper
from app.state import PatientState

print("Loading Whisper model...")
model = whisper.load_model("base.en")  # load once at startup
print("Whisper model loaded.")

def transcribe_audio(state: PatientState) -> dict:
    audio_path = state.audio_input
    print(f"transcribe_audio called — path: {audio_path}")
    
    print("Running Whisper transcription...")
    result = model.transcribe(audio_path, language="en")
    print(f"Transcription done: {result['text']}")
    
    return {"transcription": result["text"]}
    