# app/nodes/audio_input.py
import io
import numpy as np
import threading
import sounddevice as sd
from scipy.io.wavfile import write
from app.state import PatientState

SAMPLE_RATE = 16000
SILENCE_DURATION = 2.0   # seconds of silence before stopping
THRESHOLD = 0.01          # amplitude floor for voice activity

def record_audio(state: PatientState) -> dict:
    print("Listening... (speak your symptoms, press Enter when done)")
    chunks, silent_chunks = [], 0
    max_silent = int(SILENCE_DURATION * SAMPLE_RATE / 1024)
    stop_flag = threading.Event()

    def wait_for_enter():
        input()   # blocks until user presses Enter
        stop_flag.set()

    listener = threading.Thread(target=wait_for_enter, daemon=True)
    listener.start()
    
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                        blocksize=1024) as stream:
        while not stop_flag.is_set():
            chunk, _ = stream.read(1024)
            chunks.append(chunk)
            if np.abs(chunk).mean() < THRESHOLD:
                silent_chunks += 1
            else:
                silent_chunks = 0
            if silent_chunks > max_silent and len(chunks) > 10:
                break
    print("\nRecording stopped.")
    audio = np.concatenate(chunks, axis=0)
    buf = io.BytesIO()
    write(buf, SAMPLE_RATE, (audio * 32767).astype(np.int16))
    buf.seek(0)
    return {"_audio_buffer": buf}   # passed to transcription node