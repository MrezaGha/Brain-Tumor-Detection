# tests/test_audio.py
import os
import io
import threading
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

SAMPLE_RATE = 16000
THRESHOLD_MULTIPLIER = 3.0   # voice must be this many times louder than background
CALIBRATION_SECONDS = 1.5    # listens to room noise first to set baseline

def calibrate_noise_floor():
    """Listen to background noise for 1.5 seconds to set baseline threshold."""
    print("Calibrating background noise — stay quiet for 1.5 seconds...")
    frames = []
    calibration_samples = int(CALIBRATION_SECONDS * SAMPLE_RATE)

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                        dtype="float32", blocksize=1024) as stream:
        collected = 0
        while collected < calibration_samples:
            chunk, _ = stream.read(1024)
            frames.append(chunk)
            collected += len(chunk)

    noise = np.concatenate(frames, axis=0)
    baseline = np.abs(noise).mean()
    threshold = baseline * THRESHOLD_MULTIPLIER
    print(f"Noise floor set: {baseline:.5f} | Voice threshold: {threshold:.5f}")
    return threshold


def record_with_manual_stop(threshold):
    """Record audio, filtering out background noise. Press Enter to stop."""
    print("\nRecording... Press ENTER to stop.\n")
    chunks = []
    stop_flag = threading.Event()

    def wait_for_enter():
        input()   # blocks until user presses Enter
        stop_flag.set()

    # Start Enter-listener in background thread
    listener = threading.Thread(target=wait_for_enter, daemon=True)
    listener.start()

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                        dtype="float32", blocksize=1024) as stream:
        while not stop_flag.is_set():
            chunk, _ = stream.read(1024)
            amplitude = np.abs(chunk).mean()

            # Only keep chunks above noise threshold (your voice)
            if amplitude > threshold:
                chunks.append(chunk)

    if not chunks:
        raise RuntimeError("No voice detected above background noise. "
                           "Try speaking louder or recalibrating.")

    print("\nRecording stopped.")
    audio = np.concatenate(chunks, axis=0)
    buf = io.BytesIO()
    write(buf, SAMPLE_RATE, (audio * 32767).astype(np.int16))
    buf.seek(0)
    return buf


def test_audio_transcription():
    threshold = calibrate_noise_floor()
    audio_buffer = record_with_manual_stop(threshold)

    client = OpenAI(api_key=os.getenv("OPENAI_KEY"))
    transcription = client.audio.transcriptions.create(
        model="whisper-1",
        file=("audio.wav", audio_buffer, "audio/wav"),
        prompt="Patient describing neurological symptoms"
    )

    print(f"\nTranscript: {transcription.text}")

    assert transcription.text is not None
    assert len(transcription.text.strip()) > 0