import os
from TTS.api import TTS

MODEL_NAME = os.getenv("TTS_MODEL", "tts_models/en/ljspeech/tacotron2-DDC")

# Load TTS model
tts = TTS(MODEL_NAME)

def synthesize(text, out_path="out.wav"):
    """
    Generate speech using Mozilla/Coqui TTS.
    Works for single-speaker models.
    """
    # For single-speaker models, don't pass speaker
    tts.tts_to_file(text=text, file_path=out_path)
    return out_path