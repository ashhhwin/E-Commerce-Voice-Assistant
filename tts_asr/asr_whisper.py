import whisper

def transcribe(audio_path, model_name="small"):
    # Requires ffmpeg installed on system
    model = whisper.load_model(model_name)
    res = model.transcribe(audio_path)
    return res["text"]
