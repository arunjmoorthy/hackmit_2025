import modal

stub = modal.Stub("coqui-tts")

# Define the environment (Python version + pip packages)
image = (
    modal.Image.debian_slim(python_version="3.11")  # pin to 3.11 for Coqui
    .pip_install("TTS", "torch", "torchaudio", "torchvision")
)

@stub.function(image=image, gpu="A10G")  # optional GPU
def generate_tts(text: str, speaker_wav: str = None) -> bytes:
    from TTS.api import TTS
    import tempfile

    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")

    out_file = tempfile.mktemp(suffix=".wav")

    tts.tts_to_file(text=text, speaker_wav=speaker_wav, file_path=out_file)

    with open(out_file, "rb") as f:
        return f.read()

# Optional: expose a webhook for browser/JS/frontend
@stub.webhook()
def web_tts(request):
    text = request.query_params.get("text", "Hello from Modal!")
    return generate_tts.remote(text)
