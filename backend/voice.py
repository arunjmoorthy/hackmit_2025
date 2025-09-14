import modal
import tempfile
import os
from TTS.api import TTS

app = modal.App("coqui-tts")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "TTS==0.15.4", 
        "torch==2.2.0", 
        "torchaudio==2.2.0", 
        "torchvision==0.17.0",
        "fastapi[standard]"
    )
)

@app.function(image=image, gpu="A10G", timeout=300)
def generate_simple_tts(text: str) -> bytes:
    """Simple TTS without voice cloning using a more reliable model"""
    print(f"Generating simple TTS for: {text}")
    
    try:
        # Use a simpler, more reliable model first
        tts = TTS("tts_models/en/ljspeech/tacotron2-DDC")
        out_file = tempfile.mktemp(suffix=".wav")
        
        print(f"Using model: tts_models/en/ljspeech/tacotron2-DDC")
        tts.tts_to_file(text=text, file_path=out_file)
        
        with open(out_file, "rb") as f:
            audio_data = f.read()
        
        print(f"Generated audio size: {len(audio_data)} bytes")
        return audio_data
        
    except Exception as e:
        print(f"Error with tacotron2-DDC: {e}")
        # Fallback to an even simpler model
        try:
            print("Trying fallback model...")
            tts = TTS("tts_models/en/ljspeech/glow-tts")
            tts.tts_to_file(text=text, file_path=out_file)
            
            with open(out_file, "rb") as f:
                audio_data = f.read()
            
            print(f"Fallback generated audio size: {len(audio_data)} bytes")
            return audio_data
            
        except Exception as e2:
            print(f"Fallback also failed: {e2}")
            raise Exception(f"Both models failed. Primary: {e}, Fallback: {e2}")
    
    finally:
        if os.path.exists(out_file):
            os.unlink(out_file)

@app.function(image=image, gpu="A10G", timeout=600)
def generate_voice_clone_tts(text: str, speaker_wav: str = None) -> bytes:
    """Voice cloning TTS with the XTTS model"""
    print(f"Generating voice clone TTS for: {text}")
    
    try:
        # Try the full model name path
        tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False)
        out_file = tempfile.mktemp(suffix=".wav")
        
        print(f"Using XTTS v2 model")
        if speaker_wav and os.path.exists(speaker_wav):
            print(f"Using speaker file: {speaker_wav}")
            tts.tts_to_file(text=text, speaker_wav=speaker_wav, file_path=out_file, language="en")
        else:
            print("No speaker file, using default voice")
            tts.tts_to_file(text=text, file_path=out_file, language="en")
        
        with open(out_file, "rb") as f:
            audio_data = f.read()
        
        print(f"Generated audio size: {len(audio_data)} bytes")
        return audio_data
        
    except Exception as e:
        print(f"XTTS v2 failed: {e}")
        # Fall back to simple TTS
        print("Falling back to simple TTS")
        return generate_simple_tts.remote(text)
    
    finally:
        if os.path.exists(out_file):
            os.unlink(out_file)

from fastapi import FastAPI, Form, File, UploadFile, Query
from fastapi.responses import Response

web_app = FastAPI()

@web_app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "TTS service is running", "endpoints": ["/simple", "/clone"]}

@web_app.get("/simple")
async def simple_tts_endpoint(text: str = Query(default="Hello from Modal!")):
    """Simple TTS endpoint using reliable models"""
    try:
        print(f"Simple TTS request: {text}")
        audio_data = generate_simple_tts.remote(text)
        return Response(content=audio_data, media_type="audio/wav")
    except Exception as e:
        print(f"Error in simple TTS endpoint: {e}")
        return {"error": str(e)}

@web_app.post("/clone")
async def voice_clone_endpoint(
    text: str = Form(...),
    speaker_wav: UploadFile = File(None)
):
    """Voice cloning endpoint using XTTS v2"""
    speaker_path = None
    
    try:
        print(f"Voice clone request: {text}")
        
        if speaker_wav:
            print(f"Received speaker file: {speaker_wav.filename}")
            speaker_path = tempfile.mktemp(suffix=".wav")
            with open(speaker_path, "wb") as f:
                content = await speaker_wav.read()
                f.write(content)
            print(f"Saved speaker file: {len(content)} bytes")
        
        audio_data = generate_voice_clone_tts.remote(text, speaker_path)
        return Response(content=audio_data, media_type="audio/wav")
        
    except Exception as e:
        print(f"Error in voice cloning endpoint: {e}")
        return {"error": str(e)}
    finally:
        if speaker_path and os.path.exists(speaker_path):
            os.unlink(speaker_path)

@app.function(image=image, timeout=300)
@modal.asgi_app()
def fastapi_app():
    return web_app