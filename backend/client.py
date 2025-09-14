import requests
import os
import json

# Updated URLs for the FastAPI app approach
BASE_URL = "https://hack2025--coqui-tts-fastapi-app.modal.run"
SIMPLE_URL = f"{BASE_URL}/simple"
CLONE_URL = f"{BASE_URL}/clone"

def check_response(response, filename):
    """Check if response is valid audio or an error"""
    print(f"Status: {response.status_code}")
    print(f"Content-Type: {response.headers.get('content-type')}")
    print(f"Size: {len(response.content)} bytes")
    
    if response.status_code != 200:
        print(f"❌ Error: {response.text}")
        return False
    
    # Check if it's JSON (error) or binary (audio)
    try:
        error_data = json.loads(response.content)
        print(f"❌ Server Error: {error_data}")
        return False
    except:
        # Not JSON, should be binary audio
        pass
    
    # Save and check file
    with open(filename, "wb") as f:
        f.write(response.content)
    
    # Verify it's a WAV file
    if len(response.content) < 44:  # WAV header is at least 44 bytes
        print("❌ File too small to be valid audio")
        return False
    
    with open(filename, "rb") as f:
        header = f.read(12)
        if header.startswith(b'RIFF') and b'WAVE' in header:
            print(f"✅ Valid WAV file saved as {filename}")
            return True
        else:
            print(f"❌ Not a valid WAV file. First bytes: {header}")
            return False

def test_simple_tts():
    """Test simple TTS"""
    print("\n🧪 Testing Simple TTS...")
    
    text = "Hello, this is a test of the simple TTS system."
    params = {"text": text}
    
    print(f"URL: {SIMPLE_URL}")
    print(f"Text: {text}")
    
    try:
        response = requests.get(SIMPLE_URL, params=params, timeout=120)
        return check_response(response, "test_simple.wav")
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return False

def test_voice_clone():
    """Test voice cloning"""
    print("\n🧪 Testing Voice Cloning...")
    
    speaker_file = "speaker_reference.wav"
    
    if not os.path.exists(speaker_file):
        print(f"⚠️  No speaker file found: {speaker_file}")
        print("Creating a dummy test without speaker file...")
        speaker_file = None
    else:
        print(f"Speaker file: {speaker_file} ({os.path.getsize(speaker_file)} bytes)")
    
    text = "This is a test of the voice cloning system."
    
    print(f"URL: {CLONE_URL}")
    print(f"Text: {text}")
    
    try:
        data = {"text": text}
        files = {}
        
        if speaker_file:
            with open(speaker_file, "rb") as f:
                files["speaker_wav"] = ("speaker.wav", f, "audio/wav")
                response = requests.post(CLONE_URL, data=data, files=files, timeout=300)
        else:
            # Test without speaker file
            response = requests.post(CLONE_URL, data=data, timeout=300)
            
        return check_response(response, "test_cloned.wav")
        
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return False

def main():
    print("🔧 Modal TTS Fixed Test")
    print("=" * 40)
    
    # Test simple TTS first
    if test_simple_tts():
        print("🎉 Simple TTS works!")
    else:
        print("💥 Simple TTS failed")
    
    # Test voice cloning
    if test_voice_clone():
        print("🎉 Voice cloning works!")
    else:
        print("💥 Voice cloning failed")

if __name__ == "__main__":
    main()