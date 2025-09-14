import argparse
import json
import os
from pathlib import Path
from typing import Optional

import requests
from dotenv import load_dotenv


load_dotenv()


ELEVEN_VOICES_ADD_URL = "https://api.elevenlabs.io/v1/voices/add"
ELEVEN_IVC_CREATE_URL = "https://api.elevenlabs.io/v1/voices/ivc/create"  # fallback if needed
ELEVEN_TTS_URL_TMPL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"


class ElevenLabsError(RuntimeError):
    pass


def _get_api_key(explicit: Optional[str] = None) -> str:
    api_key = explicit or os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        raise ElevenLabsError("ELEVENLABS_API_KEY not set in environment or provided explicitly")
    return api_key


def upload_reference_voice(voice_wav_path: str, name: str, *, api_key: Optional[str] = None) -> str:
    """
    Upload a reference .wav to ElevenLabs Instant Voice Cloning and return the created voice_id.
    """
    key = _get_api_key(api_key)

    p = Path(voice_wav_path)
    if not p.exists() or not p.is_file():
        raise ElevenLabsError(f"Voice WAV not found: {voice_wav_path}")

    with p.open("rb") as f:
        files = {"files": (p.name, f, "audio/wav")}
        data = {"name": name}
        headers = {"xi-api-key": key}

        # Prefer official endpoint; fallback to IVC if method not allowed
        resp = requests.post(ELEVEN_VOICES_ADD_URL, headers=headers, files=files, data=data, timeout=120)
        if resp.status_code in (404, 405):
            # try legacy/alternate path
            resp = requests.post(ELEVEN_IVC_CREATE_URL, headers=headers, files=files, data=data, timeout=120)

    if not resp.ok:
        detail = resp.text
        try:
            j = resp.json()
            detail = json.dumps(j)
        except Exception:
            pass
        raise ElevenLabsError(f"IVC create failed: {resp.status_code} {detail}")

    try:
        payload = resp.json()
    except Exception as e:
        raise ElevenLabsError(f"Invalid JSON from IVC create: {e}; body={resp.text[:500]}")

    voice_id = payload.get("voice_id") or payload.get("voiceId") or payload.get("id")
    if not voice_id:
        raise ElevenLabsError(f"No voice_id in response: {payload}")
    return str(voice_id)


def synthesize_to_file(voice_id: str, text: str, output_path: str, *, api_key: Optional[str] = None) -> str:
    """
    Generate speech for `text` using the provided `voice_id`, saving to `output_path`.
    Returns the path to the written file.
    """
    key = _get_api_key(api_key)
    output_path = str(output_path)

    # Heuristic: choose Accept by extension
    ext = (os.path.splitext(output_path)[1] or "").lower()
    if ext in (".wav", ".wave"):
        accept = "audio/wav"
    else:
        accept = "audio/mpeg"

    url = ELEVEN_TTS_URL_TMPL.format(voice_id=voice_id)
    headers = {
        "xi-api-key": key,
        "Content-Type": "application/json",
        "Accept": accept,
    }
    body = {"text": text}

    resp = requests.post(url, headers=headers, json=body, timeout=300)
    if not resp.ok:
        # Try to surface API error JSON
        try:
            j = resp.json()
            raise ElevenLabsError(f"TTS failed: {resp.status_code} {json.dumps(j)}")
        except ValueError:
            raise ElevenLabsError(f"TTS failed: {resp.status_code} {resp.text}")

    ct = (resp.headers.get("Content-Type") or resp.headers.get("content-type") or "").lower()
    if not ct.startswith("audio/"):
        # Likely an error payload
        try:
            j = resp.json()
            raise ElevenLabsError(f"Unexpected non-audio response: {json.dumps(j)}")
        except ValueError:
            raise ElevenLabsError(f"Unexpected non-audio response with content-type {ct}")

    # Ensure directory exists
    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(output_path, "wb") as f:
        f.write(resp.content)

    return output_path


def _read_text_arg(text: Optional[str], text_file: Optional[str]) -> str:
    if text and text.strip():
        return text
    if text_file:
        p = Path(text_file)
        if not p.exists() or not p.is_file():
            raise ElevenLabsError(f"Transcript file not found: {text_file}")
        ext = p.suffix.lower()
        if ext in (".wav", ".mp3", ".m4a", ".flac", ".ogg"):
            raise ElevenLabsError(
                f"'{text_file}' looks like an audio file. Provide a transcript via --text or a UTF-8 .txt/.srt file."
            )
        # Minimal .srt support: strip indices and timecodes
        if ext == ".srt":
            try:
                raw = p.read_text(encoding="utf-8", errors="ignore")
            except Exception as e:
                raise ElevenLabsError(f"Could not read SRT file: {e}")
            lines = []
            for line in raw.splitlines():
                s = line.strip()
                if not s or s.isdigit() or "-->" in s:
                    continue
                lines.append(s)
            joined = " ".join(lines).strip()
            if not joined:
                raise ElevenLabsError("No subtitle text found in SRT file")
            return joined
        try:
            return p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            raise ElevenLabsError(
                f"Transcript file is not valid UTF-8 text: {text_file}. Use --text or a plain-text .txt/.srt file."
            )
    raise ElevenLabsError("Provide --text or --text-file")


def main():
    parser = argparse.ArgumentParser(description="Generate speech via ElevenLabs using a reference WAV + transcript.")
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--voice-id", type=str, help="Existing ElevenLabs voice_id to use (skip upload)")
    group.add_argument("--voice-wav", type=str, help="Path to reference .wav to upload as a new voice")

    parser.add_argument("--name", type=str, default="Custom Voice", help="Name for the uploaded voice (when using --voice-wav)")
    parser.add_argument("--text", type=str, help="Transcript text to synthesize")
    parser.add_argument("--text-file", type=str, help="Path to a .txt file with transcript")
    parser.add_argument("-o", "--out", type=str, default="output.wav", help="Output audio path (.wav or .mp3)")
    parser.add_argument("--api-key", type=str, default=None, help="Override ELEVENLABS_API_KEY")

    args = parser.parse_args()

    api_key = args.api_key or os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        raise ElevenLabsError("ELEVENLABS_API_KEY not configured. Add it to your .env or pass --api-key.")

    text = _read_text_arg(args.text, args.text_file)

    voice_id = args.voice_id
    if not voice_id:
        if not args.voice_wav:
            raise ElevenLabsError("Either --voice-id or --voice-wav must be provided")
        voice_id = upload_reference_voice(args.voice_wav, args.name, api_key=api_key)
        print(f"Created ElevenLabs voice_id: {voice_id}")

    out_path = synthesize_to_file(voice_id, text, args.out, api_key=api_key)
    print(f"✅ Saved: {out_path}")


if __name__ == "__main__":
    main()


