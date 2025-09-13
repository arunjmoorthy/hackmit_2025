# app.py
import os
import json
import time
import requests
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from typing import List, Literal, Dict, Any

load_dotenv()  # loads SUNO_API_KEY from .env if present

app = FastAPI()

SUNO_URL = "https://studio-api.prod.suno.com/api/v2/external/hackmit/generate"

class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Song/topic prompt")
    tags: Optional[str] = Field(None, description="Comma-separated tags or provider-specific tags")
    makeInstrumental: Optional[bool] = Field(False, description="True to make instrumental")

class ClipOut(BaseModel):
    id: str
    status: Optional[str] = None
    created_at: Optional[str] = None

class GenerateResponse(BaseModel):
    success: bool
    clips: list[ClipOut]

@app.post("/generate", response_model=GenerateResponse)
def generate_song(req: GenerateRequest):
    # Validate prompt
    if not req.prompt or not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt is required")

    api_key = os.getenv("SUNO_API_KEY")
    if not api_key:
        # mirror your Next.js behavior
        print("SUNO_API_KEY not found in environment variables")
        raise HTTPException(status_code=500, detail="API key not configured")

    # Build request to Suno HackMIT endpoint
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "topic": req.prompt,
        # Only send tags if provided (your Next.js sent `undefined` when absent)
        **({"tags": req.tags} if req.tags else {}),
        "make_instrumental": bool(req.makeInstrumental or False),
    }

    try:
        resp = requests.post(SUNO_URL, headers=headers, data=json.dumps(payload), timeout=60)
    except requests.RequestException as e:
        print("Network error calling Suno:", e)
        raise HTTPException(status_code=502, detail="Upstream network error")

    if not resp.ok:
        # Return the same style of error mapping you had
        error_text = resp.text
        print("Suno API generation error:", error_text)
        raise HTTPException(status_code=resp.status_code, detail="Failed to start song generation")

    # Expect a single clip JSON with id/status/created_at
    try:
        clip = resp.json()
    except ValueError:
        print("Invalid JSON from Suno:", resp.text[:500])
        raise HTTPException(status_code=502, detail="Invalid response from Suno API")

    if not clip or "id" not in clip:
        print("Invalid response format:", clip)
        raise HTTPException(status_code=500, detail="Invalid response from Suno API")

    return GenerateResponse(
        success=True,
        clips=[ClipOut(id=clip["id"], status=clip.get("status"), created_at=clip.get("created_at"))],
    )

STATUS_URL = "https://studio-api.prod.suno.com/api/v2/external/hackmit/clips"

PENDING_STATUSES = {"submitted", "pending", "queued", "processing", "generating", "created"}
DONE_STATUSES = {"done", "completed", "success", "ready", "finished"}
FAILED_STATUSES = {"error", "failed"}

def _fetch_clips_status(clip_ids: List[str], api_key: str) -> List[Dict[str, Any]]:
    """One-shot fetch of statuses for the given IDs."""
    resp = requests.get(
        STATUS_URL,
        params={"ids": ",".join(clip_ids)},
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=30,
    )
    if not resp.ok:
        raise RuntimeError(f"Status check failed: {resp.status_code} {resp.text}")
    data = resp.json()
    # API returns an array of clip objects
    if not isinstance(data, list):
        raise RuntimeError(f"Unexpected status response: {data}")
    return data

def poll_clips(
    clip_ids: List[str],
    *,
    timeout: int = 240,
    interval: int = 5,
    return_when: Literal["any_ready", "all_ready"] = "any_ready",
) -> Dict[str, Any]:
    """
    Poll /clips?ids=... until ready or timeout.
    Returns a dict: {"clips": [...], "ready": [...], "not_ready": [...], "failed": [...]}.
    Each clip in lists is the raw clip JSON from the API (with audio_url/video_url if present).
    """
    api_key = os.getenv("SUNO_API_KEY")
    if not api_key:
        raise RuntimeError("SUNO_API_KEY not configured")

    deadline = time.time() + timeout
    last_batch: List[Dict[str, Any]] = []

    while time.time() < deadline:
        try:
            clips = _fetch_clips_status(clip_ids, api_key)
        except Exception as e:
            # Keep retrying on transient errors
            print("[poll] transient error:", e)
            time.sleep(interval)
            continue

        ready = []
        not_ready = []
        failed = []

        for c in clips:
            status = (c.get("status") or "").lower()
            if status in DONE_STATUSES or c.get("audio_url") or c.get("video_url"):
                ready.append(c)
            elif status in FAILED_STATUSES:
                failed.append(c)
            else:
                not_ready.append(c)

        # Early return conditions
        if return_when == "any_ready" and ready:
            return {"clips": clips, "ready": ready, "not_ready": not_ready, "failed": failed}
        if return_when == "all_ready" and not not_ready and not failed:
            return {"clips": clips, "ready": ready, "not_ready": not_ready, "failed": failed}
        if failed:
            # If any failed, return immediately so caller can surface error
            return {"clips": clips, "ready": ready, "not_ready": not_ready, "failed": failed}

        last_batch = clips
        time.sleep(interval)

    # Timeout: return last seen state
    return {
        "clips": last_batch,
        "ready": [c for c in (last_batch or []) if (c.get("status") or "").lower() in DONE_STATUSES or c.get("audio_url") or c.get("video_url")],
        "not_ready": [c for c in (last_batch or []) if (c.get("status") or "").lower() in PENDING_STATUSES],
        "failed": [c for c in (last_batch or []) if (c.get("status") or "").lower() in FAILED_STATUSES],
        "timeout": True,
    }

if __name__ == "__main__":
    prompt = "a song about the magic of HackMIT"
    tags = "pop, upbeat"
    make_instrumental = True

    try:
        req = GenerateRequest(prompt=prompt, tags=tags, makeInstrumental=make_instrumental)
        resp_model = generate_song(req)

        print("Initial JSON:", json.dumps(resp_model.model_dump(), indent=2))
        clip_id = resp_model.clips[0].id
        print("Clip ID:", clip_id)

        clip_ids = [clip_id]  # replace with yours

        out = poll_clips(clip_ids, timeout=300, interval=6, return_when="any_ready")
        print(json.dumps(out, indent=2))

        # Grab the first ready clip’s URL
        if out.get("ready"):
            c0 = out["ready"][0]
            print("Audio URL:", c0.get("audio_url") or c0.get("video_url"))
        else:
            print("No ready clip yet.", "(Timed out)" if out.get("timeout") else "")

    except Exception as e:
        print("Error:", e)
