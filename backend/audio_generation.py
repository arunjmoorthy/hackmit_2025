import os
import json
import time
import requests
from typing import Optional, List, Literal, Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()  # loads SUNO_API_KEY from .env if present

app = FastAPI()

SUNO_URL = "https://studio-api.prod.suno.com/api/v2/external/hackmit/generate"
STATUS_URL = "https://studio-api.prod.suno.com/api/v2/external/hackmit/clips"

# ---------------- Models ----------------

class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Song/topic prompt")
    tags: Optional[str] = Field(None, description="Comma-separated tags or provider-specific tags")
    makeInstrumental: Optional[bool] = Field(False, description="True to make instrumental")
    duration: Optional[int] = Field(60, description="Length in seconds (30, 60, 90, 120)")

class ClipOut(BaseModel):
    id: str
    status: Optional[str] = None
    created_at: Optional[str] = None
    audio_url: Optional[str] = None
    video_url: Optional[str] = None

class GenerateResponse(BaseModel):
    success: bool
    clips: list[ClipOut]

# ---------------- Generate ----------------

@app.post("/generate", response_model=GenerateResponse)
def generate_song(req: GenerateRequest):
    if not req.prompt or not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt is required")

    api_key = os.getenv("SUNO_API_KEY")
    if not api_key:
        print("SUNO_API_KEY not found in environment variables")
        raise HTTPException(status_code=500, detail="API key not configured")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "topic": req.prompt,
        **({"tags": req.tags} if req.tags else {}),
        "make_instrumental": bool(req.makeInstrumental or False),
        "duration": req.duration or 60,  # 👈 new field
    }

    try:
        resp = requests.post(SUNO_URL, headers=headers, data=json.dumps(payload), timeout=60)
    except requests.RequestException as e:
        print("Network error calling Suno:", e)
        raise HTTPException(status_code=502, detail="Upstream network error")

    if not resp.ok:
        error_text = resp.text
        print("Suno API generation error:", error_text)
        raise HTTPException(status_code=resp.status_code, detail="Failed to start song generation")

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
        clips=[ClipOut(
            id=clip["id"],
            status=clip.get("status"),
            created_at=clip.get("created_at")
        )],
    )

# ---------------- Poll ----------------

PENDING_STATUSES = {"submitted", "pending", "queued", "processing", "generating", "created"}
DONE_STATUSES = {"done", "completed", "success", "ready", "finished"}
FAILED_STATUSES = {"error", "failed"}

def _fetch_clips_status(clip_ids: List[str], api_key: str) -> List[Dict[str, Any]]:
    resp = requests.get(
        STATUS_URL,
        params={"ids": ",".join(clip_ids)},
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=30,
    )
    if not resp.ok:
        raise RuntimeError(f"Status check failed: {resp.status_code} {resp.text}")
    data = resp.json()
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
    api_key = os.getenv("SUNO_API_KEY")
    if not api_key:
        raise RuntimeError("SUNO_API_KEY not configured")

    deadline = time.time() + timeout
    last_batch: List[Dict[str, Any]] = []

    while time.time() < deadline:
        try:
            clips = _fetch_clips_status(clip_ids, api_key)
        except Exception as e:
            print("[poll] transient error:", e)
            time.sleep(interval)
            continue

        ready, not_ready, failed = [], [], []
        for c in clips:
            status = (c.get("status") or "").lower()
            if status in DONE_STATUSES or c.get("audio_url") or c.get("video_url"):
                ready.append(c)
            elif status in FAILED_STATUSES:
                failed.append(c)
            else:
                not_ready.append(c)

        if return_when == "any_ready" and ready:
            return {"clips": clips, "ready": ready, "not_ready": not_ready, "failed": failed}
        if return_when == "all_ready" and not not_ready and not failed:
            return {"clips": clips, "ready": ready, "not_ready": not_ready, "failed": failed}
        if failed:
            return {"clips": clips, "ready": ready, "not_ready": not_ready, "failed": failed}

        last_batch = clips
        time.sleep(interval)

    return {
        "clips": last_batch,
        "ready": [c for c in (last_batch or []) if (c.get("status") or "").lower() in DONE_STATUSES or c.get("audio_url") or c.get("video_url")],
        "not_ready": [c for c in (last_batch or []) if (c.get("status") or "").lower() in PENDING_STATUSES],
        "failed": [c for c in (last_batch or []) if (c.get("status") or "").lower() in FAILED_STATUSES],
        "timeout": True,
    }

# ---------------- Example usage ----------------

if __name__ == "__main__":
    prompt = "Generate a Github Demo Login background music where I'm demoing how to login to Github"
    tags = "pop, upbeat"
    make_instrumental = True
    duration = 90  # 👈 control length here

    try:
        req = GenerateRequest(prompt=prompt, tags=tags, makeInstrumental=make_instrumental, duration=duration)
        resp_model = generate_song(req)

        print("Initial JSON:", json.dumps(resp_model.model_dump(), indent=2))
        clip_id = resp_model.clips[0].id
        print("Clip ID:", clip_id)

        out = poll_clips([clip_id], timeout=300, interval=6, return_when="any_ready")
        print(json.dumps(out, indent=2))

        if out.get("ready"):
            c0 = out["ready"][0]
            print("Audio URL:", c0.get("audio_url") or c0.get("video_url"))
        else:
            print("No ready clip yet.", "(Timed out)" if out.get("timeout") else "")

    except Exception as e:
        print("Error:", e)
