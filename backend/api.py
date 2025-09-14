import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncGenerator, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.responses import StreamingResponse
from starlette.staticfiles import StaticFiles

from main import create_demo_video_with_timestamps, _build_from_history_obj
from trim_from_history import jumpcut_video
from fit_transcript import (
    segments_from_history_json,
    fit_transcript_to_time,
)
from elevenlabs_tts import upload_reference_voice, synthesize_to_file
from pydub import AudioSegment
import subprocess
import shutil
import os
import audio_generation as suno


app = FastAPI(title="HackMIT Demo API", version="0.1.0")

# CORS for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories
ARTIFACTS_DIR = Path("artifacts")
VIDEOS_DIR = ARTIFACTS_DIR / "videos"
LOGS_DIR = ARTIFACTS_DIR / "logs"
AUDIO_DIR = ARTIFACTS_DIR / "audio"
for d in (VIDEOS_DIR, LOGS_DIR, AUDIO_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Serve artifacts statically for convenience
app.mount("/videos", StaticFiles(directory=str(VIDEOS_DIR), html=False), name="videos")
app.mount("/logs", StaticFiles(directory=str(LOGS_DIR), html=False), name="logs")
app.mount("/audio", StaticFiles(directory=str(AUDIO_DIR), html=False), name="audio")


@app.get("/health")
async def health():
    return {"ok": True, "time": datetime.now(timezone.utc).isoformat()}


def _sse_event(event: str, data: Dict[str, Any]) -> str:
    payload = json.dumps({"event": event, "data": data}, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


@app.post("/api/process")
async def process(
    url: str = Form(...),
    description: str = Form(""),
    audio: UploadFile = File(...),
    use_uploaded_voice: bool = Form(False),
):
    # Save uploaded audio first, outside the stream
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    audio_name = f"user_{ts}_{audio.filename or 'audio.wav'}"
    audio_path = AUDIO_DIR / audio_name
    try:
        content = await audio.read()
        with open(audio_path, "wb") as f:
            f.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed saving audio: {e}")

    async def event_stream() -> AsyncGenerator[bytes, None]:
        yield _sse_event("audio_saved", {"path": f"/audio/{audio_name}"}).encode()

        # 1) Start browser agent and create full video + history
        yield _sse_event("status", {"message": "Creating the visuals via browser agent..."}).encode()
        try:
            result = await create_demo_video_with_timestamps(url, description)
        except Exception as e:
            yield _sse_event("error", {"message": f"Agent failed: {e}"}).encode()
            return

        history_path = result.get("history_path")
        video_dir_hint = result.get("video_dir")
        if not history_path:
            yield _sse_event("error", {"message": "No history produced by agent."}).encode()
            return

        # Find newest .mp4 in videos directory as the agent recording (best effort)
        full_video_path = None
        try:
            # Check multiple possible locations and patterns
            possible_patterns = [
                VIDEOS_DIR.glob("*.mp4"),
                VIDEOS_DIR.glob("**/*.mp4"),  # recursive search
                Path("artifacts/videos").glob("*.mp4"),
                Path(".").glob("artifacts/videos/*.mp4"),
            ]
            
            all_videos = []
            for pattern in possible_patterns:
                try:
                    all_videos.extend(list(pattern))
                except:
                    continue
            
            if all_videos:
                # Get the newest video file
                full_video_path = max(all_videos, key=lambda p: p.stat().st_mtime)
                yield _sse_event("status", {"message": f"Found video: {full_video_path.name}"}).encode()
            else:
                yield _sse_event("status", {"message": "No video files found, checking video_dir_hint..."}).encode()
                
                # Try to find video in the hinted directory
                if video_dir_hint:
                    hint_path = Path(video_dir_hint)
                    if hint_path.exists():
                        video_files = list(hint_path.glob("*.mp4"))
                        if video_files:
                            full_video_path = max(video_files, key=lambda p: p.stat().st_mtime)
                            yield _sse_event("status", {"message": f"Found video in hint dir: {full_video_path.name}"}).encode()
        except Exception as e:
            yield _sse_event("status", {"message": f"Video search error: {e}"}).encode()

        yield _sse_event("agent_done", {"history": f"/logs/{Path(history_path).name}", "video_dir": video_dir_hint, "full_video": f"/videos/{full_video_path.name}" if full_video_path else None}).encode()

        # 2) Trim video based on stillness and remap history
        yield _sse_event("status", {"message": "Trimming the video..."}).encode()
        trimmed_name = f"demo_trimmed_{ts}.mp4"
        trimmed_path = VIDEOS_DIR / trimmed_name
        warp_json = LOGS_DIR / f"timewarp_{ts}.json"
        remapped_history_json = LOGS_DIR / f"agent_history_trimmed_{ts}.json"
        try:
            if not full_video_path or not full_video_path.exists():
                # Create a dummy video as fallback
                yield _sse_event("status", {"message": "No video found, creating placeholder..."}).encode()
                # Copy an existing video or create a minimal one
                existing_videos = list(VIDEOS_DIR.glob("*.mp4"))
                if existing_videos:
                    import shutil
                    shutil.copy2(existing_videos[0], trimmed_path)
                    yield _sse_event("status", {"message": "Used existing video as placeholder"}).encode()
                else:
                    raise RuntimeError("No video files available for processing")
            else:
                jumpcut_video(
                    input_path=str(full_video_path),
                    output_path=str(trimmed_path),
                    mode="cut",
                    still_min_seconds=2.5,
                    frame_step=3,
                    diff_threshold=1.2,
                    warp_out_path=str(warp_json),
                    history_json_path=str(history_path),
                    remapped_history_path=str(remapped_history_json),
                )
        except Exception as e:
            yield _sse_event("error", {"message": f"Trimming failed: {e}"}).encode()
            return

        yield _sse_event("trim_done", {"trimmed_video": f"/videos/{trimmed_name}", "warp": f"/logs/{warp_json.name}", "remapped_history": f"/logs/{remapped_history_json.name}"}).encode()

        # 3) Build transcript segments (fitted to time)
        yield _sse_event("status", {"message": "Building transcript..."}).encode()
        try:
            transcript_source = remapped_history_json if remapped_history_json.exists() else Path(history_path)
            raw_segments = segments_from_history_json(str(transcript_source))
            if not raw_segments:
                raise RuntimeError("No segments found from history")
            fitted_segments = fit_transcript_to_time(
                raw_segments,
                wpm=155,
                safety=0.88,
                min_words_per_segment=3,
                max_words_per_segment=40,
                model="claude-sonnet-4-20250514",
                style_hint="product demo voice-over; concise, confident, friendly",
            )
            # Write SRT
            def _to_srt_timestamp(seconds: float) -> str:
                if seconds < 0:
                    seconds = 0
                hours = int(seconds // 3600)
                minutes = int((seconds % 3600) // 60)
                secs = int(seconds % 60)
                millis = int(round((seconds - int(seconds)) * 1000))
                return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"
            srt_lines = []
            for idx, s in enumerate(fitted_segments, start=1):
                t0 = s.start.total_seconds(); t1 = s.end.total_seconds()
                text = (s.text or " ").strip() or " "
                srt_lines.append(str(idx))
                srt_lines.append(f"{_to_srt_timestamp(t0)} --> {_to_srt_timestamp(t1)}")
                if len(text) > 54:
                    mid = len(text)//2
                    cut = text.rfind(" ", 0, mid)
                    if cut == -1:
                        cut = mid
                    text = text[:cut].strip() + "\n" + text[cut:].strip()
                srt_lines.append(text)
                srt_lines.append("")
            captions_name = f"captions_{ts}.srt"
            captions_path = LOGS_DIR / captions_name
            captions_path.write_text("\n".join(srt_lines), encoding="utf-8")
        except Exception as e:
            yield _sse_event("error", {"message": f"Transcript failed: {e}"}).encode()
            return

        yield _sse_event("transcript_done", {"captions": f"/logs/{captions_name}"}).encode()

        # 4) Choose voice (upload or base)
        BASE_VOICE_ID = "9llyPeVLVUPas3NvBogz"
        voice_id = BASE_VOICE_ID
        if use_uploaded_voice:
            yield _sse_event("status", {"message": "Uploading reference voice..."}).encode()
            try:
                voice_id = upload_reference_voice(str(audio_path), name=f"UserVoice {ts}")
                yield _sse_event("status", {"message": f"Voice created: {voice_id}"}).encode()
            except Exception as e:
                yield _sse_event("error", {"message": f"Voice upload failed, using base voice: {e}"}).encode()
                voice_id = BASE_VOICE_ID

        # 5) Synthesize TTS per segment and build mixed audio
        yield _sse_event("status", {"message": "Synthesizing voiceover..."}).encode()
        try:
            def _run_ff(cmd: list[str]) -> None:
                p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                if p.returncode != 0:
                    raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(cmd)}\n{p.stdout}")
            def _ffprobe_duration(path: str) -> float:
                cmd = ["ffprobe","-v","error","-show_entries","format=duration","-of","default=noprint_wrappers=1:nokey=1",path]
                p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                if p.returncode != 0:
                    raise RuntimeError(f"ffprobe failed for {path}: {p.stdout}")
                return float(p.stdout.strip())
            def _has_audio_stream(path: str) -> bool:
                cmd = ["ffprobe","-v","error","-select_streams","a","-show_entries","stream=codec_type","-of","default=nk=1:nw=1",path]
                p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                if p.returncode != 0:
                    return False
                return "audio" in (p.stdout or "").strip().lower()
            def _extract_audio(input_video: str, out_wav: str, sr: int = 44100) -> None:
                _run_ff(["ffmpeg","-y","-i",input_video,"-vn","-ac","2","-ar",str(sr),out_wav])
            def _ffmpeg_subtitles_filter_arg(srt_path: str) -> str:
                pth = os.path.abspath(srt_path).replace("\\","\\\\").replace(":","\\:")
                return f"subtitles='{pth}'"
            def _pad_or_trim_to_duration(src_wav: str, dst_wav: str, target_seconds: float) -> None:
                if target_seconds <= 0:
                    shutil.copyfile(src_wav, dst_wav); return
                clip = AudioSegment.from_file(src_wav).set_channels(2)
                target_ms = int(target_seconds * 1000); cur_ms = len(clip)
                if cur_ms < target_ms:
                    pad = AudioSegment.silent(duration=target_ms - cur_ms, frame_rate=clip.frame_rate)
                    out = clip + pad
                elif cur_ms > target_ms:
                    fade = 60 if target_ms > 120 else max(0, target_ms // 2)
                    out = clip[:target_ms].fade_out(fade)
                else:
                    out = clip
                out.export(dst_wav, format="wav")
            def _build_voiceover_track(segments, tts_wavs: list[str], total_seconds: float, sample_rate: int = 44100) -> AudioSegment:
                base = AudioSegment.silent(duration=int(total_seconds * 1000), frame_rate=sample_rate)
                for seg, wav in zip(segments, tts_wavs):
                    if not os.path.exists(wav):
                        continue
                    vo = AudioSegment.from_file(wav).set_frame_rate(sample_rate).set_channels(2)
                    start_ms = int(seg.start.total_seconds() * 1000)
                    base = base.overlay(vo, position=start_ms)
                return base
            def _duck_background(bg: AudioSegment, segments, duck_db: float = 10.0) -> AudioSegment:
                outbg = bg
                for seg in segments:
                    start_ms = int(seg.start.total_seconds() * 1000)
                    end_ms = int(seg.end.total_seconds() * 1000)
                    if end_ms <= start_ms:
                        continue
                    before = outbg[:start_ms]
                    middle = outbg[start_ms:end_ms] - duck_db
                    after = outbg[end_ms:]
                    outbg = before + middle + after
                return outbg
            def _mux_audio_and_burn_subs(input_video: str, audio_wav: str, srt_path: str, out_video: str) -> None:
                tmp_vid = str(VIDEOS_DIR / f"tmp_{ts}.mp4")
                _run_ff(["ffmpeg","-y","-i",input_video,"-i",audio_wav,"-map","0:v:0","-map","1:a:0","-c:v","copy","-c:a","aac","-shortest",tmp_vid])
                sub_filter = _ffmpeg_subtitles_filter_arg(srt_path)
                _run_ff(["ffmpeg","-y","-i",tmp_vid,"-vf",sub_filter,"-c:a","copy",out_video])
                os.remove(tmp_vid)

            # Synthesize all segments
            seg_dir = AUDIO_DIR / f"segments_{ts}"
            seg_dir.mkdir(parents=True, exist_ok=True)
            tts_wavs_fit = []
            for idx, seg in enumerate(fitted_segments):
                target = max(0.05, (seg.end - seg.start).total_seconds())
                seg_wav = seg_dir / f"seg_{idx:04}.wav"
                text = (seg.text or "").strip()
                if not text:
                    AudioSegment.silent(duration=1).export(str(seg_wav), format="wav")
                else:
                    synthesize_to_file(voice_id, text, str(seg_wav))
                seg_fit_wav = seg_dir / f"seg_{idx:04}_fit.wav"
                _pad_or_trim_to_duration(str(seg_wav), str(seg_fit_wav), target_seconds=target)
                tts_wavs_fit.append(str(seg_fit_wav))

            # Build VO track and background
            video_dur = _ffprobe_duration(str(trimmed_path))
            vo_full = _build_voiceover_track(fitted_segments, tts_wavs_fit, total_seconds=video_dur, sample_rate=44100)

            # Layer 1: Original video audio (if any) - ducked during voiceover
            orig_wav = AUDIO_DIR / f"orig_{ts}.wav"
            if _has_audio_stream(str(trimmed_path)):
                _extract_audio(str(trimmed_path), str(orig_wav), sr=44100)
                bg = AudioSegment.from_file(str(orig_wav)).set_frame_rate(44100).set_channels(2)
                # Duck the original audio during voiceover segments to avoid conflicts
                bg_ducked = _duck_background(bg, fitted_segments, duck_db=15.0)  # More aggressive ducking
            else:
                bg_ducked = AudioSegment.silent(duration=int(video_dur * 1000), frame_rate=44100).set_channels(2)

            # Layer 2: Suno background music (quiet)
            mixed = bg_ducked
            try:
                bg_prompt = "ambient, minimal, modern tech demo background music"
                bg_music_path = suno.generate_background_music(bg_prompt, target_duration_seconds=int(max(1, video_dur)))
                bgm = AudioSegment.from_file(str(bg_music_path)).set_frame_rate(44100).set_channels(2)
                target_ms = int(video_dur * 1000)
                if len(bgm) < target_ms:
                    times = target_ms // max(1, len(bgm)) + 1
                    bgm = (bgm * times)[:target_ms]
                elif len(bgm) > target_ms:
                    bgm = bgm[:target_ms]
                bgm = bgm - 18  # Make background music quieter
                mixed = mixed.overlay(bgm)
            except Exception as e:
                yield _sse_event("status", {"message": f"Background music skipped: {e}"}).encode()

            # Layer 3: ElevenLabs voiceover (primary audio)
            mixed = mixed.overlay(vo_full)
            mixed_wav_name = f"mixed_{ts}.wav"
            mixed_wav_path = AUDIO_DIR / mixed_wav_name
            mixed.export(str(mixed_wav_path), format="wav")

            final_name = f"demo_voiceover_{ts}.mp4"
            final_path = VIDEOS_DIR / final_name
            _mux_audio_and_burn_subs(str(trimmed_path), str(mixed_wav_path), str(captions_path), str(final_path))
        except Exception as e:
            yield _sse_event("error", {"message": f"Audio/Video assembly failed: {e}"}).encode()
            return

        # 6) Finalize
        yield _sse_event("status", {"message": "Finalizing your video..."}).encode()
        yield _sse_event("complete", {
            "trimmed_video": f"/videos/{trimmed_name}",
            "history": f"/logs/{Path(history_path).name}",
            "remapped_history": f"/logs/{remapped_history_json.name}",
            "captions": f"/logs/{captions_name}",
            "mixed_audio": f"/audio/{mixed_wav_name}",
            "final_video": f"/videos/{final_name}",
            "voice_id": voice_id,
        }).encode()

    headers = {"Content-Type": "text/event-stream", "Cache-Control": "no-cache", "Connection": "keep-alive"}
    return StreamingResponse(event_stream(), headers=headers)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), reload=True)


