# Sign in with email 'hacker923121@gmail.com' and password 'ArunMoorthy123'. Make a new repo with a random name. Then respond 'done' and stop.
import asyncio
import json
import os
import subprocess
import shutil
from pathlib import Path
from datetime import datetime, timezone

from main import create_demo_video_with_timestamps, _build_from_history_obj
from screen_record import ScreenRecorder
# from trim_from_history import main as trim_video
from trim_from_history import jumpcut_video as trim_video
from dotenv import load_dotenv

# Transcript fitter and AV helpers
from fit_transcript import (
    segments_from_history_json,
    fit_transcript_to_time,
)
from elevenlabs_tts import synthesize_to_file
import audio_generation as suno
from pydub import AudioSegment

URL = "github.com"
DESCRIPTION = (
    "Sign in with email 'hacker923121@gmail.com' and password 'ArunMoorthy123'. "
    "Make a new repo with a random name. "
    # "Then, go to collaborators and invite a collaborator with the github username 'arunjmoorthy'. "
    "Then respond 'done' and stop."
)

# ElevenLabs voice to use
ELEVEN_VOICE_ID = "9llyPeVLVUPas3NvBogz"

# Style hint passed to transcript fitter
TRANSCRIPT_STYLE = "product demo voice-over; concise, confident, friendly"


def _to_srt_timestamp(seconds: float) -> str:
    if seconds < 0:
        seconds = 0
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"


def _write_srt(segments, path: str) -> None:
    lines = []
    for idx, s in enumerate(segments, start=1):
        t0 = s.start.total_seconds()
        t1 = s.end.total_seconds()
        text = (s.text or " ").strip() or " "
        lines.append(str(idx))
        lines.append(f"{_to_srt_timestamp(t0)} --> {_to_srt_timestamp(t1)}")
        if len(text) > 54:
            mid = len(text)//2
            cut = text.rfind(" ", 0, mid)
            if cut == -1:
                cut = mid
            text = text[:cut].strip() + "\n" + text[cut:].strip()
        lines.append(text)
        lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _run_ff(cmd: list[str]) -> None:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(cmd)}\n{p.stdout}")


def _ffprobe_duration(path: str) -> float:
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", path
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path}: {p.stdout}")
    try:
        return float(p.stdout.strip())
    except Exception as e:
        raise RuntimeError(f"Could not parse duration from ffprobe: {p.stdout}") from e


def _has_audio_stream(path: str) -> bool:
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "a",
        "-show_entries", "stream=codec_type",
        "-of", "default=nk=1:nw=1",
        path
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        return False
    out = (p.stdout or "").strip().lower()
    return "audio" in out


def _pad_or_trim_to_duration(src_wav: str, dst_wav: str, target_seconds: float) -> None:
    if target_seconds <= 0:
        shutil.copyfile(src_wav, dst_wav)
        return
    clip = AudioSegment.from_file(src_wav).set_channels(2)
    target_ms = int(target_seconds * 1000)
    cur_ms = len(clip)
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


def _extract_audio(input_video: str, out_wav: str, sr: int = 44100) -> None:
    cmd = ["ffmpeg", "-y", "-i", input_video, "-vn", "-ac", "2", "-ar", str(sr), out_wav]
    _run_ff(cmd)


def _ffmpeg_subtitles_filter_arg(srt_path: str) -> str:
    p = os.path.abspath(srt_path)
    p = p.replace("\\", "\\\\").replace(":", "\\:")
    return f"subtitles='{p}'"


def _duck_background(bg: AudioSegment, segments, duck_db: float = 10.0) -> AudioSegment:
    out = bg
    for seg in segments:
        start_ms = int(seg.start.total_seconds() * 1000)
        end_ms = int(seg.end.total_seconds() * 1000)
        if end_ms <= start_ms:
            continue
        before = out[:start_ms]
        middle = out[start_ms:end_ms] - duck_db
        after = out[end_ms:]
        out = before + middle + after
    return out


def _mux_audio_and_burn_subs(input_video: str, audio_wav: str, srt_path: str, out_video: str) -> None:
    tmp_vid = out_video + ".tmp_nosubs.mp4"
    _run_ff([
        "ffmpeg", "-y",
        "-i", input_video, "-i", audio_wav,
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy", "-c:a", "aac", "-shortest", tmp_vid
    ])
    sub_filter = _ffmpeg_subtitles_filter_arg(srt_path)
    _run_ff([
        "ffmpeg", "-y",
        "-i", tmp_vid,
        "-vf", sub_filter,
        "-c:a", "copy",
        out_video
    ])
    os.remove(tmp_vid)


async def main() -> None:
    load_dotenv()
    # 1) Start full-screen recorder (saved under artifacts/videos) with unique filenames
    videos_dir = Path("artifacts/videos")
    videos_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = Path("artifacts/audio")
    audio_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = Path("artifacts/logs")
    logs_dir.mkdir(parents=True, exist_ok=True)

    def _next_available_path(base: Path) -> Path:
        if not base.exists():
            return base
        stem = base.stem
        suffix = base.suffix
        n = 1
        while True:
            candidate = base.with_name(f"{stem} ({n}){suffix}")
            if not candidate.exists():
                return candidate
            n += 1

    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    full_video = _next_available_path(videos_dir / f"demo_full.mp4")
    # Force mac screen capture with auto screen selection; disable audio to avoid device conflicts
    # Force screen capture device selection in code (macOS):
    # Probe devices and pick the best match for the screen; otherwise fallback to index 1
    tmp_recorder = ScreenRecorder(out_path=str(full_video), fps=30, display="auto", audio=None, platform="auto")
    try:
        vids, _ = tmp_recorder._list_avfoundation_devices()
        # print(f"[DEBUG] Available AVFoundation video devices: {vids}")
        # Prefer exact 'Capture screen 0', then names starting with 'Capture screen', then any containing 'screen'
        screen_idx = None
        for i, n in vids:
            if n.strip().startswith("Capture screen 0"):
                screen_idx = i
                print(f"Selected screen capture device: '{n}' at index {i}")
                break
        if screen_idx is None:
            for i, n in vids:
                if n.strip().lower().startswith("capture screen"):
                    screen_idx = i
                    print(f"Selected screen capture device: '{n}' at index {i}")
                    break
        if screen_idx is None:
            for i, n in vids:
                if "screen" in n.lower():
                    screen_idx = i
                    print(f"Selected screen-like device: '{n}' at index {i}")
                    break
        if screen_idx is None:
            screen_idx = 1
            print(f"No screen device found, falling back to index {screen_idx}")
        display_sel = screen_idx
    except Exception as e:
        print(f"Exception during device probing: {e}, using 'auto'")
        display_sel = "auto"

    print(f"Starting screen recording with device index: {display_sel}")
    recorder = ScreenRecorder(out_path=str(full_video), fps=30, display=display_sel, audio=None, platform="auto")
    recorder.start()

    try:
        # 2) Run the agent and save agent history JSON
        result = await create_demo_video_with_timestamps(URL, DESCRIPTION)
    finally:
        # 3) Stop recorder regardless of agent outcome
        recorder.stop()

    history_path = result.get("history_path")
    if not history_path:
        print("No history_path returned; cannot generate transcript.")
        return

    history_file = Path(history_path)
    if not history_file.exists():
        print(f"Agent history file not found at {history_file}; cannot generate transcript.")
        return

    # 4) Summarize from history (optional pretty print)
    with open(history_file, "r", encoding="utf-8") as f:
        history_json = json.load(f)
    built = _build_from_history_obj(history_json)

    print("\n=== Step Ranges (derived from agent history durations) ===")
    step_ranges = built.get("step_ranges", [])
    if step_ranges:
        for s in step_ranges:
            print(
                f"Step {s.get('step_index')}: "
                f"{s.get('start_s')}s → {s.get('end_s')}s"
                f" (Δ {s.get('duration_s')}s) — {s.get('summary')}"
            )
    else:
        print("No steps found in history.")

    print("\nAgent history JSON:", history_file)
    print("Full screen recording:", full_video)

    # 5) Trim the video based on detected stillness and produce a time-warp + remapped history
    trimmed_video = _next_available_path(videos_dir / "demo_trimmed.mp4")
    warp_json = logs_dir / "timewarp.json"
    remapped_history_json = logs_dir / "agent_history_trimmed.json"

    try:
        trim_video(
            input_path=str(full_video),
            output_path=str(trimmed_video),
            mode='cut',                 # drop still stretches entirely
            still_min_seconds=2.5,      # treat >=2.5s still as removable
            frame_step=3,               # slightly denser sampling for accuracy
            diff_threshold=1.2,         # stricter stillness detection
            warp_out_path=str(warp_json),
            history_json_path=str(history_file),
            remapped_history_path=str(remapped_history_json),
        )
    except Exception as e:
        print("Trimming failed:", e)
        return

    # 6) Build transcript segments from remapped history (fallback to original if remap missing)
    transcript_source = remapped_history_json if remapped_history_json.exists() else history_file
    raw_segments = segments_from_history_json(str(transcript_source))
    if not raw_segments:
        print("No segments found in history; cannot synthesize voiceover.")
        return

    fitted_segments = fit_transcript_to_time(
        raw_segments,
        wpm=155,
        safety=0.88,
        min_words_per_segment=3,
        max_words_per_segment=40,
        model="claude-sonnet-4-20250514",
        style_hint=TRANSCRIPT_STYLE,
    )

    # 7) Write SRT captions for the fitted transcript
    srt_path = logs_dir / f"captions_{ts}.srt"
    _write_srt(fitted_segments, str(srt_path))

    # 8) Synthesize TTS per segment via ElevenLabs, then pad/trim to exact duration
    seg_dir = audio_dir / f"segments_{ts}"
    seg_dir.mkdir(parents=True, exist_ok=True)

    tts_wavs_fit = []
    for idx, seg in enumerate(fitted_segments):
        target = max(0.05, (seg.end - seg.start).total_seconds())
        seg_wav = seg_dir / f"seg_{idx:04}.wav"
        text = (seg.text or "").strip()
        if not text:
            # generate 1ms silence if no text
            AudioSegment.silent(duration=1).export(str(seg_wav), format="wav")
        else:
            # synthesize with ElevenLabs
            synthesize_to_file(ELEVEN_VOICE_ID, text, str(seg_wav))

        seg_fit_wav = seg_dir / f"seg_{idx:04}_fit.wav"
        _pad_or_trim_to_duration(str(seg_wav), str(seg_fit_wav), target_seconds=target)
        tts_wavs_fit.append(str(seg_fit_wav))

    # 9) Build full-length VO track and mix with background (ducking during VO)
    video_dur = _ffprobe_duration(str(trimmed_video))
    vo_full = _build_voiceover_track(fitted_segments, tts_wavs_fit, total_seconds=video_dur, sample_rate=44100)

    # Background: use trimmed video audio if present; else silence bed
    orig_wav = audio_dir / f"orig_{ts}.wav"
    if _has_audio_stream(str(trimmed_video)):
        _extract_audio(str(trimmed_video), str(orig_wav), sr=44100)
        bg = AudioSegment.from_file(str(orig_wav)).set_frame_rate(44100).set_channels(2)
    else:
        bg = AudioSegment.silent(duration=int(video_dur * 1000), frame_rate=44100).set_channels(2)

    bg_ducked = _duck_background(bg, fitted_segments, duck_db=10.0)
    # 9.1) Optional: generate Suno background music, loop/trim to video length, and add quietly
    try:
        bg_prompt = "ambient, minimal, modern tech demo background music"
        bg_music_path = suno.generate_background_music(bg_prompt, target_duration_seconds=int(max(1, video_dur)))
        bgm = AudioSegment.from_file(str(bg_music_path)).set_frame_rate(44100).set_channels(2)
        # Loop or trim to exactly match video duration
        target_ms = int(video_dur * 1000)
        if len(bgm) < target_ms:
            times = target_ms // max(1, len(bgm)) + 1
            bgm = (bgm * times)[:target_ms]
        elif len(bgm) > target_ms:
            bgm = bgm[:target_ms]
        # Reduce volume for background
        bgm = bgm - 16
        # Mix: base (ducked original) + bgm + VO
        mixed = bg_ducked.overlay(bgm).overlay(vo_full)
    except Exception as e:
        print("Background music generation failed or skipped:", e)
        mixed = bg_ducked.overlay(vo_full)
    mixed_wav = audio_dir / f"mixed_{ts}.wav"
    mixed.export(str(mixed_wav), format="wav")

    # 10) Mux new audio and burn subtitles onto the trimmed video
    final_video = _next_available_path(videos_dir / "demo_voiceover.mp4")
    _mux_audio_and_burn_subs(str(trimmed_video), str(mixed_wav), str(srt_path), str(final_video))

    print("\nArtifacts:")
    print("- Full recording:", full_video)
    print("- Trimmed video:", trimmed_video)
    print("- Captions SRT:", srt_path)
    print("- Mixed audio:", mixed_wav)
    print("- Final video with VO + captions:", final_video)

if __name__ == "__main__":
    asyncio.run(main())
