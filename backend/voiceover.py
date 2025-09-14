import os
import json
import math
import tempfile
import subprocess
import argparse
import shutil
from dataclasses import dataclass
from typing import List, Optional, Tuple

from pydub import AudioSegment
import pyttsx3

# Import your fitter
from fit_transcript import (
    segments_from_history_json,
    fit_transcript_to_time,
    Segment,
)

# ----------------------------
# Helpers: ffmpeg / ffprobe
# ----------------------------

def run(cmd: list[str]) -> None:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(cmd)}\n{p.stdout}")

def ffprobe_duration(path: str) -> float:
    """Return media duration (seconds) using ffprobe."""
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

def has_audio_stream(path: str) -> bool:
    """
    Return True if ffprobe finds at least one audio stream.
    """
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
    # one "audio" per audio stream; empty if none
    return "audio" in out

def ffmpeg_subtitles_filter_arg(srt_path: str) -> str:
    """
    Build a subtitles filter argument that works on Windows paths.
    FFmpeg needs backslashes escaped and ':' escaped.
    """
    # Normalize to absolute
    p = os.path.abspath(srt_path)
    # Escape backslashes and colons for filter graph
    p = p.replace("\\", "\\\\").replace(":", "\\:")
    # Quote for safety (ffmpeg filter arg doesn't love quotes, but path with spaces needs them escaped)
    return f"subtitles='{p}'"

def atempo_chain(factor: float) -> str:
    """
    Build an ffmpeg atempo chain (each link between 0.5 and 2.0) that multiplies to 'factor'.
    """
    if factor <= 0:
        factor = 1.0
    stages = []
    remaining = factor
    # break into 0.5..2.0 chunks
    while remaining < 0.5:
        stages.append(0.5)
        remaining /= 0.5
    while remaining > 2.0:
        stages.append(2.0)
        remaining /= 2.0
    stages.append(remaining)
    return ",".join(f"atempo={s:.6f}" for s in stages if 0.499 <= s <= 2.001)

def stretch_audio_to_duration(src_wav: str, dst_wav: str, target_seconds: float) -> None:
    """Time-stretch using ffmpeg atempo to fit exact duration (no pitch correction; simple tempo)."""
    if target_seconds <= 0:
        shutil.copyfile(src_wav, dst_wav)
        return
    src_dur = ffprobe_duration(src_wav)
    if src_dur <= 0:
        shutil.copyfile(src_wav, dst_wav)
        return
    factor = src_dur / target_seconds
    chain = atempo_chain(1.0 / factor)  # we want new_dur = target, so tempo = (src/target)^-1
    cmd = ["ffmpeg", "-y", "-i", src_wav, "-filter:a", chain, "-vn", dst_wav]
    run(cmd)

def extract_audio(input_video: str, out_wav: str, sr: int = 44100) -> None:
    cmd = ["ffmpeg", "-y", "-i", input_video, "-vn", "-ac", "2", "-ar", str(sr), out_wav]
    run(cmd)

def mux_audio_and_burn_subs(input_video: str, audio_wav: str, srt_path: str, out_video: str) -> None:
    """
    Replace audio with audio_wav and burn SRT as hard subtitles.
    """
    tmp_vid = out_video + ".tmp_nosubs.mp4"
    run([
        "ffmpeg", "-y",
        "-i", input_video, "-i", audio_wav,
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy", "-c:a", "aac", "-shortest", tmp_vid
    ])

    # Windows-safe subtitles filter
    sub_filter = ffmpeg_subtitles_filter_arg(srt_path)

    run([
        "ffmpeg", "-y",
        "-i", tmp_vid,
        "-vf", sub_filter,
        "-c:a", "copy",
        out_video
    ])
    os.remove(tmp_vid)

# ----------------------------
# Subtitles
# ----------------------------

def to_srt_timestamp(seconds: float) -> str:
    if seconds < 0:
        seconds = 0
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

def write_srt(segments: List[Segment], path: str) -> None:
    lines = []
    for idx, s in enumerate(segments, start=1):
        t0 = s.start.total_seconds()
        t1 = s.end.total_seconds()
        text = s.text.strip() or " "
        lines.append(str(idx))
        lines.append(f"{to_srt_timestamp(t0)} --> {to_srt_timestamp(t1)}")
        # Allow 1–2 lines; simple wrap by \n if long
        if len(text) > 54:
            # naive wrap
            mid = len(text)//2
            # split at nearest space to mid
            cut = text.rfind(" ", 0, mid)
            if cut == -1:
                cut = mid
            text = text[:cut].strip() + "\n" + text[cut:].strip()
        lines.append(text)
        lines.append("")  # blank between cues
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

# ----------------------------
# TTS
# ----------------------------

def synth_tts_pyttsx3(text: str, out_wav: str, voice_name: Optional[str] = None, rate_wpm: Optional[int] = None) -> None:
    engine = pyttsx3.init()
    # choose voice if provided
    if voice_name:
        for v in engine.getProperty("voices"):
            if voice_name.lower() in (v.name or "").lower():
                engine.setProperty("voice", v.id)
                break
    if rate_wpm:
        # pyttsx3 "rate" is engine-specific; roughly maps to wpm—set directly
        engine.setProperty("rate", rate_wpm)
    engine.save_to_file(text, out_wav)
    engine.runAndWait()
    engine.stop()

# ----------------------------
# Audio mixing with ducking
# ----------------------------

def build_voiceover_track(segments: List[Segment], tts_wavs: List[str], total_seconds: float, sample_rate: int = 44100) -> AudioSegment:
    base = AudioSegment.silent(duration=int(total_seconds * 1000), frame_rate=sample_rate)
    for seg, wav in zip(segments, tts_wavs):
        if not os.path.exists(wav):
            continue
        vo = AudioSegment.from_file(wav).set_frame_rate(sample_rate).set_channels(2)
        start_ms = int(seg.start.total_seconds() * 1000)
        base = base.overlay(vo, position=start_ms)
    return base

from pydub import AudioSegment, effects

def pad_or_trim_to_duration(src_wav: str, dst_wav: str, target_seconds: float) -> None:
    """
    Keep natural speaking speed. Never speed up.
    - If shorter than target: pad with silence to target length.
    - If longer than target: trim with a short fade-out to target length.
    """
    if target_seconds <= 0:
        shutil.copyfile(src_wav, dst_wav)
        return

    clip = AudioSegment.from_file(src_wav).set_channels(2)
    target_ms = int(target_seconds * 1000)
    cur_ms = len(clip)

    if cur_ms < target_ms:
        # pad with silence at end
        pad = AudioSegment.silent(duration=target_ms - cur_ms, frame_rate=clip.frame_rate)
        out = clip + pad
    elif cur_ms > target_ms:
        # trim with gentle 60ms fade-out
        fade = 60 if target_ms > 120 else max(0, target_ms // 2)
        out = clip[:target_ms].fade_out(fade)
    else:
        out = clip

    out.export(dst_wav, format="wav")

def duck_background(bg: AudioSegment, segments: List[Segment], duck_db: float = 10.0) -> AudioSegment:
    """
    Apply simple ducking: reduce bg by duck_db during each VO interval.
    """
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

# ----------------------------
# Main pipeline
# ----------------------------

def main():
    ap = argparse.ArgumentParser(description="Generate voiceover + burned captions from a video and history JSON.")
    ap.add_argument("video", help="Input video file (mp4/mov...)")
    ap.add_argument("history_json", help="History JSON (as used by fit_transcript.py)")
    ap.add_argument("-o", "--out", default="output_captioned_vo.mp4", help="Output video path")
    ap.add_argument("--wpm", type=int, default=155, help="Words per minute for budgeting (fit stage)")
    ap.add_argument("--safety", type=float, default=0.88, help="Safety factor for word budget")
    ap.add_argument("--min_words", type=int, default=3)
    ap.add_argument("--max_words", type=int, default=50)
    ap.add_argument("--model", type=str, default="claude-sonnet-4-20250514")
    ap.add_argument("--style", type=str, default="product demo voice-over; concise, confident, friendly")
    ap.add_argument("--voice", type=str, default=None, help="pyttsx3 voice name contains this substring")
    ap.add_argument("--tts_rate", type=int, default=None, help="pyttsx3 base rate; engine-specific")
    ap.add_argument("--keep", action="store_true", help="Keep temp artifacts (wav/srt)")
    ap.add_argument("--no_speedup", action="store_true",
               help="Never speed up speech; pad or trim instead (recommended).")

    args = ap.parse_args()

    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise RuntimeError("ffmpeg/ffprobe not found on PATH. Please install ffmpeg.")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    # 1) Load + fit captions to timing
    raw_segments = segments_from_history_json(args.history_json)
    if not raw_segments:
        raise RuntimeError("No segments found from history JSON.")

    fitted_segments = fit_transcript_to_time(
        raw_segments,
        wpm=args.wpm,
        safety=args.safety,
        min_words_per_segment=args.min_words,
        max_words_per_segment=args.max_words,
        model=args.model,
        style_hint=args.style,
    )

    # Clamp any segments to video duration
    video_dur = ffprobe_duration(args.video)
    clamped: List[Segment] = []
    for s in fitted_segments:
        t0 = max(0.0, s.start.total_seconds())
        t1 = min(video_dur, max(t0, s.end.total_seconds()))
        clamped.append(Segment(start=s.start.__class__(seconds=t0), end=s.end.__class__(seconds=t1), text=s.text))
    segments = clamped

    # 2) Write SRT
    tempdir = tempfile.mkdtemp(prefix="voiceover_")
    srt_path = os.path.join(tempdir, "captions.srt")
    write_srt(segments, srt_path)

    # 3) Synthesize TTS per segment, then time-stretch to exact duration
    tts_wavs_raw = []
    tts_wavs_fit = []
    for idx, seg in enumerate(segments):
        seg_wav = os.path.join(tempdir, f"seg_{idx:04}.wav")
        text = seg.text.strip()
        if not text:
            # generate 1 frame of silence
            AudioSegment.silent(duration=1).export(seg_wav, format="wav")
        else:
            synth_tts_pyttsx3(text, seg_wav, voice_name=args.voice, rate_wpm=args.tts_rate)
        tts_wavs_raw.append(seg_wav)

    # Stretch to match exact segment duration
    for idx, seg in enumerate(segments):
        src = tts_wavs_raw[idx]
        dst = os.path.join(tempdir, f"seg_{idx:04}_fit.wav")
        target = max(0.05, seg.end.total_seconds() - seg.start.total_seconds())
        pad_or_trim_to_duration(src, dst, target_seconds=target)

        if args.no_speedup:
            pad_or_trim_to_duration(src, dst, target)
        else:
            stretch_audio_to_duration(src, dst, target)  # legacy behavior


        tts_wavs_fit.append(dst)


    # 4) Build voiceover full-length track
    vo_full = build_voiceover_track(segments, tts_wavs_fit, total_seconds=video_dur, sample_rate=44100)

    # 5) Extract original audio, duck it, mix with VO
    orig_wav = os.path.join(tempdir, "orig.wav")
    if has_audio_stream(args.video):
        extract_audio(args.video, orig_wav, sr=44100)
        bg = AudioSegment.from_file(orig_wav).set_frame_rate(44100).set_channels(2)
    else:
        # Make a silent bed if the video has no audio stream
        bg = AudioSegment.silent(duration=int(video_dur * 1000), frame_rate=44100).set_channels(2)

    # Duck background during VO intervals (no-op effect on silence)
    bg_ducked = duck_background(bg, segments, duck_db=10.0)


    # Mix: overlay VO on ducked bg (VO already positioned in timeline)
    mixed = bg_ducked.overlay(vo_full)

    mixed_wav = os.path.join(tempdir, "mixed.wav")
    mixed.export(mixed_wav, format="wav")

    # 6) Mux new audio, burn subtitles
    out_video = args.out
    mux_audio_and_burn_subs(args.video, mixed_wav, srt_path, out_video)

    # 7) Optionally keep artifacts
    if args.keep:
        kept_dir = os.path.splitext(out_video)[0] + "_artifacts"
        os.makedirs(kept_dir, exist_ok=True)
        shutil.copyfile(srt_path, os.path.join(kept_dir, "captions.srt"))
        shutil.copyfile(mixed_wav, os.path.join(kept_dir, "mixed.wav"))
        for p in tts_wavs_fit:
            shutil.copyfile(p, os.path.join(kept_dir, os.path.basename(p)))
        print(f"Artifacts saved to: {kept_dir}")
    else:
        shutil.rmtree(tempdir, ignore_errors=True)

    print(f"✅ Done. Wrote: {out_video}")

if __name__ == "__main__":
    main()
