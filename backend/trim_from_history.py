import os
import subprocess
import tempfile
import math
import cv2
import numpy as np
from tqdm import tqdm

def get_video_duration(path: str) -> float:
    """Return duration in seconds using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", path
    ]
    out = subprocess.check_output(cmd).decode().strip()
    return float(out)

def detect_stills(
    path: str,
    frame_step: int = 2,
    diff_threshold: float = 1.0,
    still_min_seconds: float = 3.0,
):
    """
    Detect intervals where consecutive frames are almost identical.
    - frame_step: analyze every Nth frame for speed.
    - diff_threshold: MSE threshold between gray frames; smaller = stricter (more "still").
    - still_min_seconds: only mark runs at least this long as still.
    Returns: list of (start_sec, end_sec) still intervals (merged, non-overlapping).
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or total_frames <= 0:
        raise RuntimeError("Could not read FPS or frame count.")

    prev_gray = None
    is_still_run = False
    run_start_time = None
    still_intervals = []

    # Helper: mean squared error between frames
    def mse(a, b):
        diff = cv2.absdiff(a, b).astype(np.float32)
        return float((diff * diff).mean())

    # Iterate sampled frames
    for idx in tqdm(range(0, total_frames, frame_step), desc="Scanning frames"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_gray is None:
            prev_gray = gray
            continue

        d = mse(gray, prev_gray)
        t = idx / fps
        # Consider "still" if below threshold
        if d < diff_threshold:
            if not is_still_run:
                is_still_run = True
                run_start_time = t
        else:
            if is_still_run:
                run_end_time = t
                if run_end_time - run_start_time >= still_min_seconds:
                    still_intervals.append((run_start_time, run_end_time))
                is_still_run = False
            prev_gray = gray

    # Close an open run at end
    if is_still_run:
        end_time = total_frames / fps
        if end_time - run_start_time >= still_min_seconds:
            still_intervals.append((run_start_time, end_time))

    cap.release()

    # Merge overlapping/adjacent intervals (robustness)
    if not still_intervals:
        return []
    still_intervals.sort()
    merged = [still_intervals[0]]
    for s, e in still_intervals[1:]:
        ls, le = merged[-1]
        if s <= le + 1e-3:
            merged[-1] = (ls, max(le, e))
        else:
            merged.append((s, e))
    return merged

def invert_and_cap_intervals(
    full_duration: float,
    still_intervals: list,
    cap_seconds: float | None = None,
):
    """
    Build 'keep' intervals from 0..duration given still_intervals.
    - If cap_seconds is None: remove still intervals entirely.
    - If cap_seconds is a float: retain the first cap_seconds inside each still interval.
    Returns list of (start, end) intervals to keep, non-overlapping, sorted.
    """
    keep = []
    cursor = 0.0
    for (s, e) in still_intervals:
        # keep from cursor up to start of still
        if s > cursor:
            keep.append((cursor, s))
        if cap_seconds is not None and cap_seconds > 0:
            # keep first cap_seconds of the still run
            keep.append((s, min(e, s + cap_seconds)))
        cursor = e
    if cursor < full_duration:
        keep.append((cursor, full_duration))

    # Merge tiny gaps
    merged = []
    for seg in keep:
        if not merged:
            merged.append(seg)
        else:
            ps, pe = merged[-1]
            cs, ce = seg
            if cs <= pe + 1e-3:
                merged[-1] = (ps, max(pe, ce))
            else:
                merged.append((cs, ce))
    # Remove zero/negative spans
    merged = [(s, e) for s, e in merged if e - s > 1e-3]
    return merged

def cut_with_ffmpeg(input_path: str, output_path: str, keep_segments: list, reencode=True):
    """
    Cut the video to keep only keep_segments and concatenate.
    Re-encodes segments for accurate, keyframe-independent cuts, then streams copy at concat.
    """
    if not keep_segments:
        raise RuntimeError("No segments to keep; nothing to output.")

    with tempfile.TemporaryDirectory() as tmp:
        part_paths = []
        for i, (s, e) in enumerate(keep_segments):
            part = os.path.join(tmp, f"part_{i:04d}.mp4")
            duration = max(0.0, e - s)
            if duration <= 0:
                continue
            # Re-encode each piece for accuracy & concat-compatibility
            cmd = [
                "ffmpeg", "-y",
                "-ss", f"{s:.3f}",
                "-to", f"{e:.3f}",
                "-i", input_path,
                "-map", "0:v?", "-map", "0:a?",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                "-c:a", "aac", "-b:a", "160k",
                "-movflags", "+faststart",
                part
            ]
            subprocess.run(cmd, check=True)
            part_paths.append(part)

        # Concat
        list_file = os.path.join(tmp, "list.txt")
        with open(list_file, "w") as f:
            for p in part_paths:
                f.write(f"file '{p}'\n")

        # Since parts match codecs, we can stream-copy on concat for speed.
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0", "-i", list_file,
            "-c", "copy",
            output_path
        ]
        subprocess.run(cmd, check=True)
    return output_path

def jumpcut_video(
    input_path: str,
    output_path: str,
    mode: str = "cap",          # "cap" or "cut"
    cap_seconds: float = 2.0,   # used if mode == "cap"
    still_min_seconds: float = 3.0,
    frame_step: int = 2,
    diff_threshold: float = 1.0,
):
    """
    Main wrapper.
    - mode="cut": drop still stretches >= still_min_seconds
    - mode="cap": keep only first cap_seconds of each still stretch >= still_min_seconds
    Tuning tips:
      - Increase diff_threshold if it’s too aggressive (treats small motions as still).
      - Increase still_min_seconds to ignore short pauses.
      - Increase frame_step to scan faster; decrease for more precision.
    """
    duration = get_video_duration(input_path)
    print(f"Duration: {duration:.2f}s")

    stills = detect_stills(
        input_path,
        frame_step=frame_step,
        diff_threshold=diff_threshold,
        still_min_seconds=still_min_seconds,
    )
    print(f"Detected {len(stills)} still intervals:")
    for s, e in stills:
        print(f"  still: {s:.2f}s → {e:.2f}s ({e - s:.2f}s)")

    if mode == "cut":
        keep = invert_and_cap_intervals(duration, stills, cap_seconds=None)
    elif mode == "cap":
        keep = invert_and_cap_intervals(duration, stills, cap_seconds=cap_seconds)
    else:
        raise ValueError("mode must be 'cut' or 'cap'")

    kept_total = sum(e - s for s, e in keep)
    print(f"Keeping {len(keep)} segments, total {kept_total:.2f}s")
    for s, e in keep:
        print(f"  keep: {s:.2f}s → {e:.2f}s ({e - s:.2f}s)")

    cut_with_ffmpeg(input_path, output_path, keep)
    print(f"Done → {output_path}")

if __name__ == "__main__":
    # Example usage:
    # jumpcut_video("input.mp4", "output_jumpcut.mp4",
    #               mode="cap", cap_seconds=2.0,
    #               still_min_seconds=5.0, frame_step=2, diff_threshold=1.0)
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--input")
    p.add_argument("--output")
    p.add_argument("--mode", choices=["cut", "cap"], default="cap")
    p.add_argument("--cap_seconds", type=float, default=2.0)
    p.add_argument("--still_min_seconds", type=float, default=5.0)
    p.add_argument("--frame_step", type=int, default=2)
    p.add_argument("--diff_threshold", type=float, default=1.0)
    args = p.parse_args()

    jumpcut_video(
        args.input,
        args.output,
        mode=args.mode,
        cap_seconds=args.cap_seconds,
        still_min_seconds=args.still_min_seconds,
        frame_step=args.frame_step,
        diff_threshold=args.diff_threshold,
    )
