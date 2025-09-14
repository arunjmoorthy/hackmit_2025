import os
import subprocess
import tempfile
import math
import cv2
import numpy as np
from tqdm import tqdm
import json
from typing import List, Tuple, Dict, Any, Optional

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

def build_timewarp(keep_intervals: List[Tuple[float, float]]) -> Dict[str, Any]:
    """
    Build piecewise linear mapping from original video time -> trimmed video time.
    For each kept interval [a,b), it maps to [C, C + (b-a)), where C is cumulative.
    Returns a dict with 'pieces' and 'total_dst'.
    """
    pieces: List[Dict[str, float]] = []
    cumulative = 0.0
    for (a, b) in keep_intervals:
        length = max(0.0, b - a)
        pieces.append({"src_start": float(a), "src_end": float(b), "dst_start": float(cumulative)})
        cumulative += length
    return {"pieces": pieces, "total_dst": float(cumulative)}

def warp_time(t: float, warp: Dict[str, Any]) -> float:
    """
    Map original time t (seconds since video start) to trimmed time using warp mapping.
    If t is inside a removed gap, returns the next dst_start (i.e., snaps forward).
    """
    for p in warp.get("pieces", []):
        a, b, C = p["src_start"], p["src_end"], p["dst_start"]
        if a <= t < b:
            return C + (t - a)
    if warp.get("pieces"):
        if t < warp["pieces"][0]["src_start"]:
            return 0.0
        last = warp["pieces"][-1]
        return last["dst_start"] + (last["src_end"] - last["src_start"])
    return t

def remap_history_durations_to_trimmed(history: Dict[str, Any], warp: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remap per-step start/end times onto the TRIMMED clock.
    Preference order for source times:
      1) metadata.step_start_time / metadata.step_end_time (epoch seconds)
         → convert to video-relative seconds using first available step_start_time
      2) metadata.duration_seconds cumulative timeline
    Writes metadata.trimmed_step_start and metadata.trimmed_step_end for each step.
    """
    out = json.loads(json.dumps(history))
    items = out.get("history") or []
    if not isinstance(items, list):
        return out

    # Check for epoch-based timing
    epochs: List[Optional[float]] = []
    for it in items:
        meta = it.get("metadata") or {}
        try:
            epochs.append(float(meta.get("step_start_time")))
        except Exception:
            epochs.append(None)

    have_epochs = any(e is not None for e in epochs)

    if have_epochs:
        # Compute relative 0 from first available start epoch
        first_epoch: Optional[float] = None
        for it in items:
            meta = it.get("metadata") or {}
            try:
                val = float(meta.get("step_start_time"))
            except Exception:
                val = None
            if val is not None:
                first_epoch = val
                break
        if first_epoch is None:
            first_epoch = 0.0

        for item in items:
            if not isinstance(item, dict):
                continue
            meta = item.get("metadata") or {}
            try:
                s_epoch = float(meta.get("step_start_time"))
            except Exception:
                s_epoch = None
            try:
                e_epoch = float(meta.get("step_end_time"))
            except Exception:
                e_epoch = None

            if s_epoch is not None and e_epoch is not None and e_epoch < s_epoch:
                s_epoch, e_epoch = e_epoch, s_epoch

            if s_epoch is not None and e_epoch is not None:
                start_s = max(0.0, s_epoch - first_epoch)
                end_s = max(start_s, e_epoch - first_epoch)
            else:
                # Fallback to zero-length
                start_s = 0.0
                end_s = 0.0

            t_start_trim = round(warp_time(start_s, warp), 3)
            t_end_trim = round(warp_time(end_s, warp), 3)
            item.setdefault("metadata", {})
            item["metadata"]["trimmed_step_start"] = t_start_trim
            item["metadata"]["trimmed_step_end"] = t_end_trim
        return out

    # Fallback to cumulative durations when epochs missing
    cursor = 0.0
    for item in items:
        if not isinstance(item, dict):
            continue
        meta = item.get("metadata") or {}
        try:
            dur = float(meta.get("duration_seconds") or 0.0)
        except Exception:
            dur = 0.0
        start_s = cursor
        end_s = cursor + max(0.0, dur)
        t_start_trim = round(warp_time(start_s, warp), 3)
        t_end_trim = round(warp_time(end_s, warp), 3)
        item.setdefault("metadata", {})
        item["metadata"]["trimmed_step_start"] = t_start_trim
        item["metadata"]["trimmed_step_end"] = t_end_trim
        cursor = end_s
    return out

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
    warp_out_path: Optional[str] = None,
    history_json_path: Optional[str] = None,
    remapped_history_path: Optional[str] = None,
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

    # Build and optionally emit time-warp mapping and remapped history
    warp = build_timewarp(keep)
    if warp_out_path:
        try:
            with open(warp_out_path, "w", encoding="utf-8") as f:
                json.dump(warp, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print("Failed to write warp mapping:", e)

    if history_json_path and remapped_history_path:
        try:
            with open(history_json_path, "r", encoding="utf-8") as f:
                history_obj = json.load(f)
            remapped = remap_history_durations_to_trimmed(history_obj, warp)
            with open(remapped_history_path, "w", encoding="utf-8") as f:
                json.dump(remapped, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print("Failed to remap history:", e)

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
