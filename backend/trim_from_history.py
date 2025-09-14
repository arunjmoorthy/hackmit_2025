import os
import subprocess
import tempfile
import math
import cv2
import numpy as np
from tqdm import tqdm
import json
from typing import List, Tuple, Dict, Any, Optional

# Tuning constants
PRESERVE_STEP_WINDOWS = False  # Set True to force-keep full step windows (reduces trimming)
KEEP_LEAD_PAD = 0.15          # seconds of context before each kept segment
KEEP_TAIL_PAD = 0.15          # seconds of context after each kept segment

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
    If t is inside a removed gap, snaps to the start of the next kept segment.
    """
    pieces = warp.get("pieces", [])
    if not pieces:
        return t
    
    # If before first kept segment, map to 0
    if t < pieces[0]["src_start"]:
        return 0.0
    
    # Check if t falls within any kept segment
    for p in pieces:
        src_start, src_end, dst_start = p["src_start"], p["src_end"], p["dst_start"]
        if src_start <= t < src_end:
            # t is within this kept segment
            return dst_start + (t - src_start)
    
    # t is after all kept segments, map to end of trimmed video
    last = pieces[-1]
    return last["dst_start"] + (last["src_end"] - last["src_start"])

def _warp_interval_with_overlap(start_s: float, end_s: float, warp: Dict[str, Any]) -> Tuple[float, float, float]:
    """
    Map [start_s, end_s] on original timeline into the trimmed timeline using piecewise mapping.
    Returns (trimmed_start, trimmed_end, overlap_len_seconds).
    If the interval has no overlap with any kept segment, returns the next dst boundary with 0 overlap.
    """
    pieces = warp.get("pieces", [])
    if not pieces:
        return start_s, end_s, max(0.0, end_s - start_s)

    # Entirely before first kept segment
    if end_s <= pieces[0]["src_start"]:
        return 0.0, 0.0, 0.0

    trimmed_start: Optional[float] = None
    trimmed_end: Optional[float] = None
    total_overlap = 0.0

    for p in pieces:
        a, b, C = p["src_start"], p["src_end"], p["dst_start"]
        ov_s = max(start_s, a)
        ov_e = min(end_s, b)
        if ov_e > ov_s:
            dst_s = C + (ov_s - a)
            dst_e = C + (ov_e - a)
            if trimmed_start is None:
                trimmed_start = dst_s
            trimmed_end = dst_e
            total_overlap += (ov_e - ov_s)

    if trimmed_start is None:
        # No overlap; snap to next kept piece or end
        for p in pieces:
            if start_s < p["src_end"]:
                return p["dst_start"], p["dst_start"], 0.0
        last = pieces[-1]
        end_dst = last["dst_start"] + (last["src_end"] - last["src_start"])
        return end_dst, end_dst, 0.0

    return round(float(trimmed_start), 3), round(float(trimmed_end or trimmed_start), 3), round(float(total_overlap), 3)

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

        new_items: List[Dict[str, Any]] = []
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

            t_start_trim, t_end_trim, ov_len = _warp_interval_with_overlap(start_s, end_s, warp)
            item.setdefault("metadata", {})
            item["metadata"]["trimmed_step_start"] = t_start_trim
            item["metadata"]["trimmed_step_end"] = t_end_trim
            item["metadata"]["trimmed_overlap_length"] = ov_len
            print(f"[REMAP DEBUG] Step {item.get('metadata', {}).get('step_number', '?')}: "
                  f"epoch {start_s:.3f}-{end_s:.3f} -> trimmed {t_start_trim:.3f}-{t_end_trim:.3f} (ov {ov_len:.3f})")
            if ov_len > 0.05:
                new_items.append(item)
        out["history"] = new_items
        return out

    # Fallback to cumulative durations when epochs missing
    cursor = 0.0
    new_items: List[Dict[str, Any]] = []
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
        t_start_trim, t_end_trim, ov_len = _warp_interval_with_overlap(start_s, end_s, warp)
        item.setdefault("metadata", {})
        item["metadata"]["trimmed_step_start"] = t_start_trim
        item["metadata"]["trimmed_step_end"] = t_end_trim
        item["metadata"]["trimmed_overlap_length"] = ov_len
        print(f"[REMAP DEBUG] Step {item.get('metadata', {}).get('step_number', '?')}: "
              f"duration {start_s:.3f}-{end_s:.3f} -> trimmed {t_start_trim:.3f}-{t_end_trim:.3f} (ov {ov_len:.3f})")
        cursor = end_s
        if ov_len > 0.05:
            new_items.append(item)
    out["history"] = new_items
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

def merge_intervals(intervals: List[Tuple[float, float]], pad: float = 0.0) -> List[Tuple[float, float]]:
    """
    Merge overlapping/touching intervals; optionally expand each by 'pad' seconds on both ends.
    """
    if not intervals:
        return []
    expanded = [(max(0.0, s - pad), max(s - pad, e + pad)) for (s, e) in intervals]
    expanded.sort()
    merged: List[Tuple[float, float]] = []
    cs, ce = expanded[0]
    for s, e in expanded[1:]:
        if s <= ce + 1e-3:
            ce = max(ce, e)
        else:
            merged.append((cs, ce))
            cs, ce = s, e
    merged.append((cs, ce))
    return merged

def pad_keep_segments(keep: List[Tuple[float, float]], lead: float, tail: float, duration: float) -> List[Tuple[float, float]]:
    if not keep:
        return []
    padded = []
    for s, e in keep:
        ns = max(0.0, s - lead)
        ne = min(duration, e + tail)
        if ne > ns:
            padded.append((ns, ne))
    # merge touching
    padded.sort()
    merged: List[Tuple[float, float]] = []
    cs, ce = padded[0]
    for s, e in padded[1:]:
        if s <= ce + 1e-3:
            ce = max(ce, e)
        else:
            merged.append((cs, ce))
            cs, ce = s, e
    merged.append((cs, ce))
    return merged

def windows_from_history(history_obj: Dict[str, Any]) -> Tuple[List[Tuple[float, float]], float]:
    """
    Extract step windows in seconds relative to first step start if epoch times exist,
    otherwise use cumulative metadata.duration_seconds. Returns (windows, origin_epoch).
    """
    items = history_obj.get("history") or []
    items = [i for i in items if isinstance(i, dict)]
    # Check for epochs
    epochs: List[Optional[float]] = []
    for it in items:
        meta = it.get("metadata") or {}
        try:
            epochs.append(float(meta.get("step_start_time")))
        except Exception:
            epochs.append(None)
    have_epochs = any(e is not None for e in epochs)

    if have_epochs:
        first_epoch: Optional[float] = None
        for it in items:
            meta = it.get("metadata") or {}
            try:
                v = float(meta.get("step_start_time"))
            except Exception:
                v = None
            if v is not None:
                first_epoch = v
                break
        if first_epoch is None:
            first_epoch = 0.0
        wins: List[Tuple[float, float]] = []
        for it in items:
            meta = it.get("metadata") or {}
            try:
                s = float(meta.get("step_start_time"))
            except Exception:
                s = None
            try:
                e = float(meta.get("step_end_time"))
            except Exception:
                e = None
            if s is not None and e is not None:
                if e < s:
                    s, e = e, s
                wins.append((max(0.0, s - first_epoch), max(0.0, e - first_epoch)))
        return wins, first_epoch

    # Fallback to cumulative durations
    wins: List[Tuple[float, float]] = []
    cursor = 0.0
    for it in items:
        meta = it.get("metadata") or {}
        try:
            dur = float(meta.get("duration_seconds") or 0.0)
        except Exception:
            dur = 0.0
        s = cursor
        e = cursor + max(0.0, dur)
        wins.append((s, e))
        cursor = e
    return wins, 0.0

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

    # Add small context around kept segments, then merge
    keep = pad_keep_segments(keep, KEEP_LEAD_PAD, KEEP_TAIL_PAD, duration)

    kept_total = sum(e - s for s, e in keep)
    print(f"Keeping {len(keep)} segments, total {kept_total:.2f}s")
    for s, e in keep:
        print(f"  keep: {s:.2f}s → {e:.2f}s ({e - s:.2f}s)")

    # If history provided, ensure we ALWAYS keep around each step window so timestamps map correctly
    if history_json_path and PRESERVE_STEP_WINDOWS:
        try:
            with open(history_json_path, "r", encoding="utf-8") as f:
                history_obj = json.load(f)
            step_wins, _ = windows_from_history(history_obj)
            # Add a small pad around steps to ensure context remains
            combined = merge_intervals(keep + step_wins, pad=0.20)
            keep = combined
            kept_total = sum(e - s for s, e in keep)
            print(f"[ALIGN DEBUG] After union with step windows → keeping {len(keep)} segments, total {kept_total:.2f}s")
            for s, e in keep:
                print(f"  keep: {s:.2f}s → {e:.2f}s ({e - s:.2f}s)")
        except Exception as e:
            print("[ALIGN DEBUG] Failed to union step windows:", e)

    # Build and optionally emit time-warp mapping and remapped history
    warp = build_timewarp(keep)
    print(f"[WARP DEBUG] Time warp mapping:")
    for i, piece in enumerate(warp.get("pieces", [])):
        print(f"  Piece {i}: src {piece['src_start']:.3f}-{piece['src_end']:.3f} -> dst {piece['dst_start']:.3f}-{piece['dst_start'] + (piece['src_end'] - piece['src_start']):.3f}")
    print(f"  Total trimmed duration: {warp.get('total_dst', 0):.3f}s")
    
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
