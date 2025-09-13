import json
import math
import subprocess
import shlex
from pathlib import Path
from typing import List, Tuple, Dict

# ---------- CONFIG (tweak as needed) ----------
IDLE_GAP_THRESHOLD_S = 2.0   # gaps >= this are considered downtime and removed
LEAD_PAD_S = 0.20            # add context before each step window
TAIL_PAD_S = 0.20            # add context after each step window
MIN_KEEP_S = 0.50            # drop tiny keep windows under this length
# ---------------------------------------------

def load_history(path: Path) -> Dict:
    return json.loads(path.read_text())

def steps_to_windows(history: Dict) -> List[Tuple[float, float]]:
    """
    Returns list of (start_epoch, end_epoch) per step.
    """
    wins = []
    for item in history.get("history", []):
        meta = item.get("metadata", {})
        t0 = meta.get("step_start_time")
        t1 = meta.get("step_end_time")
        if t0 is None or t1 is None:
            continue
        if t1 < t0:
            t0, t1 = t1, t0
        wins.append((float(t0), float(t1)))
    return wins

def normalize_to_video_clock(windows_epoch: List[Tuple[float, float]]) -> Tuple[List[Tuple[float, float]], float]:
    """
    Convert epoch windows to video-relative seconds: t' = t - first_step_start.
    Returns (windows_video, video_t0_epoch)
    """
    if not windows_epoch:
        return [], 0.0
    t0 = windows_epoch[0][0]
    converted = [(w0 - t0, w1 - t0) for (w0, w1) in windows_epoch]
    return converted, t0

def pad_and_merge(windows: List[Tuple[float, float]],
                  lead_pad: float, tail_pad: float,
                  idle_gap_threshold: float,
                  min_keep: float) -> List[Tuple[float, float]]:
    """
    1) Expand each window by LEAD/Tail pads.
    2) Merge adjacent windows if the gap between them is < idle_gap_threshold.
    3) Drop tiny windows (< min_keep).
    Assumes windows are in ascending order by start time.
    """
    if not windows:
        return []

    # pad
    padded = [(max(0.0, s - lead_pad), e + tail_pad) for (s, e) in windows]

    # merge
    merged: List[Tuple[float, float]] = []
    cs, ce = padded[0]
    for (s, e) in padded[1:]:
        gap = s - ce
        if gap < idle_gap_threshold:
            ce = max(ce, e)  # merge
        else:
            if ce - cs >= min_keep:
                merged.append((cs, ce))
            cs, ce = s, e
    if ce - cs >= min_keep:
        merged.append((cs, ce))

    return merged

def build_timewarp(keep_intervals: List[Tuple[float, float]]) -> Dict:
    """
    Build piecewise linear mapping from original video time -> trimmed video time.
    For each kept interval [a,b), it maps to [C, C + (b-a)), where C is cumulative.
    Returns:
        {
          "pieces": [
              {"src_start": a, "src_end": b, "dst_start": C},
              ...
          ],
          "total_dst": last cumulative duration
        }
    """
    pieces = []
    cumulative = 0.0
    for (a, b) in keep_intervals:
        length = max(0.0, b - a)
        pieces.append({"src_start": a, "src_end": b, "dst_start": cumulative})
        cumulative += length
    return {"pieces": pieces, "total_dst": cumulative}

def warp_time(t: float, warp: Dict) -> float:
    """
    Map original time t (seconds since video start) to trimmed time using warp mapping.
    If t is inside a removed gap, returns the next dst_start (i.e., snaps forward).
    """
    for p in warp["pieces"]:
        a, b, C = p["src_start"], p["src_end"], p["dst_start"]
        if a <= t < b:
            return C + (t - a)
    # if before first kept segment
    if warp["pieces"]:
        if t < warp["pieces"][0]["src_start"]:
            return 0.0
        # if after last kept segment, snap to end
        last = warp["pieces"][-1]
        return last["dst_start"] + (last["src_end"] - last["src_start"])
    return t

def remap_history_times(history: Dict, video_t0_epoch: float, warp: Dict) -> Dict:
    """
    Adds trimmed times for each step: 'trimmed_step_start', 'trimmed_step_end',
    relative to the TRIMMED video timeline.
    """
    out = json.loads(json.dumps(history))  # deep copy
    for item in out.get("history", []):
        meta = item.get("metadata", {})
        s_epoch = meta.get("step_start_time")
        e_epoch = meta.get("step_end_time")
        if s_epoch is None or e_epoch is None:
            continue
        # convert epoch -> original video time (seconds)
        s0 = float(s_epoch) - video_t0_epoch
        e0 = float(e_epoch) - video_t0_epoch
        # warp into trimmed timeline
        item.setdefault("metadata", {})
        item["metadata"]["trimmed_step_start"] = round(warp_time(s0, warp), 3)
        item["metadata"]["trimmed_step_end"] = round(warp_time(e0, warp), 3)
    return out

def _probe_has_audio(input_video: Path) -> bool:
    """Return True if input has at least one audio stream."""
    try:
        cmd = f'ffprobe -v error -select_streams a -show_entries stream=index -of csv=p=0 "{input_video}"'
        out = subprocess.check_output(shlex.split(cmd), stderr=subprocess.DEVNULL)
        return bool(out.strip())
    except Exception:
        return False

def write_ffmpeg_concat_command(input_video: Path, keep_intervals: List[Tuple[float, float]], output_video: Path) -> str:
    """
    Build a filter_complex command that:
    - trims each [start,end) for video (and audio if present)
    - concatenates them back-to-back with reset timestamps
    Produces a single output: output_video
    Returns the full command string (for logging); executes ffmpeg.
    """
    n = len(keep_intervals)
    if n == 0:
        raise RuntimeError("No keep intervals computed; nothing to output.")

    has_audio = _probe_has_audio(input_video)

    v_labels = []
    filter_parts = []
    for i, (start, end) in enumerate(keep_intervals):
        v_lbl = f"v{i}"
        v_labels.append(f"[{v_lbl}]")
        start = max(0.0, start)
        end = max(start, end)
        filter_parts.append(f"[0:v]trim=start={start}:end={end},setpts=PTS-STARTPTS[{v_lbl}]")

    if has_audio:
        a_labels = []
        for i, (start, end) in enumerate(keep_intervals):
            a_lbl = f"a{i}"
            a_labels.append(f"[{a_lbl}]")
            start = max(0.0, start)
            end = max(start, end)
            filter_parts.append(f"[0:a]atrim=start={start}:end={end},asetpts=PTS-STARTPTS[{a_lbl}]")
        v_concat_in = "".join(v_labels)
        a_concat_in = "".join(a_labels)
        filter_parts.append(f"{v_concat_in}concat=n={n}:v=1:a=0[vout]")
        filter_parts.append(f"{a_concat_in}concat=n={n}:v=0:a=1[aout]")
        map_args = ["-map", "[vout]", "-map", "[aout]"]
    else:
        v_concat_in = "".join(v_labels)
        filter_parts.append(f"{v_concat_in}concat=n={n}:v=1:a=0[vout]")
        map_args = ["-map", "[vout]"]

    filter_complex = "; ".join(filter_parts)
    cmd = [
        "ffmpeg", "-y", "-i", str(input_video),
        "-filter_complex", filter_complex,
        *map_args,
        "-c:v", "libx264", "-crf", "18", "-preset", "veryfast"
    ]
    if has_audio:
        cmd += ["-c:a", "aac", "-b:a", "192k"]
    cmd += [str(output_video)]

    subprocess.run(cmd, check=True)
    return " ".join(shlex.quote(x) for x in cmd)

def main(history_json_path: str, input_video_path: str, output_video_path: str, warp_out_path: str, remapped_history_path: str):
    history_path = Path(history_json_path)
    video_in = Path(input_video_path)
    video_out = Path(output_video_path)
    warp_path = Path(warp_out_path)
    remap_path = Path(remapped_history_path)

    history = load_history(history_path)
    step_windows_epoch = steps_to_windows(history)
    if not step_windows_epoch:
        raise RuntimeError("No steps with start/end times found in history.")

    windows_video, video_t0_epoch = normalize_to_video_clock(step_windows_epoch)

    # Merge steps into keep intervals (remove idle gaps)
    keep = pad_and_merge(
        windows_video,
        lead_pad=LEAD_PAD_S,
        tail_pad=TAIL_PAD_S,
        idle_gap_threshold=IDLE_GAP_THRESHOLD_S,
        min_keep=MIN_KEEP_S
    )

    # Build time-warp mapping and save it
    warp = build_timewarp(keep)
    warp_path.write_text(json.dumps(warp, indent=2))

    # Remap the JSON history onto trimmed clock
    remapped = remap_history_times(history, video_t0_epoch, warp)
    remap_path.write_text(json.dumps(remapped, indent=2))

    # Trim & concat with ffmpeg
    cmd = write_ffmpeg_concat_command(video_in, keep, video_out)
    print("FFmpeg command used:\n", cmd)
    print(f"Kept {len(keep)} intervals; trimmed duration = {warp['total_dst']:.3f}s")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Trim screen recording using browser-use step times; output trimmed video + remapped timestamps.")
    p.add_argument("--history", required=True, help="Path to JSON with top-level key 'history' (your pasted structure).")
    p.add_argument("--video", required=True, help="Path to original screen recording (with audio if present).")
    p.add_argument("--out", required=True, help="Path for trimmed mp4, e.g., video_trimmed.mp4")
    p.add_argument("--warp_json", default="timewarp.json", help="Where to write the time-warp mapping JSON.")
    p.add_argument("--remapped_history", default="history_trimmed.json", help="Where to write the remapped history JSON.")
    args = p.parse_args()

    main(args.history, args.video, args.out, args.warp_json, args.remapped_history)
