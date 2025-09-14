import asyncio
import json
from pathlib import Path

from main import create_demo_video_with_timestamps, _build_from_history_obj
from screen_record import ScreenRecorder
# from trim_from_history import main as trim_video
from trim_from_history import jumpcut_video as trim_video
from create_transcript import generate_transcript_from_history

URL = "github.com"
DESCRIPTION = (
    "Sign in with email 'hacker41832@gmail.com' and password 'Hacker418'. "
    "Make a new repo with a random name. "
    "Then respond 'done' and stop."
)


async def main() -> None:
    # 1) Start full-screen recorder (saved under artifacts/videos) with unique filenames
    videos_dir = Path("artifacts/videos")
    videos_dir.mkdir(parents=True, exist_ok=True)

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

    full_video = _next_available_path(videos_dir / "demo_full.mp4")
    recorder = ScreenRecorder(out_path=str(full_video), fps=30, display="auto", audio=None)
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
    logs_dir = Path("artifacts/logs")
    logs_dir.mkdir(parents=True, exist_ok=True)

    trimmed_video = _next_available_path(videos_dir / "demo_trimmed.mp4")
    warp_json = logs_dir / "timewarp.json"
    remapped_history_json = logs_dir / "agent_history_trimmed.json"

    try:
        trim_video(
            input_path=str(full_video),
            output_path=str(trimmed_video),
<<<<<<< Updated upstream
            mode='cut',
            still_min_seconds=3.0,
            frame_step=10,
            warp_out_path=str(warp_json),
            history_json_path=str(history_file),
            remapped_history_path=str(remapped_history_json),
=======
            mode='cap',
            still_min_seconds=1.0,
            cap_seconds=2.0,
            frame_step=4,
            diff_threshold=2,
            # warp_out_path=str(warp_json),
            # remapped_history_path=str(remapped_history_json),
>>>>>>> Stashed changes
        )
    except Exception as e:
        print("Trimming failed:", e)
        return

    # 6) Generate transcript segments aligned to TRIMMED time ranges (use remapped history)
    try:
        segments = generate_transcript_from_history(str(remapped_history_json))
    except Exception as e:
        print("Transcript generation failed:", e)
        return

    # Name transcript file alongside logs, keyed by history timestamp if present
    hist_stem = history_file.stem  # e.g., agent_history_20250913-195800
    ts = hist_stem.replace("agent_history_", "")
    transcript_path = logs_dir / f"transcript_trimmed_{ts}.json"
    with open(transcript_path, "w", encoding="utf-8") as f:
        json.dump({"segments": segments}, f, ensure_ascii=False, indent=2)

    print("\nTranscript:", transcript_path)


if __name__ == "__main__":
    asyncio.run(main())
