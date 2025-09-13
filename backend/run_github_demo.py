import asyncio
import json
from pathlib import Path

from main import create_demo_video_with_timestamps, _build_from_history_obj
from screen_record import ScreenRecorder
# from trim_from_history import main as trim_video
from trim_from_history import jumpcut_video as trim_video

URL = "github.com"
DESCRIPTION = (
    "Sign in with email 'hacker41832@gmail.com' and password 'Hacker418'. "
    "Make a new repo with a random name. "
    "Then respond 'done' and stop."
)


async def main() -> None:
    # 1) Start full-screen recorder (saved under artifacts/videos)
    full_video = Path("artifacts/videos/demo_full2.mp4")
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
        print("No history_path returned; cannot trim.")
        return

    history_file = Path(history_path)
    if not history_file.exists():
        print(f"Agent history file not found at {history_file}; cannot trim.")
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

    # 5) Trim the video based on precise step start/end epoch times
    trimmed_video = Path("artifacts/videos/demo_trimmed2.mp4")
    warp_json = Path("artifacts/logs/timewarp.json")
    remapped_history_json = Path("artifacts/logs/agent_history_trimmed.json")

    try:
        trim_video(
            # history_json_path=str(history_file),
            input_path=str(full_video),
            output_path=str(trimmed_video),
            mode='cut',
            still_min_seconds=3.0,
            frame_step=10,
            # warp_out_path=str(warp_json),
            # remapped_history_path=str(remapped_history_json),
        )
    except Exception as e:
        print("Trimming failed:", e)
        return

    print("\nTrimmed video:", trimmed_video)
    print("Timewarp mapping:", warp_json)
    print("Remapped history:", remapped_history_json)


if __name__ == "__main__":
    asyncio.run(main())
