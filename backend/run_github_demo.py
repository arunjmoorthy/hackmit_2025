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
    logs_dir = Path("artifacts/logs")
    logs_dir.mkdir(parents=True, exist_ok=True)

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

    # 6) Skip transcript generation for now; focus on remapped history correctness
    if not remapped_history_json.exists():
        print("Warning: remapped history not generated. Check trimming alignment logs.")


if __name__ == "__main__":
    asyncio.run(main())
