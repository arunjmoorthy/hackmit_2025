import asyncio
import json
from pathlib import Path

from main import create_demo_video_with_timestamps, _build_from_history_obj

URL = "github.com"
DESCRIPTION = (
    "Sign in with email 'hacker41832@gmail.com' and password 'Hacker418'. "
    "Make a new repo with a random name. "
    "Then respond 'done' and stop."
)


async def main() -> None:
    result = await create_demo_video_with_timestamps(URL, DESCRIPTION)

    history_path = result.get("history_path")
    if not history_path:
        print("No history_path returned; nothing to show.")
        return

    path = Path(history_path)
    if not path.exists():
        print(f"Agent history file not found at {path}")
        return

    with open(path, "r", encoding="utf-8") as f:
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

    print("\n=== Events (timestamped) ===")
    events = built.get("events", [])
    if events:
        for e in events:
            kind = e.get("event_type")
            t_rel = e.get("t_rel_s")
            step = e.get("step_index")
            msg = e.get("message")
            print(f"[{t_rel:>7.3f}s] step={step} {kind}: {msg}")
    else:
        print("No events parsed from history.")

    print()
    print("Agent history JSON:", path)


if __name__ == "__main__":
    asyncio.run(main())
