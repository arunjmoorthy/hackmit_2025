import sys
import asyncio
import os
import json
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

# ✅ Correct public imports
from browser_use import Agent, ChatAnthropic  # no BrowserConfig / LLMConfig
from screen_record import ScreenRecorder

"""
Usage:
  python main.py https://example.com
  python main.py https://example.com "Fill the login form and submit, then visit the dashboard"

What it does:
  - Loads ANTHROPIC_API_KEY from .env
  - Creates a Browser Use Agent with Claude
  - Opens the provided URL
  - If description is provided: follows the description as steps
  - Attempts to record video via Playwright context if supported by library
  - Builds timestamps from agent history durations
"""

async def run(url: str):
    # 1) Load .env so ANTHROPIC_API_KEY is in the environment
    load_dotenv()

    # 2) Choose a Claude model you have access to
    llm = ChatAnthropic(model="claude-sonnet-4-0")
    
    # 3) Create a very explicit, single-step task and keep limits low
    task = (
        f"Open {url}. Wait until the page appears fully loaded. "
        f"Then respond 'done' and stop."
    )

    task = (
        f"Open https://github.com/. Wait until the page appears fully loaded. "
        # f"Sign in with email 'hacker41832@gmail.com' and password 'Hacker418'. "
        # f"Make a new repo with a random name. "
        f"Then respond 'done' and stop."
    )

    agent = Agent(
        task=task,
        llm=llm,
        max_actions_per_step=5,
        max_failures=3,
        enable_memory=False,
    )

    rec = ScreenRecorder(out_path="artifacts/videos/demo_full2.mp4", fps=30, display="auto", audio=None)
    rec.start()

    # 4) Run for a tiny number of steps so it navigates once and exits
    history = await agent.run(max_steps=50)

    rec.stop()

    # 5) Optional: print visited URLs from the history object
    try:
        print("Visited URLs:", history.urls())
    except Exception:
        pass


def _extract_item_summary(item: Dict[str, Any]) -> str:
    """Create a concise description of a history item for narration."""
    model_output = item.get("model_output") or item.get("assistant_output") or {}
    if isinstance(model_output, dict):
        for key in ("content", "text", "thought"):
            if key in model_output and isinstance(model_output[key], str) and model_output[key].strip():
                return model_output[key].strip()
    action = item.get("action") or {}
    if isinstance(action, dict):
        name = action.get("name") or action.get("tool_name")
        params = action.get("input") or action.get("params") or {}
        if name:
            try:
                params_str = json.dumps(params, ensure_ascii=False) if params else ""
            except Exception:
                params_str = str(params)
            return f"Action: {name} {params_str}".strip()
    observation = item.get("observation") or {}
    if isinstance(observation, dict):
        for key in ("result", "text", "content", "status"):
            if key in observation and isinstance(observation[key], str) and observation[key].strip():
                return observation[key].strip()
    if isinstance(item.get("type"), str):
        return f"Step type: {item['type']}"
    return str(item)


def _build_from_history_obj(history_obj: Any) -> Dict[str, Any]:
    """Build events, step ranges, and timeline from AgentHistoryList using cumulative durations."""
    # Accept either pydantic object or raw dict loaded from disk
    if isinstance(history_obj, dict) and "history" in history_obj:
        history_dict = history_obj
    else:
        try:
            history_dict = history_obj.model_dump()
        except Exception:
            try:
                history_dict = history_obj.dict()
            except Exception:
                history_dict = {"history": []}

    items: List[Dict[str, Any]] = []
    try:
        maybe_items = history_dict.get("history")
        if isinstance(maybe_items, list):
            items = [i if isinstance(i, dict) else {} for i in maybe_items]
    except Exception:
        items = []

    events: List[Dict[str, Any]] = []
    step_ranges: List[Dict[str, Any]] = []
    timeline: List[Dict[str, Any]] = []

    t_cursor = 0.0
    for idx, item in enumerate(items):
        metadata = item.get("metadata") or {}
        try:
            duration = float(metadata.get("duration_seconds") or 0.0)
        except Exception:
            duration = 0.0
        start_s = round(t_cursor, 3)
        end_s = round(t_cursor + max(0.0, duration), 3)
        step_index = item.get("step_index")
        if not isinstance(step_index, int):
            step_index = idx + 1

        summary = _extract_item_summary(item)

        events.append({
            "event_type": "step_start",
            "step_index": step_index,
            "message": summary,
            "t_rel_s": start_s,
        })
        events.append({
            "event_type": "step_end",
            "step_index": step_index,
            "message": summary,
            "t_rel_s": end_s,
        })

        step_ranges.append({
            "step_index": step_index,
            "start_s": start_s,
            "end_s": end_s,
            "duration_s": round(max(0.0, end_s - start_s), 3),
            "summary": summary,
        })

        timeline.append({
            "index": idx,
            "t_rel_s": start_s,
            "thought": summary,
        })

        t_cursor = end_s

    return {
        "events": events,
        "step_ranges": step_ranges,
        "timeline": timeline,
        "total_duration_s": round(t_cursor, 3),
        "raw": history_dict,
    }


async def create_demo_video_with_timestamps(url: str, description: str, output_dir: str = "artifacts") -> Dict[str, Any]:
    """
    Runs a browser agent to perform actions on the given URL per the description,
    attempts to record a screen video, and saves the raw agent history JSON.

    Returns a dict with keys: video_dir, history_path, plus parsed events/step_ranges/timeline computed from in-memory history.
    """
    load_dotenv()

    output_root = Path(output_dir)
    videos_dir = output_root / "videos"
    logs_dir = output_root / "logs"
    videos_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

    browser_context_kwargs: Dict[str, Any] = {
        "record_video_dir": str(videos_dir),
        "record_video_size": {"width": 1280, "height": 720},
    }

    llm = ChatAnthropic(model="claude-sonnet-4-0")

    task = (
        f"Open {url}. Then, using the following description, complete the end-to-end flow: \n"
        f"DESCRIPTION: {description}\n"
        f"At each significant action, think concisely about what you are doing in one short sentence,"
        f" then proceed. Do not ask for further confirmation. When complete, respond 'done' and stop."
    )

    # task = (
    #     f"Open {url}. "
    # )

    try:
        agent = Agent(
            task=task,
            llm=llm,
            max_actions_per_step=3,
            max_failures=3,
            enable_memory=False,
            browser_context_kwargs=browser_context_kwargs,  # type: ignore[arg-type]
        )
    except TypeError:
        agent = Agent(
            task=task,
            llm=llm,
            max_actions_per_step=3,
            max_failures=3,
            enable_memory=False,
        )

    history = await agent.run(max_steps=40)

    # Persist full history as emitted by the library (for auditing/analysis)
    history_path = logs_dir / f"agent_history_{timestamp_str}.json"
    try:
        if hasattr(history, "save_to_file"):
            history.save_to_file(history_path)
        else:
            try:
                data = history.model_dump()
            except Exception:
                try:
                    data = history.dict()
                except Exception:
                    data = {}
            with open(history_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    built = _build_from_history_obj(history)

    video_hint = str(videos_dir)

    print(f"Full agent history: {history_path}")
    print(f"Video directory (if recorded): {video_hint}\n")

    return {
        "video_dir": video_hint,
        "history_path": str(history_path),
        "timeline": built["timeline"],
        "events": built["events"],
        "step_ranges": built["step_ranges"],
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <URL> [DESCRIPTION]\n        ")
        sys.exit(1)

    url_arg = sys.argv[1]
    desc_arg = " ".join(sys.argv[2:]).strip() if len(sys.argv) > 2 else ""

    if desc_arg:
        asyncio.run(create_demo_video_with_timestamps(url_arg, desc_arg))
    else:
        asyncio.run(run(url_arg))
