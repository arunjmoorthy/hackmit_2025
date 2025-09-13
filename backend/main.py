import sys
import asyncio
import os
import json
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

from browser_use import Agent, ChatAnthropic

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
  - Captures a timeline of events with timestamps for later audio alignment
"""

async def run(url: str):
    # 1) Load .env so ANTHROPOPIC_API_KEY is in the environment
    load_dotenv()

    # 2) Choose a Claude model you have access to
    llm = ChatAnthropic(model="claude-sonnet-4-0")
    
    # 3) Create a very explicit, single-step task and keep limits low
    task = (
        f"Open {url}. Wait until the page appears fully loaded. "
        f"Then respond 'done' and stop."
    )

    agent = Agent(
        task=task,
        llm=llm,
        max_actions_per_step=1,
        max_failures=1,
        enable_memory=False,
    )

    # 4) Run for a tiny number of steps so it navigates once and exits
    history = await agent.run(max_steps=2)

    # 5) Optional: print visited URLs from the history object
    try:
        print("Visited URLs:", history.urls())
    except Exception:
        pass


def _extract_item_summary(item: Dict[str, Any]) -> str:
    """Create a concise description of a history item for narration."""
    # Prefer model output text/thought
    model_output = item.get("model_output") or item.get("assistant_output") or {}
    if isinstance(model_output, dict):
        for key in ("content", "text", "thought"):
            if key in model_output and isinstance(model_output[key], str) and model_output[key].strip():
                return model_output[key].strip()
    # If action is present
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
    # Observation / tool result
    observation = item.get("observation") or {}
    if isinstance(observation, dict):
        for key in ("result", "text", "content", "status"):
            if key in observation and isinstance(observation[key], str) and observation[key].strip():
                return observation[key].strip()
    # Fallback to type or stringified item
    if isinstance(item.get("type"), str):
        return f"Step type: {item['type']}"
    return str(item)


def _build_from_history_obj(history_obj: Any) -> Dict[str, Any]:
    """Build events, step ranges, and timeline from AgentHistoryList using cumulative durations."""
    # Access raw dict via pydantic model_dump if available
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

        # Emit start/end events
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

        # Step range
        step_ranges.append({
            "step_index": step_index,
            "start_s": start_s,
            "end_s": end_s,
            "duration_s": round(max(0.0, end_s - start_s), 3),
            "summary": summary,
        })

        # Timeline thought/summary at step start
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
    while attempting to record a screen video and capturing timestamped events.

    Returns a dict with keys: video_path (if known), video_dir, timeline, events, step_ranges, started_at, ended_at, meta_path, history_path.
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

    run_started = datetime.now(timezone.utc)
    epoch_started = time.time()

    history = await agent.run(max_steps=40)

    run_ended = datetime.now(timezone.utc)
    epoch_ended = time.time()

    # Persist full history as emitted by the library (for auditing/analysis)
    history_path = logs_dir / f"agent_history_{timestamp_str}.json"
    try:
        # Prefer the library's own serializer
        if hasattr(history, "save_to_file"):
            history.save_to_file(history_path)
        else:
            # Fallback to dump model
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

    # If the library's computed total differs from wall clock, prefer library
    duration_s = built.get("total_duration_s") or round(epoch_ended - epoch_started, 3)

    video_hint = str(videos_dir)

    meta = {
        "url": url,
        "description": description,
        "started_at": run_started.isoformat(),
        "ended_at": run_ended.isoformat(),
        "duration_s": duration_s,
        "video_dir": video_hint,
        "timeline": built["timeline"],
        "events": built["events"],
        "step_ranges": built["step_ranges"],
        "history_path": str(history_path),
    }
    meta_path = logs_dir / f"demo_meta_{timestamp_str}.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\nSaved demo metadata: {meta_path}")
    print(f"Full agent history: {history_path}")
    print(f"Video directory (if recorded): {video_hint}\n")

    return {
        "video_path": None,
        "video_dir": video_hint,
        "timeline": built["timeline"],
        "events": built["events"],
        "step_ranges": built["step_ranges"],
        "started_at": run_started.isoformat(),
        "ended_at": run_ended.isoformat(),
        "meta_path": str(meta_path),
        "history_path": str(history_path),
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
