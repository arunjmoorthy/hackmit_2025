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

import logging
import re

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


class _EventCollector:
    """Parses console/log lines into structured events with timestamps."""

    step_re = re.compile(r"\bStep\s+(?P<idx>\d+):")
    eval_re = re.compile(r"\bEval:\s*(?P<msg>.*)")
    action_re = re.compile(r"\[ACTION[^\]]*\]\s*(?P<msg>.*)")

    def __init__(self, epoch_started: float):
        self.epoch_started = epoch_started
        self.events: List[Dict[str, Any]] = []
        self.current_step: Optional[int] = None

    def _now_rel(self) -> float:
        return max(0.0, time.time() - self.epoch_started)

    def parse_line(self, line: str) -> Optional[Dict[str, Any]]:
        message = line.rstrip("\n")
        now_rel = round(self._now_rel(), 3)
        event: Optional[Dict[str, Any]] = None

        m_step = self.step_re.search(message)
        if m_step:
            try:
                self.current_step = int(m_step.group("idx"))
            except Exception:
                self.current_step = None
            event = {
                "event_type": "step_start",
                "step_index": self.current_step,
                "message": message,
                "t_rel_s": now_rel,
            }
        else:
            m_eval = self.eval_re.search(message)
            if m_eval:
                event = {
                    "event_type": "eval",
                    "step_index": self.current_step,
                    "message": m_eval.group("msg").strip(),
                    "t_rel_s": now_rel,
                }
            else:
                m_action = self.action_re.search(message)
                if m_action:
                    event = {
                        "event_type": "action",
                        "step_index": self.current_step,
                        "message": m_action.group("msg").strip(),
                        "t_rel_s": now_rel,
                    }
                else:
                    if any(token in message for token in [
                        "Navigated to", "Scrolled", "Clicked", "Searched", "waited", "Final Result", "Task completed"
                    ]):
                        event = {
                            "event_type": "tool",
                            "step_index": self.current_step,
                            "message": message,
                            "t_rel_s": now_rel,
                        }
                    elif any(token in message for token in ["ERROR", "WARNING"]):
                        event = {
                            "event_type": "log",
                            "level": "ERROR" if "ERROR" in message else "WARNING",
                            "step_index": self.current_step,
                            "message": message,
                            "t_rel_s": now_rel,
                        }
        if event:
            self.events.append(event)
        return event


class AgentEventLogger(logging.Handler):
    """Captures browser_use logs via logging module (if routed there)."""

    def __init__(self, collector: _EventCollector):
        super().__init__(level=logging.INFO)
        self.collector = collector

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.collector.parse_line(record.getMessage())
        except Exception:
            return


class _StreamCapture:
    """File-like stream that forwards writes to the event collector with timestamps."""

    def __init__(self, underlying, collector: _EventCollector):
        self._underlying = underlying
        self._collector = collector
        self._buffer = ""

    def write(self, data: str) -> int:
        # Mirror to original stream
        try:
            self._underlying.write(data)
        except Exception:
            pass
        # Accumulate and parse per line
        self._buffer += data
        lines = self._buffer.split("\n")
        self._buffer = lines.pop()  # keep trailing partial
        for line in lines:
            if line:
                self._collector.parse_line(line)
        return len(data)

    def flush(self) -> None:
        try:
            self._underlying.flush()
        except Exception:
            pass


def _derive_step_ranges(events: List[Dict[str, Any]], duration_s: float) -> List[Dict[str, Any]]:
    """Build [start, end) time ranges for each step based on step_start events."""
    steps: List[Dict[str, Any]] = []
    step_starts = [e for e in events if e.get("event_type") == "step_start" and isinstance(e.get("step_index"), int)]
    step_starts.sort(key=lambda e: e["t_rel_s"])  # chronological
    for i, e in enumerate(step_starts):
        start = float(e["t_rel_s"])
        end = float(step_starts[i + 1]["t_rel_s"]) if i + 1 < len(step_starts) else float(duration_s)
        steps.append({
            "step_index": int(e["step_index"]),
            "start_s": round(start, 3),
            "end_s": round(end, 3),
            "duration_s": round(max(0.0, end - start), 3),
        })
    return steps


async def create_demo_video_with_timestamps(url: str, description: str, output_dir: str = "artifacts") -> Dict[str, Any]:
    """
    Runs a browser agent to perform actions on the given URL per the description,
    while attempting to record a screen video and capturing timestamped events.

    Returns a dict with keys: video_path (if known), video_dir, timeline, events, step_ranges, started_at, ended_at, meta_path.
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

    collector = _EventCollector(epoch_started)

    # Attach both logging handler and stdout/stderr capture to be robust
    event_handler = AgentEventLogger(collector)
    root_logger = logging.getLogger()
    root_original_level = root_logger.level
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(event_handler)

    orig_stdout, orig_stderr = sys.stdout, sys.stderr
    sys.stdout = _StreamCapture(orig_stdout, collector)  # type: ignore[assignment]
    sys.stderr = _StreamCapture(orig_stderr, collector)  # type: ignore[assignment]

    try:
        history = await agent.run(max_steps=40)
    finally:
        # Restore streams and logging even if run fails
        sys.stdout = orig_stdout  # type: ignore[assignment]
        sys.stderr = orig_stderr  # type: ignore[assignment]
        root_logger.removeHandler(event_handler)
        root_logger.setLevel(root_original_level)

    run_ended = datetime.now(timezone.utc)
    epoch_ended = time.time()

    timeline: List[Dict[str, Any]] = []

    candidate_attrs = [
        "events", "steps", "items", "records", "actions"
    ]

    extracted = False
    for attr in candidate_attrs:
        try:
            seq = getattr(history, attr, None)
            if not seq:
                continue
            for idx, item in enumerate(seq):
                ts: Optional[float] = None
                text: Optional[str] = None
                try:
                    ts = getattr(item, "timestamp", None) or getattr(item, "ts", None) or getattr(item, "time", None)
                except Exception:
                    ts = None
                try:
                    text = getattr(item, "content", None) or getattr(item, "text", None)
                except Exception:
                    text = None
                if text is None:
                    text = str(item)
                if isinstance(ts, (int, float)):
                    rel = max(0.0, float(ts) - float(epoch_started))
                else:
                    rel = collector.events[idx]["t_rel_s"] if idx < len(collector.events) else max(0.0, (time.time() - epoch_started))
                timeline.append({
                    "index": idx,
                    "t_rel_s": round(rel, 3),
                    "thought": text,
                })
            extracted = True
            break
        except Exception:
            continue

    if not extracted:
        timeline = [
            {"index": 0, "t_rel_s": 0.0, "thought": f"Started demo on {url}"},
            {"index": 1, "t_rel_s": round(epoch_ended - epoch_started, 3), "thought": "Demo completed"},
        ]

    duration_s = round(epoch_ended - epoch_started, 3)
    step_ranges = _derive_step_ranges(collector.events, duration_s)

    video_hint = str(videos_dir)

    meta = {
        "url": url,
        "description": description,
        "started_at": run_started.isoformat(),
        "ended_at": run_ended.isoformat(),
        "duration_s": duration_s,
        "video_dir": video_hint,
        "timeline": timeline,
        "events": collector.events,
        "step_ranges": step_ranges,
    }
    meta_path = logs_dir / f"demo_meta_{timestamp_str}.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\nSaved demo metadata: {meta_path}")
    print(f"Video directory (if recorded): {video_hint}\n")

    return {
        "video_path": None,
        "video_dir": video_hint,
        "timeline": timeline,
        "events": collector.events,
        "step_ranges": step_ranges,
        "started_at": run_started.isoformat(),
        "ended_at": run_ended.isoformat(),
        "meta_path": str(meta_path),
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <URL> [DESCRIPTION]")
        sys.exit(1)

    url_arg = sys.argv[1]
    desc_arg = " ".join(sys.argv[2:]).strip() if len(sys.argv) > 2 else ""

    if desc_arg:
        asyncio.run(create_demo_video_with_timestamps(url_arg, desc_arg))
    else:
        asyncio.run(run(url_arg))
