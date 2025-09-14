import os
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from anthropic import Anthropic


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _extract_text_bits(item: Dict[str, Any]) -> Dict[str, str]:
    """
    Pull concise, human-meaningful strings from the history item to help the LLM
    craft a natural narration.
    """
    out: Dict[str, str] = {}

    model_output = item.get("model_output") or {}
    if isinstance(model_output, dict):
        for k in (
            "evaluation_previous_goal",
            "memory",
            "next_goal",
            "thinking",
        ):
            v = model_output.get(k)
            if isinstance(v, str) and v.strip():
                out[k] = v.strip()

        # Sometimes actions are embedded here
        action = model_output.get("action")
        if action:
            try:
                out["action"] = json.dumps(action, ensure_ascii=False)
            except Exception:
                out["action"] = str(action)

    # Results from tools/executions
    results = item.get("result")
    if isinstance(results, list) and results:
        # Grab a couple of representative strings
        extracted: List[str] = []
        for r in results[:2]:
            if not isinstance(r, dict):
                continue
            for key in ("extracted_content", "status", "long_term_memory"):
                val = r.get(key)
                if isinstance(val, str) and val.strip():
                    extracted.append(val.strip())
                    break
        if extracted:
            out["result_summaries"] = " | ".join(extracted)

    # Page context
    state = item.get("state") or {}
    if isinstance(state, dict):
        url = state.get("url")
        title = state.get("title")
        if isinstance(url, str) and url:
            out["url"] = url
        if isinstance(title, str) and title:
            out["title"] = title

    return out


def _build_prompt_for_step(step_index: int, context: Dict[str, str]) -> str:
    """
    Create a concise user prompt that guides the model to produce a 1–2 sentence
    natural voiceover narration for this time window.
    """
    lines: List[str] = []
    lines.append("You are writing a cohesive demo voiceover that will be combined into a single paragraph.")
    lines.append("For this step, write ONE concise sentence that flows naturally with adjacent sentences, in first-person past tense.")
    lines.append("Avoid meta narration or UI element IDs; describe the visible action plainly (e.g., 'I clicked Sign in and opened the login page').")
    lines.append("Do not repeat details already implied (titles/URLs) unless needed for clarity.")
    lines.append("Output ONLY the sentence with no quotes, labels, bullets, or headings.")
    lines.append("")
    lines.append(f"Step: {step_index}")

    if context.get("title") or context.get("url"):
        lines.append("Page context:")
        if context.get("title"):
            lines.append(f"- Title: {context['title']}")
        if context.get("url"):
            lines.append(f"- URL: {context['url']}")

    for key in (
        "evaluation_previous_goal",
        "next_goal",
        "thinking",
        "action",
        "result_summaries",
        "memory",
    ):
        if context.get(key):
            pretty = key.replace("_", " ").title()
            lines.append(f"{pretty}: {context[key]}")

    return "\n".join(lines)


def _collect_step_windows(history: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract per-step windows with epoch times if available; otherwise fall back
    to cumulative durations. Returns a list of dicts with fields:
      - index (1-based)
      - start_epoch, end_epoch (Optional[float])
      - start_s, end_s (floats, relative to first step start if epochs exist;
        otherwise cumulative from 0 using duration_seconds)
      - raw (original step item)
    """
    items = history.get("history") or []
    items = [i for i in items if isinstance(i, dict)]

    # Prefer pre-trimmed times if present (produced by remapping after trimming)
    for it in items:
        meta = it.get("metadata") or {}
        if meta.get("trimmed_step_start") is not None and meta.get("trimmed_step_end") is not None:
            windows: List[Dict[str, Any]] = []
            for idx, it2 in enumerate(items, start=1):
                meta2 = it2.get("metadata") or {}
                s = _safe_float(meta2.get("trimmed_step_start"))
                e = _safe_float(meta2.get("trimmed_step_end"))
                if s is None or e is None:
                    s, e = 0.0, 0.0
                if e < s:
                    s, e = e, s
                windows.append({
                    "index": idx,
                    "start_epoch": None,
                    "end_epoch": None,
                    "start_s": round(float(s), 3),
                    "end_s": round(float(e), 3),
                    "raw": it2,
                })
            return windows

    # Prefer absolute epochs
    epochs: List[Optional[float]] = []
    for it in items:
        meta = it.get("metadata") or {}
        epochs.append(_safe_float(meta.get("step_start_time")))

    have_epochs = any(e is not None for e in epochs)

    windows: List[Dict[str, Any]] = []

    if have_epochs:
        # Compute relative clock based on first available start epoch
        first_epoch: Optional[float] = None
        for it in items:
            meta = it.get("metadata") or {}
            s_epoch = _safe_float(meta.get("step_start_time"))
            if s_epoch is not None:
                first_epoch = s_epoch
                break
        if first_epoch is None:
            first_epoch = 0.0

        for idx, it in enumerate(items, start=1):
            meta = it.get("metadata") or {}
            s_epoch = _safe_float(meta.get("step_start_time"))
            e_epoch = _safe_float(meta.get("step_end_time"))
            if s_epoch is not None and e_epoch is not None and e_epoch < s_epoch:
                s_epoch, e_epoch = e_epoch, s_epoch

            if s_epoch is not None and e_epoch is not None:
                start_s = max(0.0, s_epoch - first_epoch)
                end_s = max(start_s, e_epoch - first_epoch)
            else:
                # Fallback to 0-length window; will be handled downstream
                start_s = 0.0
                end_s = 0.0

            windows.append(
                {
                    "index": idx,
                    "start_epoch": s_epoch,
                    "end_epoch": e_epoch,
                    "start_s": round(start_s, 3),
                    "end_s": round(end_s, 3),
                    "raw": it,
                }
            )
    else:
        # Fall back to cumulative durations from metadata.duration_seconds
        t = 0.0
        for idx, it in enumerate(items, start=1):
            meta = it.get("metadata") or {}
            dur = _safe_float(meta.get("duration_seconds")) or 0.0
            s = t
            e = t + max(0.0, dur)
            windows.append(
                {
                    "index": idx,
                    "start_epoch": None,
                    "end_epoch": None,
                    "start_s": round(s, 3),
                    "end_s": round(e, 3),
                    "raw": it,
                }
            )
            t = e

    return windows


def generate_transcript_from_history(
    history_json_path: str,
    *,
    model: str = "claude-sonnet-4-20250514",
    max_tokens: int = 200,
    temperature: float = 0.2,
) -> List[Dict[str, Any]]:
    """
    Generate a per-step transcript for a browser-use agent history JSON.

    Returns a list of segments like:
      {
        "index": 1,
        "start_s": 0.0,
        "end_s": 2.18,
        "start_epoch": 1.75779e9,
        "end_epoch": 1.75779e9,
        "title": "Sign in to GitHub · GitHub",
        "url": "https://github.com/login",
        "text": "I click Sign in and land on the GitHub login page.",
      }
    """
    load_dotenv()
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set in the environment.")

    client = Anthropic(api_key=api_key)

    path = Path(history_json_path)
    history: Dict[str, Any] = json.loads(path.read_text())

    windows = _collect_step_windows(history)

    segments: List[Dict[str, Any]] = []
    for w in windows:
        raw_item = w["raw"]
        step_idx = w["index"]
        ctx = _extract_text_bits(raw_item)
        prompt = _build_prompt_for_step(step_idx, ctx)

        resp = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=(
                "You write natural, concise voiceover narration for product demo videos. "
                "Use past tense, first person singular. Be factual and avoid speculation."
            ),
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )

        # Extract plain text from the response
        text_out = ""
        try:
            parts = resp.content or []  # type: ignore[attr-defined]
            for p in parts:
                if getattr(p, "type", None) == "text":
                    text_out += getattr(p, "text", "")
        except Exception:
            pass
        text_out = (text_out or "").strip()

        seg: Dict[str, Any] = {
            "index": step_idx,
            "start_s": w["start_s"],
            "end_s": w["end_s"],
            "start_epoch": w.get("start_epoch"),
            "end_epoch": w.get("end_epoch"),
            "text": text_out,
        }
        # Include page context for convenience
        if ctx.get("title"):
            seg["title"] = ctx["title"]
        if ctx.get("url"):
            seg["url"] = ctx["url"]

        segments.append(seg)

    return segments


def segments_to_paragraph(segments: List[Dict[str, Any]]) -> str:
    """
    Join segment texts into a single clean paragraph.
    - Trims whitespace
    - Ensures each segment ends with appropriate punctuation
    - Joins with a single space
    """
    sentences: List[str] = []
    for seg in segments:
        txt = (seg.get("text") or "").strip()
        if not txt:
            continue
        if txt[-1] not in ".!?":
            txt = txt + "."
        sentences.append(txt)
    return " ".join(sentences)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Generate per-step transcript from agent history JSON using Anthropic.")
    p.add_argument("history_json", help="Path to agent_history_*.json")
    p.add_argument("--out", help="Optional path to write transcript JSON")
    p.add_argument("--paragraph_out", help="Optional path to write a single-paragraph transcript text file")
    p.add_argument("--model", default="claude-3-5-sonnet-20240620")
    p.add_argument("--max_tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.2)
    args = p.parse_args()

    segs = generate_transcript_from_history(
        args.history_json,
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    out_json = json.dumps({"segments": segs}, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).write_text(out_json)
        print(f"Wrote transcript: {args.out}")
    else:
        print(out_json)

    if args.paragraph_out:
        para = segments_to_paragraph(segs)
        Path(args.paragraph_out).write_text(para)
        print(f"Wrote paragraph transcript: {args.paragraph_out}")


