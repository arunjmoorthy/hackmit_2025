import os
import re
import json
import math
import datetime as dt
from dataclasses import dataclass
from typing import List, Callable, Optional, Dict, Any

import anthropic  # pip install anthropic
from anthropic import NotFoundError, APIStatusError
from dotenv import load_dotenv

load_dotenv()  # ANTHROPIC_API_KEY

# ----------------------------
# Data model
# ----------------------------
@dataclass
class Segment:
    start: dt.timedelta
    end: dt.timedelta
    text: str

# ----------------------------
# Claude adapter (robust model selection)
# ----------------------------
# replace your _claude_llm with this

def _claude_llm(prompt: str, *, model: str = "claude-sonnet-4-20250514") -> str:
    """
    Minimal Claude wrapper that ONLY calls the specified model.
    Requires ANTHROPIC_API_KEY in env. No fallbacks.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("Set ANTHROPIC_API_KEY in your environment.")

    client = anthropic.Anthropic(api_key=api_key)
    try:
        resp = client.messages.create(
            model=model,
            max_tokens=300,
            temperature=0.3,
            system=(
                "You write concise, fluent voice-over lines that must obey a strict word budget. "
                "Never exceed the given maximum words. Output plain text only."
            ),
            messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        )
    except NotFoundError as e:
        raise RuntimeError(
            f"Anthropic model '{model}' not available on this API key. "
            "Enable it on your account or change the model string."
        ) from e
    except APIStatusError as e:
        # Surface upstream error cleanly
        raise RuntimeError(f"Anthropic API error for model '{model}': {e}") from e

    for block in resp.content:
        if getattr(block, "type", None) == "text" and getattr(block, "text", ""):
            return block.text.strip()
    return ""

# ----------------------------
# Helpers
# ----------------------------
_WORD_RE = re.compile(r"\w+(?:'\w+)?", re.UNICODE)
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PW_HINT_RE = re.compile(r"(password|passcode|pwd)\s*[:=]\s*\S+", re.I)

def count_words(s: str) -> int:
    return len(_WORD_RE.findall(s))

def hard_cap_words(s: str, max_words: int) -> str:
    if max_words <= 0:
        return ""
    words = _WORD_RE.findall(s)
    if len(words) <= max_words:
        return s.strip()
    return " ".join(words[:max_words]).strip()

def sanitize(source: str) -> str:
    """Mask obvious secrets from logs so we don't narrate them."""
    s = _EMAIL_RE.sub("«email»", source)
    s = _PW_HINT_RE.sub(r"\1: «secret»", s)
    return s

def build_prompt(original: str, target_words: int, style_hint: str = "", rolling_context: str = "") -> str:
    style = f"\nStyle hint: {style_hint}" if style_hint else ""
    context = f"\nPrior narration (context): {rolling_context.strip()}" if rolling_context else ""
    return (
        "You are an entrepreneur narrating a DEMO-DAY product walkthrough to judges.\n"
        "Rewrite the line into a polished, clear, and persuasive voice-over.\n"
        "Write in first-person plural ('we'), confident, friendly, and benefit-focused.\n"
        "Explain the intent and the impact for the user; avoid low-level UI mechanics.\n"
        "Use 1–2 full sentences per line, with smooth transitions and complete thoughts.\n"
        "No fragments. No filler. No stage directions. Do not narrate waiting/loading.\n"
        "Keep proper nouns when helpful; generalize any secrets.\n"
        f"MAX WORDS: {target_words}. You MUST stay ≤ this limit.\n"
        "If the budget is small, prefer one complete, information-dense sentence instead of fragments.\n"
        "Plain text only. No bullets, no timestamps, no stage directions."
        f"{style}{context}\n\n"
        f"Source line:\n{original.strip()}\n\n"
        "Rewrite (≤ MAX WORDS):"
    )

# ----------------------------
# JSON → Segments
# ----------------------------
def _pick_source_text(step: Dict[str, Any]) -> str:
    """Choose the best field(s) from a history step to narrate."""
    mo = (step.get("model_output") or {})
    res = (step.get("result") or [])
    st = (step.get("state") or {})
    meta = (step.get("metadata") or {})

    # Priority: 'evaluation_previous_goal' + 'next_goal', else 'thinking', else result summaries.
    fields = []
    if mo.get("evaluation_previous_goal"): fields.append(mo["evaluation_previous_goal"])
    if mo.get("next_goal"): fields.append(mo["next_goal"])
    elif mo.get("thinking"): fields.append(mo["thinking"])
    
    # add small state hint
    if st.get("title"): fields.append(f"Page: {st['title']}")
    if st.get("url"): fields.append(f"URL: {st['url']}")
    # result snippets (avoid flooding)
    for r in res[:2]:
        xc = r.get("extracted_content")
        if xc: fields.append(xc)

    # Join and sanitize
    raw = "  ".join(str(x) for x in fields if x)
    return sanitize(raw) or "Continue."

def segments_from_history_json(path: str) -> List[Segment]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    history = data.get("history") or []
    if not history:
        return []

    # Normalize times to start at 0
    first_start = (history[0].get("metadata") or {}).get("trimmed_step_start")
    if first_start is None:
        # fallback: sequential 4s segments if metadata missing
        t = 0.0
        segs = []
        for step in history:
            text = _pick_source_text(step)
            segs.append(Segment(start=dt.timedelta(seconds=t), end=dt.timedelta(seconds=t+4), text=text))
            t += 4.0
        return segs

    segs: List[Segment] = []
    for step in history:
        meta = (step.get("metadata") or {})
        st0 = meta.get("trimmed_step_start")
        st1 = meta.get("trimmed_step_end")
        if st0 is None or st1 is None:
            continue
        start = max(0.0, float(st0) - float(first_start))
        end = max(start, float(st1) - float(first_start))
        # guard tiny/zero durations -> give minimal 1.5s
        if end - start < 0.5:
            end = start + 1.5
        text = _pick_source_text(step)
        segs.append(Segment(start=dt.timedelta(seconds=start), end=dt.timedelta(seconds=end), text=text))
    return segs

# ----------------------------
# Main fitter (RNN-like rolling context)
# ----------------------------
def fit_transcript_to_time(
    segments: List[Segment],
    *,
    wpm: int = 155,
    safety: float = 0.9,
    min_words_per_segment: int = 3,
    max_words_per_segment: Optional[int] = None,
    llm_fn: Optional[Callable[[str], str]] = None,
    model: str = "claude-sonnet-4-20250514",   # <- your exact model
    style_hint: str = "clear, confident, natural",
    context_keep_chars: int = 480,
    min_seconds_for_min_words: float = 1.8,
) -> List[Segment]:
    if llm_fn is None:
        def _llm(p: str) -> str:
            # strictly use the provided/default model
            return _claude_llm(p, model=model)
        llm = _llm
    else:
        llm = llm_fn

    rolling: str = ""  # accumulate prior outputs
    out: List[Segment] = []

    for seg in sorted(segments, key=lambda s: s.start):
        dur = max(0.0, (seg.end - seg.start).total_seconds())
        # Base time-proportional budget
        budget = max(0, math.floor((dur * wpm / 60.0) * safety))
        # Clamp upper bound first
        if max_words_per_segment is not None:
            budget = min(budget, max_words_per_segment)
        # Only enforce a minimum number of words when the segment is long enough
        if dur >= min_seconds_for_min_words and 0 < budget < min_words_per_segment:
            budget = min_words_per_segment

        if dur <= 0 or budget == 0:
            rewritten = ""
        else:
            prompt = build_prompt(seg.text, target_words=max(0, budget),
                                  style_hint=style_hint, rolling_context=rolling[-context_keep_chars:])
            candidate = llm(prompt).strip()
            rewritten = hard_cap_words(candidate, max(0, budget))

        out.append(Segment(start=seg.start, end=seg.end, text=rewritten))
        # Update rolling context
        if rewritten:
            rolling = (rolling + " " + rewritten).strip()

    return out

# ----------------------------
# CLI usage
# ----------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fit narration to time from a history JSON.")
    parser.add_argument("json_path", help="Path to history JSON")
    parser.add_argument("--wpm", type=int, default=155)
    parser.add_argument("--safety", type=float, default=0.88)
    parser.add_argument("--min_words", type=int, default=3)
    parser.add_argument("--max_words", type=int, default=30)
    parser.add_argument("--model", type=str, default="claude-sonnet-4-20250514", help="Anthropic model name (optional)")
    parser.add_argument("--style", type=str, default="product demo voice-over; concise, confident, friendly")
    args = parser.parse_args()

    raw_segments = segments_from_history_json(args.json_path)
    fitted = fit_transcript_to_time(
        raw_segments,
        wpm=args.wpm,
        safety=args.safety,
        min_words_per_segment=args.min_words,
        max_words_per_segment=args.max_words,
        model=args.model,
        style_hint=args.style,
    )

    # Print result
    for s in fitted:
        t0 = int(s.start.total_seconds()); t1 = int(s.end.total_seconds())
        print(f"[{t0:>3}-{t1:<3}] {count_words(s.text):>2}w :: {s.text}")
