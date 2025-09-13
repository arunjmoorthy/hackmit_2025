import sys
import asyncio
from dotenv import load_dotenv

# ✅ Correct public imports
from browser_use import Agent, ChatAnthropic  # no BrowserConfig / LLMConfig
from screen_record import ScreenRecorder

claude_key = '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'


"""
Usage:
  python main.py https://example.com

What it does:
  - Loads ANTHROPIC_API_KEY from .env
  - Creates a Browser Use Agent with Claude
  - Opens the provided URL
  - Ends immediately after confirming the page is loaded
"""

async def run(url: str):
    # 1) Load .env so ANTHROPIC_API_KEY is in the environment
    load_dotenv()

    # 2) Choose a Claude model you have access to
    #    See "Supported Models" in docs for current names.
    llm = ChatAnthropic(model="claude-sonnet-4-0", api_key=claude_key)  # or "claude-3-5-sonnet-20240620"
    
    # 3) Create a very explicit, single-step task and keep limits low
    task = (
        f"Open {url}. Wait until the page appears fully loaded. "
        f"Then respond 'done' and stop."
    )

    task = (
        f"Open https://github.com/. Wait until the page appears fully loaded. "
        f"Sign in with email 'hacker41832@gmail.com' and password 'Hacker418'. "
        f"Make a new repo with a random name. "
        f"Then respond 'done' and stop."
    )

    agent = Agent(
        task=task,
        llm=llm,
        max_actions_per_step=5,
        max_failures=3,
        enable_memory=False,
    )

    rec = ScreenRecorder(out_path="runs/demo_full.mp4", fps=30, display="auto", audio=None)
    rec.start()

    # 4) Run for a tiny number of steps so it navigates once and exits
    history = await agent.run(max_steps=50)

    rec.stop()

    print(history)

    for hist in history:
        print(hist)
        hist = hist[1]
        for h in hist:
            if h.metadata:
                print(h.metadata)
                print(h.metadata.duration_seconds)
                # total += h.metadata.duration_seconds

    # 5) Optional: print visited URLs from the history object
    try:
        print("Visited URLs:", history.urls())
    except Exception:
        pass

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <URL>")
        sys.exit(1)
    asyncio.run(run(sys.argv[1]))
