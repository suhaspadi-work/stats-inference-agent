"""
Entry point for Phase 8's evaluation battery. Run this directly:

    python eval/run_all.py

It's resumable -- if it stops (rate limit, interruption), re-running the
same command picks up from the last completed scenario. Takes roughly
15-25 minutes for the full battery given the pacing needed to stay under
Groq's rate limits. Once it finishes (or you want a progress check), run:

    python eval/summarize.py
"""
from pathlib import Path
from dotenv import load_dotenv
from core.dataset import LocalDiskStore
from eval.runner import run_eval_suite
from eval.scenarios import ALL_SCENARIOS

load_dotenv()

if __name__ == "__main__":
    store = LocalDiskStore(base_dir=Path("eval_data"))
    run_eval_suite(ALL_SCENARIOS, store)