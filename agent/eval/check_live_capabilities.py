"""
Live end-to-end check: one request per capability through the real graph and
real model, auto-approving every interrupt. Run from the agent folder with:
    python -m eval.check_live_capabilities
"""
import time
from pathlib import Path
from dotenv import load_dotenv
import numpy as np
import pandas as pd
from langgraph.types import Command

from core.dataset import LocalDiskStore
from core.ingestion import load_dataset
from core.graph import build_agent_graph

load_dotenv()


def main():
    rng = np.random.default_rng(3)
    n = 360
    region = rng.choice(["East", "West", "North"], n, p=[0.5, 0.3, 0.2])
    tenure = rng.normal(24, 8, n)
    visits = rng.normal(10, 3, n)
    shift = {"East": 0.0, "West": 8.0, "North": 15.0}
    spend = 20 + 2.0 * tenure + 1.5 * visits + np.array([shift[r] for r in region]) + rng.normal(0, 5, n)
    df3 = pd.DataFrame({"customer_id": range(1, n + 1), "region": region, "tenure": tenure, "visits": visits, "total_spend": spend})
    df2 = df3[df3.region != "North"].reset_index(drop=True)

    store = LocalDiskStore(base_dir=Path("/tmp/live_capabilities_store"))
    app = build_agent_graph(store)

    runs = [
        ("two_group", df2, "Did customers in the West region spend more than the East region?"),
        ("multi_group", df3, "Does total spend differ across East, West, and North regions?"),
        ("regression", df3, "What factors drive total spend? Consider region, tenure and visits."),
        ("descriptive", df3, "Can you describe this dataset for me?"),
    ]
    for i, (label, df, question) in enumerate(runs):
        try:
            path = Path(f"/tmp/live_{label}.csv")
            df.to_csv(path, index=False)
            ds = load_dataset(path, session_id=f"s-live-{label}", name=f"live_{label}", store=store)
            cfg = {"configurable": {"thread_id": f"live-{label}"}}
            result = app.invoke({"request_text": question, "session_id": f"s-live-{label}", "dataset": ds}, cfg)
            seen = []
            while "__interrupt__" in result:
                p = result["__interrupt__"][0].value
                seen.append(p.get("type"))
                if p.get("type") == "plan_approval":
                    resume = {"action": "approve"}
                elif p.get("type") == "wrangling_approval":
                    resume = {"decisions": [{"index": o["index"], "action": "approve"} for o in p["proposed_operations"]]}
                else:
                    resume = question
                result = app.invoke(Command(resume=resume), cfg)
            plan = result["plan"]
            tr = result.get("test_result") or {}
            kind = tr.get("result_type", "descriptive_result" if "descriptive_result" in result else None)
            print(f"[{label}] interrupts={seen} question_type={plan.question_type.value} "
                  f"status={plan.status.value} result_type={kind} method={tr.get('method')}")
            print("   report:", (result.get("report") or "")[:160].replace("\n", " "))
        except Exception as e:
            print(f"[{label}] FAILED: {type(e).__name__}: {str(e)[:200]}")
        if i < len(runs) - 1:
            time.sleep(20)  # stay under the per-minute token limit


if __name__ == "__main__":
    main()
