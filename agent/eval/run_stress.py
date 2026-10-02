from pathlib import Path
from dotenv import load_dotenv
from core.dataset import LocalDiskStore
from eval.runner import run_eval_suite
from eval.stress_scenarios import STRUCTURAL_STRESS_SCENARIOS

load_dotenv()

if __name__ == "__main__":
    store = LocalDiskStore(base_dir=Path("eval_data_stress"))
    run_eval_suite(STRUCTURAL_STRESS_SCENARIOS, store, results_file="eval_results_stress.jsonl")