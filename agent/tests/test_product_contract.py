"""
M1.5 step P1: the contract between the backend's pauses and the product. A real graph run (fake models) that
reaches every pause, refusals included, records each interrupt payload; the test fails when the backend emits a
pause type the product does not handle, when the product lists one the backend never emits, or when a payload
lacks a key the product reads (product/pause_types.py).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "product"))

from pause_types import HANDLED_PAUSE_TYPES, REQUIRED_PAYLOAD_KEYS  # noqa: E402

from core.dataset import LocalDiskStore  # noqa: E402
from core.graph import build_agent_graph  # noqa: E402
from core.plan import QuestionType  # noqa: E402
from tests.test_analysis_reports import _answers, _with_id  # noqa: E402
from tests.test_decision_point_guard import EXPECTED_PAUSES, _dataset, fake_classify, fake_draft, fake_propose  # noqa: E402
from tests.test_prompt_canaries import fake_model  # noqa: E402,F401 (fixture)
from tests.test_stage1_report import _drive  # noqa: E402


def _messy():
    df = _with_id(n=120, seed=91)
    df.loc[10:25, "tenure"] = np.nan  # above the 5% missingness threshold
    return pd.concat([df, df.iloc[[0, 1]]], ignore_index=True)  # 2 exact duplicate rows


@pytest.fixture
def payloads(fake_model, tmp_path):
    """Every interrupt payload of one run that reaches all five pauses, with a refusal at each of the three
    gates that can refuse: clarification, a refused then approved stage plan, cleaning, a refused edit, an
    accepted edit (plan approval again), approval, the boundary gate refused then run, round two's approval."""
    store = LocalDiskStore(base_dir=tmp_path / "store")
    app = build_agent_graph(store, classify_fn=fake_classify(QuestionType.TWO_GROUP_COMPARISON),
                            draft_fn=fake_draft(["region"]), propose_fn=fake_propose)
    state = {"request_text": "Did West spend more?", "session_id": "session-abc",
             "dataset": _dataset(_messy(), tmp_path, store, "contract")}
    _, pauses, _ = _drive(
        app, state, "contract",
        clarification_needed="I mean the two regions, East and West.",
        stage_plan_approval=_answers({"action": "approve", "run": {"clean": False}}, {"action": "approve"}),
        plan_approval=_answers({"action": "edit", "candidate_predictors": ["customer_id"]},
                               {"action": "edit", "candidate_predictors": ["region"]}, {"action": "approve"}),
        stage_gate=_answers({"action": "run", "predictors": ["region", "customer_id"]}, {"action": "run"}),
    )
    return [p["payload"] for p in pauses]


def test_every_pause_is_handled_by_the_product(payloads):
    emitted = {p["type"] for p in payloads}
    assert len(emitted) == 5, f"the run must reach all five pauses, reached {sorted(emitted)}"
    handled = set(HANDLED_PAUSE_TYPES)
    assert emitted == handled, (f"missing from the product: {sorted(emitted - handled)}; "
                                f"never emitted by the backend: {sorted(handled - emitted)}")
    # the guard's scenarios pause only at types the product handles
    assert {t for pauses in EXPECTED_PAUSES.values() for t in pauses} <= handled


def test_the_run_reaches_the_refused_and_reselected_payloads(payloads):
    by_type = {}
    for p in payloads:
        by_type.setdefault(p["type"], []).append(p)
    assert any(p["refused"] for p in by_type["stage_plan_approval"])
    assert any(p["refused"] for p in by_type["plan_approval"])
    assert any(p["reselected"] for p in by_type["plan_approval"])
    assert any(p["refused"] for p in by_type["stage_gate"])
    assert any(p["stage1_report"] for p in by_type["plan_approval"])


def test_every_payload_has_the_keys_the_product_reads(payloads):
    for p in payloads:
        missing = [k for k in REQUIRED_PAYLOAD_KEYS[p["type"]] if k not in p]
        assert missing == [], f"{p['type']} payload lacks {missing}"


def test_the_required_keys_cover_exactly_the_handled_types():
    assert set(REQUIRED_PAYLOAD_KEYS) == set(HANDLED_PAUSE_TYPES)
    assert len(HANDLED_PAUSE_TYPES) == len(set(HANDLED_PAUSE_TYPES))
    assert all(keys[0] == "type" for keys in REQUIRED_PAYLOAD_KEYS.values())
