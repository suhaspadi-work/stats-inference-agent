"""
UI tests of the pause screens (M1.5 step P2b), run with the product venv from product/: python -m pytest ui_tests.
The screens are rendered headless with Streamlit's AppTest (no browser, no model). The payloads are REAL: one run
of the real graph with fake models, the same run as the backend's contract test (agent/tests/test_product_contract
.py), which reaches all five pauses with refusals. This folder is named ui_tests so that the agent's own "tests"
package stays importable.
"""
import sys
from pathlib import Path

import pytest

PRODUCT = Path(__file__).resolve().parents[1]
AGENT = PRODUCT.parent / "agent"
for path in (str(PRODUCT), str(AGENT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from langgraph.types import Command  # noqa: E402

import core.classifier as classifier_module  # noqa: E402
import core.report as report_module  # noqa: E402
from core.dataset import LocalDiskStore  # noqa: E402
from core.graph import build_agent_graph  # noqa: E402
from core.plan import QuestionType  # noqa: E402
from tests.test_analysis_reports import BAD, _answers, _three_groups, _with_id  # noqa: E402
from tests.test_decision_point_guard import (  # noqa: E402
    _dataset, confident, fake_classify, fake_draft, fake_propose, must_not_propose,
)
from tests.test_exploratory_round import _frame  # noqa: E402
from tests.test_product_contract import _messy  # noqa: E402
from tests.test_prompt_canaries import FakeChat  # noqa: E402
from tests.test_stage1_report import _answer, _drive  # noqa: E402

CANARY = "CANARY-CODE"  # the values of a withheld (high-cardinality) text column; they must never be on a page


def _run_recording_reports(app, graph_input, thread, **overrides):
    """Like the contract run's driver, but each pause also keeps the full state values at that moment (the stage
    gate shows the main analysis's report, plan approval the describe result)."""
    config = {"configurable": {"thread_id": thread}}
    pauses = []
    while True:
        for _ in app.stream(graph_input, config, stream_mode="updates"):
            pass
        snapshot = app.get_state(config)
        if not snapshot.interrupts:
            return pauses
        payload = snapshot.interrupts[0].value
        pauses.append({"payload": payload, "values": dict(snapshot.values)})
        graph_input = Command(resume=_answer(payload, overrides))


def _refuse(*args, **kwargs):
    raise RuntimeError("a UI test tried to call the real model")


def _fake_models(mp, report_chat=None):
    mp.setattr(classifier_module, "init_chat_model", _refuse)
    chat = report_chat or FakeChat([], {})
    mp.setattr(report_module, "init_chat_model", lambda *a, **k: chat)


class _AlwaysCausal:
    """A report model whose every text uses causal language, so the narrative falls back to the computed facts."""

    def invoke(self, messages):
        return type("Response", (), {"content": BAD})()


def _with_code(n=120, seed=51):
    df = _frame(seed=seed, n=n)
    df["customer_code"] = [f"{CANARY}-{i:04d}" for i in range(n)]
    return df


@pytest.fixture(scope="session")
def real_finished(tmp_path_factory):
    """REAL final states (graph values) of finished runs, keyed two_group, exploratory, fallback, descriptive and
    multi_group; the real graph with fake models."""
    tmp = tmp_path_factory.mktemp("ui_finished")
    store = LocalDiskStore(base_dir=tmp / "store")

    def run(name, df, question_type, predictors, **overrides):
        app = build_agent_graph(store, classify_fn=confident(question_type), draft_fn=fake_draft(list(predictors)),
                                propose_fn=must_not_propose)
        state = {"request_text": "Did West spend more?", "session_id": "session-abc",
                 "dataset": _dataset(df, tmp, store, name)}
        values, _, _ = _drive(app, state, f"ui-{name}", **overrides)
        return values

    states = {}
    with pytest.MonkeyPatch.context() as mp:
        _fake_models(mp)
        states["two_group"] = run("two_group", _with_code(), QuestionType.TWO_GROUP_COMPARISON, ["region"])
        states["exploratory"] = run("exploratory", _with_id(), QuestionType.TWO_GROUP_COMPARISON, ["region"],
                                    stage_gate={"action": "run"})
        states["descriptive"] = run("descriptive", _frame(), QuestionType.DESCRIPTIVE, [])
        states["multi_group"] = run("multi_group", _three_groups((50, 60, 70), 81),
                                    QuestionType.MULTI_GROUP_COMPARISON, ["region"])
    with pytest.MonkeyPatch.context() as mp:
        _fake_models(mp, _AlwaysCausal())
        states["fallback"] = run("fallback", _frame(), QuestionType.TWO_GROUP_COMPARISON, ["region"])
    return states


@pytest.fixture(scope="session")
def real_pauses(tmp_path_factory):
    """Every pause of the contract run, as {"payload", "values"}; the same answers as the contract fixture."""
    tmp = tmp_path_factory.mktemp("ui_contract")
    with pytest.MonkeyPatch.context() as mp:
        _fake_models(mp)
        store = LocalDiskStore(base_dir=tmp / "store")
        app = build_agent_graph(store, classify_fn=fake_classify(QuestionType.TWO_GROUP_COMPARISON),
                                draft_fn=fake_draft(["region"]), propose_fn=fake_propose)
        state = {"request_text": "Did West spend more?", "session_id": "session-abc",
                 "dataset": _dataset(_messy(), tmp, store, "contract")}
        return _run_recording_reports(
            app, state, "ui-contract",
            clarification_needed="I mean the two regions, East and West.",
            stage_plan_approval=_answers({"action": "approve", "run": {"clean": False}}, {"action": "approve"}),
            plan_approval=_answers({"action": "edit", "candidate_predictors": ["customer_id"]},
                                   {"action": "edit", "candidate_predictors": ["region"]}, {"action": "approve"}),
            stage_gate=_answers({"action": "run", "predictors": ["region", "customer_id"]}, {"action": "run"}),
        )
