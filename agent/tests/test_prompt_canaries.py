"""
M0.5 step 2: prompt canaries. The real classify, draft, cleaning-proposal and report functions run through
the real graph; only the chat model is replaced by a fake that records every prompt. The data holds canary
values (names and exact numbers) that no legitimate summary can produce, and no prompt may contain one.
"""
import json
import re

import numpy as np
import pandas as pd
import pytest
from langgraph.types import Command

import core.classifier as classifier_module
import core.report as report_module
from core.classifier import _ClassificationOut, _DraftPlanOut, _ProposedOperationOut, _WranglingProposalsOut
from core.dataset import LocalDiskStore
from core.graph import build_agent_graph
from core.ingestion import load_dataset
from core.plan import CausalStatus, OutcomeType, QuestionType

NAME_CANARIES = ("CANARY-NAME-7731", "CANARY-NAME-8842", "CANARY-NAME-9953")
SPEND_CANARY = 61803.398875
TENURE_CANARY = 27182.818284
CANARIES = (*NAME_CANARIES, SPEND_CANARY, TENURE_CANARY)
REPORT_TEXT = "Report for the analysis."


# --- fake chat model ---------------------------------------------------------------------------

def _text(messages):
    return "\n".join(content for _, content in messages)


class _Structured:
    def __init__(self, chat, schema):
        self.chat, self.schema = chat, schema

    def invoke(self, messages):
        text = _text(messages)
        self.chat.prompts.append({"stage": self.schema.__name__, "text": text})
        return self.chat.answer(self.schema, text)


class FakeChat:
    def __init__(self, prompts, config):
        self.prompts, self.config = prompts, config

    def with_structured_output(self, schema):
        return _Structured(self, schema)

    def invoke(self, messages):
        self.prompts.append({"stage": "report", "text": _text(messages)})
        return type("Response", (), {"content": REPORT_TEXT})()

    def answer(self, schema, text):
        if schema is _ClassificationOut:
            return _ClassificationOut(question_type=self.config["question_type"],
                                      causal_status=CausalStatus.OBSERVATIONAL, rationale="fake",
                                      alternatives=[], confidence=0.9)
        if schema is _DraftPlanOut:
            return _DraftPlanOut(outcome_name="total_spend", outcome_type=OutcomeType.CONTINUOUS,
                                 unit_of_analysis="customer", candidate_predictors=list(self.config["predictors"]),
                                 rationale="fake")
        if schema is _WranglingProposalsOut:
            proposals = []
            for index, issue_type, columns in re.findall(r"^\[(\d+)\] (\w+):.*\(columns: ([^,)]*)", text, re.M):
                if issue_type == "duplicate_rows":
                    op, params = "dedupe", {"subset": []}
                elif issue_type == "high_missingness":
                    op, params = "filter", {"column": columns.strip(), "condition": "not_null", "value": None}
                else:
                    raise AssertionError(f"the fake has no proposal for issue type {issue_type!r}")
                proposals.append(_ProposedOperationOut(
                    issue_index=int(index), op_type=op, params=params, rationale="fake",
                    alternative_approach="leave the data as is", alternative_rejected_because="fake"))
            return _WranglingProposalsOut(proposals=proposals)
        raise AssertionError(f"unexpected schema {schema.__name__}")


@pytest.fixture
def fake_model(monkeypatch):
    """Returns (prompts, set_config): every prompt sent to any model lands in prompts."""
    prompts, config = [], {}

    def factory(*args, **kwargs):
        return FakeChat(prompts, config)
    monkeypatch.setattr(classifier_module, "init_chat_model", factory)
    monkeypatch.setattr(report_module, "init_chat_model", factory)
    return prompts, config


# --- canary data -------------------------------------------------------------------------------

def _base_df():
    rng = np.random.default_rng(20261014)
    n = 360
    names = list(NAME_CANARIES) + [f"name-{i:04d}" for i in range(1, 298)]  # 300 distinct
    names += names[:60]  # 60 repeats: the canaries appear twice, as do the next 57 names
    spend = np.clip(rng.normal(50000, 15000, n), 20000, 80000)
    tenure = np.clip(rng.normal(25000, 8000, n), 5000, 45000)
    spend[5], tenure[7] = SPEND_CANARY, TENURE_CANARY
    notes = np.where(rng.random(n) < 0.12, None, rng.choice(["ok", "late", "vip"], n))
    df = pd.DataFrame({
        "customer_name": names,
        "region": rng.choice(["East", "West"], n),
        "tier": rng.choice(["basic", "plus", "pro"], n),
        "tenure": tenure,
        "total_spend": spend,
        "notes": notes,
    })
    return df


def _with_duplicates(df):
    return pd.concat([df, df.iloc[[100, 101, 102, 103]]], ignore_index=True)  # 4 exact duplicate rows


def _clean(df):
    return df.drop(columns=["notes"])  # no missingness and no duplicates: no cleaning pause


def _variants(canary):
    if isinstance(canary, str):
        return [canary]
    return [f"{canary:.2f}", f"{canary:.4f}", f"{canary:.6f}", repr(canary), str(canary)]


def canary_hits(prompts, canaries=CANARIES):
    """(stage, canary) for every prompt that contains any variant of a canary."""
    return [(p["stage"], c) for p in prompts for c in canaries if any(v in p["text"] for v in _variants(c))]


def test_canaries_cannot_collide_with_a_legitimate_statistic():
    df = _base_df()
    for column, canary in (("total_spend", SPEND_CANARY), ("tenure", TENURE_CANARY)):
        s = df[column]
        assert (s == canary).sum() == 1, column
        stats = [s.min(), s.max(), s.mean(), s.median(), s.std()]
        for fmt in (lambda x: f"{x:.2f}", lambda x: f"{x:.4f}", lambda x: f"{x:.6f}", repr, str):
            assert fmt(canary) not in {fmt(float(v)) for v in stats}, column
        prefix = f"{canary:.2f}"[:7]
        assert not any(f"{float(v):.2f}".startswith(prefix) for v in stats), column


def test_canary_names_are_among_the_most_frequent_values():
    counts = _base_df()["customer_name"].value_counts()
    assert counts.index[:3].tolist() == list(NAME_CANARIES)
    assert 290 <= counts.size <= 310


def test_canary_hits_detects_every_variant_and_nothing_else():
    for canary in CANARIES:
        for variant in _variants(canary):
            assert canary_hits([{"stage": "s", "text": f"x {variant} y"}], [canary]) == [("s", canary)]
    clean = {"stage": "s", "text": "total_spend mean=50012.34, East (180), name-0001"}
    assert canary_hits([clean]) == []


# --- scenarios through the real graph ------------------------------------------------------------

def _drive(app, state, thread):
    """Run to the end, approving every cleaning operation and the plan."""
    config = {"configurable": {"thread_id": thread}}
    pauses = []
    graph_input = state
    while True:
        app.invoke(graph_input, config)
        snapshot = app.get_state(config)
        if not snapshot.interrupts:
            return snapshot.values, pauses
        payload = snapshot.interrupts[0].value
        pauses.append(payload["type"])
        if payload["type"] == "wrangling_approval":
            graph_input = Command(resume={"decisions": [{"index": op["index"], "action": "approve"}
                                                        for op in payload["proposed_operations"]]})
        elif payload["type"] in ("plan_approval", "stage_plan_approval"):  # M1.3a: the stage-plan gate
            graph_input = Command(resume={"action": "approve"})
        elif payload["type"] == "stage_gate":  # M1.3b: skip the exploratory round by default
            graph_input = Command(resume={"action": "skip"})
        else:
            raise AssertionError(f"unexpected pause {payload['type']}")


SCENARIOS = {
    # M1.3a: comparison and driver runs pause at the stage-plan gate first; M1.3b: comparisons end at the boundary gate
    "S1_two_group_with_cleaning": (QuestionType.TWO_GROUP_COMPARISON, ["region"], True,
                                   ["stage_plan_approval", "wrangling_approval", "plan_approval", "stage_gate"]),
    "S2_multi_group": (QuestionType.MULTI_GROUP_COMPARISON, ["tier"], False,
                       ["stage_plan_approval", "plan_approval", "stage_gate"]),
    "S3_regression": (QuestionType.DRIVER_ANALYSIS, ["tenure", "tier"], False, ["stage_plan_approval", "plan_approval"]),
    "S4_descriptive": (QuestionType.DESCRIPTIVE, [], False, []),
}


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_no_canary_reaches_any_prompt(name, fake_model, tmp_path):
    question_type, predictors, messy, expected_pauses = SCENARIOS[name]
    prompts, config = fake_model
    config.update(question_type=question_type, predictors=predictors)

    store = LocalDiskStore(base_dir=tmp_path / "store")
    df = _with_duplicates(_base_df()) if messy else _clean(_base_df())
    path = tmp_path / f"{name}.csv"
    df.to_csv(path, index=False)
    dataset = load_dataset(path, session_id="s", name=name, store=store)

    values, pauses = _drive(build_agent_graph(store),
                            {"request_text": "How does total spend look?", "session_id": "s", "dataset": dataset},
                            name)

    assert pauses == expected_pauses
    assert values.get("report") == REPORT_TEXT
    stages = [p["stage"] for p in prompts]
    expected = {"_ClassificationOut", "report"} | ({"_WranglingProposalsOut"} if messy else set())
    expected |= set() if question_type is QuestionType.DESCRIPTIVE else {"_DraftPlanOut"}
    assert set(stages) == expected, stages
    joined = "\n".join(p["text"] for p in prompts)
    assert "total_spend" in joined and any(label in joined for label in ("East", "basic"))

    hits = canary_hits(prompts)
    detail = []
    for stage, canary in hits:
        text = next(p["text"] for p in prompts if p["stage"] == stage
                    and any(v in p["text"] for v in _variants(canary)))
        lines = [ln for ln in text.splitlines() if any(v in ln for v in _variants(canary))]
        detail.append(f"{stage}: {canary!r} in line {lines[0]!r}")
    assert hits == [], "canaries reached a prompt:\n" + "\n".join(detail)


def test_descriptive_run_state_holds_no_name_canary(fake_model, tmp_path):
    """The final state of the real descriptive run (S4): no name canary anywhere in descriptive_result,
    charts included."""
    _, config = fake_model
    config.update(question_type=QuestionType.DESCRIPTIVE, predictors=[])
    store = LocalDiskStore(base_dir=tmp_path / "store")
    path = tmp_path / "s4.csv"
    _clean(_base_df()).to_csv(path, index=False)
    dataset = load_dataset(path, session_id="s", name="s4", store=store)
    values, _ = _drive(build_agent_graph(store),
                       {"request_text": "Describe this data.", "session_id": "s", "dataset": dataset}, "s4")

    result = values["descriptive_result"]
    assert result["chart_data"] and values.get("report") == REPORT_TEXT
    text = json.dumps(result, default=str)
    assert [c for c in NAME_CANARIES if c in text] == []
