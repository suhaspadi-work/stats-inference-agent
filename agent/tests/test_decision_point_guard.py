"""
M0.2 step 12: the decision-point guard.

Static checks hold the registry in core/decision_points.py to the real graph,
the record modules and the capability registry. Scenario runs drive the real
graph (with fake model functions, stopping before the report node) through
seven paths and check that, together, they emit exactly the registered decision
points, with valid internal links, JSON-safe logs and only allowed charts.

The check logic lives in small helpers so the negative tests can prove each
check fails when it should.
"""
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from langgraph.types import Command

import core.classification_records as classification_records
import core.cleaning_records as cleaning_records
import core.describe_records as describe_records
import core.draft_plan_records as draft_plan_records
import core.execution_records as execution_records
import core.freeze_records as freeze_records
import core.round_records as round_records
import core.routing_records as routing_records
import core.selection_records as selection_records
import core.stage_plan_records as stage_plan_records
from core.capabilities import ANALYSES
from core.classifier import _DraftPlanOut, _ProposedOperationOut
from core.dataset import LocalDiskStore
from core.decision_points import (
    ALLOWED_RECORD_CHART_TYPES, ANALYSES_WITHOUT_EXECUTION_RECORD, DECISION_NODES, NON_CHOOSING_NODES,
    registered_points,
)
from core.graph import build_agent_graph
from core.ingestion import load_dataset
from core.plan import CausalStatus, ClassificationDecision, OutcomeType, QuestionType

RECORD_MODULES = (classification_records, cleaning_records, describe_records, draft_plan_records, execution_records,
                  freeze_records, round_records, routing_records, selection_records, stage_plan_records)
POINT_PREFIXES = ("intake.", "planner.", "stage1.", "stage2.", "stage3.")


# ------------------------------------------------------------------ check helpers

def undeclared_nodes(graph_node_names) -> list[str]:
    """Problems with the node registry; empty when every node is declared exactly once."""
    nodes = set(graph_node_names) - {"__start__", "__end__"}
    declared = set(DECISION_NODES) | set(NON_CHOOSING_NODES)
    problems = [f"node '{n}' is in the graph but in neither DECISION_NODES nor NON_CHOOSING_NODES"
                for n in sorted(nodes - declared)]
    problems += [f"node '{n}' is declared but not in the graph" for n in sorted(declared - nodes)]
    problems += [f"node '{n}' is in both DECISION_NODES and NON_CHOOSING_NODES"
                 for n in sorted(set(DECISION_NODES) & set(NON_CHOOSING_NODES))]
    return problems


def module_points(modules=RECORD_MODULES) -> set[str]:
    """Every *_POINT string and every value of a *_POINTS dict (such as STEP_POINTS) in the record modules."""
    points = set()
    for module in modules:
        for name, value in vars(module).items():
            if name.endswith("_POINT") and isinstance(value, str):
                points.add(value)
            elif name.endswith("_POINTS") and isinstance(value, dict):
                points.update(v for v in value.values() if isinstance(v, str))
    return points


def unregistered_constants(points, registered) -> set[str]:
    return set(points) - set(registered)


def unregistered_points(log, registered=None) -> set[str]:
    registered = registered_points() if registered is None else registered
    return {e["decision_point"] for e in log} - set(registered)


def never_emitted(registered, emitted) -> set[str]:
    return set(registered) - set(emitted)


def broken_links(log) -> list[str]:
    """Duplicate ids, and supersedes or *record_id values that do not refer to an earlier record."""
    problems, seen = [], set()
    for i, entry in enumerate(log):
        if entry["id"] in seen:
            problems.append(f"record {i} ({entry['decision_point']}) repeats id {entry['id']}")
        refs = [("supersedes", entry.get("supersedes"))]
        refs += [(k, v) for k, v in (entry.get("evidence") or {}).items() if k.endswith("record_id")]
        for key, ref in refs:
            if ref is not None and ref not in seen:
                problems.append(f"record {i} ({entry['decision_point']}) {key}={ref} is not an earlier record")
        seen.add(entry["id"])
    return problems


def wrong_node_emissions(emitted_by_node, decision_nodes, non_choosing) -> list[str]:
    """Nodes that emitted points they are not declared for, choosing nodes that never emitted a declared
    point, and non-choosing nodes that emitted anything."""
    problems = []
    for node, points in sorted(emitted_by_node.items()):
        points = set(points)
        if node in non_choosing:
            if points:
                problems.append(f"non-choosing node '{node}' emitted {sorted(points)}")
        elif node in decision_nodes:
            extra = points - set(decision_nodes[node])
            if extra:
                problems.append(f"node '{node}' emitted {sorted(extra)}, which it is not declared for")
        elif points:
            problems.append(f"undeclared node '{node}' emitted {sorted(points)}")
    for node, declared in sorted(decision_nodes.items()):
        silent = set(declared) - set(emitted_by_node.get(node, ()))
        if silent:
            problems.append(f"node '{node}' is declared for {sorted(silent)} but never emitted it")
    return problems


def find_disallowed_charts(log, allowed=ALLOWED_RECORD_CHART_TYPES) -> list[str]:
    return [f"{e['decision_point']} carries a '{c.get('chart_type')}' chart"
            for e in log for c in (e.get("chart_data") or []) if c.get("chart_type") not in allowed]


# ------------------------------------------------------------------ static checks

def test_every_graph_node_is_declared_exactly_once(tmp_path):
    names = build_agent_graph(LocalDiskStore(base_dir=tmp_path)).get_graph().nodes
    assert undeclared_nodes(names) == []


def test_undeclared_node_is_reported(tmp_path):
    names = set(build_agent_graph(LocalDiskStore(base_dir=tmp_path)).get_graph().nodes) | {"mystery"}
    problems = undeclared_nodes(names)
    assert len(problems) == 1 and "'mystery'" in problems[0]


def test_every_point_constant_is_registered():
    points = module_points()
    assert unregistered_constants(points, registered_points()) == set()


def test_made_up_point_constant_is_reported():
    assert unregistered_constants(module_points() | {"stage2.made_up"}, registered_points()) == {"stage2.made_up"}


def test_made_up_point_in_a_points_dict_is_reported():
    assert set(stage_plan_records.STEP_POINTS.values()) <= module_points()
    fake = SimpleNamespace(STEP_POINTS={**stage_plan_records.STEP_POINTS, "forecast": "planner.step.forecast"})
    assert unregistered_constants(module_points((fake,)), registered_points()) == {"planner.step.forecast"}


def test_every_analysis_has_its_record_builders():
    names = {a.name for a in ANALYSES}
    for analysis in ANALYSES:
        assert analysis.record is not None, analysis.name
        assert analysis.execution_records is not None or analysis.name in ANALYSES_WITHOUT_EXECUTION_RECORD, \
            analysis.name
    for name in ANALYSES_WITHOUT_EXECUTION_RECORD:
        assert name in names, f"stale exemption: '{name}' is not an analysis"
        assert next(a for a in ANALYSES if a.name == name).execution_records is None, \
            f"stale exemption: '{name}' now has an execution_records builder"


def test_every_point_uses_a_known_prefix():
    assert [p for p in registered_points() if not p.startswith(POINT_PREFIXES)] == []


def test_default_graph_has_the_same_nodes(tmp_path):
    nodes = set(build_agent_graph(LocalDiskStore(base_dir=tmp_path)).get_graph().nodes) - {"__start__", "__end__"}
    # M1.1: plan_stages; M1.2: describe; M1.3a: stage_plan_gate; M1.3b: stage_gate and start_round after report
    assert nodes == {"classify", "clarify", "reclassify", "profile", "plan_stages", "stage_plan_gate", "wrangle_propose",
                     "wrangle_review", "describe", "draft_plan", "eda", "select_method", "freeze", "execute", "report",
                     "stage_gate", "start_round"}


# ------------------------------------------------------------------ negative tests

def _entry(point, i, **extra):
    return {"decision_point": point, "id": f"id-{i}", "supersedes": None, "evidence": {}, "chart_data": [], **extra}


def test_unregistered_point_in_a_log_is_reported():
    log = [_entry("intake.question_type", 1), _entry("stage2.made_up", 2)]
    assert unregistered_points(log) == {"stage2.made_up"}


def test_never_emitted_point_is_reported():
    assert never_emitted({"a", "b", "c"}, {"a", "c"}) == {"b"}


def test_broken_links_are_reported():
    log = [_entry("a", 1), _entry("b", 2, supersedes="id-9"),
           _entry("c", 3, evidence={"fix_record_id": "id-4"}), _entry("d", 1)]
    problems = broken_links(log)
    assert len(problems) == 3
    assert any("supersedes=id-9" in p for p in problems)
    assert any("fix_record_id=id-4" in p for p in problems)
    assert any("repeats id id-1" in p for p in problems)


DECLARED = {"classify": ("intake.question_type", "intake.causal_status"), "execute": ("stage2.post_hoc",)}
NON_CHOOSING = {"profile": "computes the profile"}


def test_node_emitting_an_undeclared_point_is_reported():
    emitted = {"classify": {"intake.question_type", "intake.causal_status", "stage2.post_hoc"},
               "execute": {"stage2.post_hoc"}}
    assert wrong_node_emissions(emitted, DECLARED, NON_CHOOSING) == [
        "node 'classify' emitted ['stage2.post_hoc'], which it is not declared for"]


def test_choosing_node_with_no_emissions_is_reported():
    emitted = {"classify": {"intake.question_type", "intake.causal_status"}}
    assert wrong_node_emissions(emitted, DECLARED, NON_CHOOSING) == [
        "node 'execute' is declared for ['stage2.post_hoc'] but never emitted it"]


def test_choosing_node_emitting_part_of_its_tuple_is_reported():
    emitted = {"classify": {"intake.question_type"}, "execute": {"stage2.post_hoc"}}
    assert wrong_node_emissions(emitted, DECLARED, NON_CHOOSING) == [
        "node 'classify' is declared for ['intake.causal_status'] but never emitted it"]


def test_non_choosing_node_emitting_a_point_is_reported():
    emitted = {"classify": {"intake.question_type", "intake.causal_status"}, "execute": {"stage2.post_hoc"},
               "profile": {"stage1.clean.scan"}}
    assert wrong_node_emissions(emitted, DECLARED, NON_CHOOSING) == [
        "non-choosing node 'profile' emitted ['stage1.clean.scan']"]


def test_disallowed_chart_is_reported():
    log = [_entry("stage2.post_hoc", 1, chart_data=[{"chart_type": "post_hoc_comparison_chart"}]),
           _entry("stage2.test_selection", 2, chart_data=[{"chart_type": "boxplot_by_group"}])]
    assert find_disallowed_charts(log) == ["stage2.test_selection carries a 'boxplot_by_group' chart"]


# ------------------------------------------------------------------ scenario fakes

def fake_classify(question_type):
    def classify(request_text):
        clarified = "Clarification:" in request_text
        return ClassificationDecision(
            question_type=question_type, causal_status=CausalStatus.OBSERVATIONAL,
            rationale="fake classification", alternatives=(), confidence=0.9 if clarified else 0.3, evidence={},
        )
    return classify


def confident(question_type):
    def classify(request_text):
        return ClassificationDecision(
            question_type=question_type, causal_status=CausalStatus.OBSERVATIONAL,
            rationale="fake classification", alternatives=(), confidence=0.9, evidence={},
        )
    return classify


def fake_draft(predictors, outcome="total_spend"):
    def draft(request_text, profile):
        return _DraftPlanOut(outcome_name=outcome, outcome_type=OutcomeType.CONTINUOUS, unit_of_analysis="customer",
                             candidate_predictors=list(predictors), rationale="fake draft")
    return draft


def fake_propose(issues):
    proposals = []
    for i, issue in enumerate(issues):
        if issue.issue_type == "duplicate_rows":
            op, params = "dedupe", {"subset": []}
        else:
            op, params = "filter", {"column": issue.affected_columns[0], "condition": "not_null", "value": None}
        proposals.append(_ProposedOperationOut(
            issue_index=i, op_type=op, params=params, rationale=f"fix {issue.issue_type}",
            alternative_approach="leave the data as is", alternative_rejected_because="it would bias the result",
        ))
    return proposals


def must_not_propose(issues):
    raise AssertionError("no issues were expected, so the proposer must not be called")


# ------------------------------------------------------------------ datasets

def _dataset(df, tmp_path, store, name):
    path = tmp_path / f"{name}.csv"
    df.to_csv(path, index=False)
    return load_dataset(path, session_id="session-abc", name=name, store=store)


def _two_group_df(rng, n=60):
    return pd.DataFrame({
        "customer_id": range(1, 2 * n + 1),
        "region": ["East"] * n + ["West"] * n,
        "channel": (["web", "store"] * n),
        "total_spend": np.concatenate([rng.normal(50, 10, n), rng.normal(56, 10, n)]),
    })


def _with_duplicate(df):
    return pd.concat([df, df.iloc[[0]]], ignore_index=True)


# ------------------------------------------------------------------ driver

def _drive(app, state, thread, resumes, emitted_by_node, gate=None, stage_gate=None, stop=("execute", "eda")):
    """
    Run to the end or to the stop after execute/eda, resuming each pause with the next value in order.
    Streams in "updates" mode so each node's own records are seen: an event {node: update} whose update
    holds "decision_log" is that node's records, merged into emitted_by_node. A dynamic pause arrives as a
    non-empty "__interrupt__" tuple; the static stop after execute/eda arrives as an empty one.
    M1.3a: a stage_plan_approval pause is answered with gate (default {"action": "approve"}) without
    using a value from resumes; it is still recorded in the returned pauses. M1.3b: a stage_gate pause is
    answered the same way with stage_gate (default {"action": "skip"}); stop=() runs past the report.
    """
    config = {"configurable": {"thread_id": thread}}
    stop = list(stop)

    def run(graph_input):
        pause = None
        for event in app.stream(graph_input, config, stream_mode="updates", interrupt_after=stop):
            for node, update in event.items():
                if node == "__interrupt__":
                    if update:
                        pause = update[0].value["type"]
                elif isinstance(update, dict) and update.get("decision_log"):
                    emitted_by_node.setdefault(node, set()).update(e["decision_point"] for e in update["decision_log"])
        return pause

    gate = {"action": "approve"} if gate is None else gate
    stage_gate = {"action": "skip"} if stage_gate is None else stage_gate
    queue, pauses = list(resumes), []
    pause = run(state)
    while pause is not None:
        pauses.append(pause)
        if pause == "stage_plan_approval":
            pause = run(Command(resume=gate))
            continue
        if pause == "stage_gate":
            pause = run(Command(resume=stage_gate))
            continue
        assert queue, f"paused at {pause} with no resume value left"
        pause = run(Command(resume=queue.pop(0)))
    assert queue == [], f"unused resume values: {queue}"
    snapshot = app.get_state(config)
    return snapshot.values, pauses, snapshot.next


@pytest.fixture(scope="module")
def scenario_runs(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("guard")
    store = LocalDiskStore(base_dir=tmp / "store")
    rng = np.random.default_rng(7)
    runs, emitted = {}, {}

    def drive(app, state, thread, resumes, **kwargs):
        return _drive(app, state, thread, resumes, emitted, **kwargs)

    def start(ds, request, **extra):
        return {"request_text": request, "session_id": "session-abc", "dataset": ds, **extra}

    # S1: two-group, duplicate plus missingness, low-confidence clarification, approve both fixes and the plan.
    df = _two_group_df(rng)
    df["notes"] = [None if i % 6 == 0 else "ok" for i in range(len(df))]  # about 17% missing
    app = build_agent_graph(store, classify_fn=fake_classify(QuestionType.TWO_GROUP_COMPARISON),
                            draft_fn=fake_draft(["region"]), propose_fn=fake_propose)
    runs["S1"] = drive(app, start(_dataset(_with_duplicate(df), tmp, store, "s1"), "Did West spend more?"), "s1", [
        "I mean the two regions, East and West.",
        {"decisions": [{"index": 0, "action": "approve"}, {"index": 1, "action": "approve"}]},
        {"action": "approve"},
    ])

    # S2: two-group, no issues, confident, the plan is edited to the other grouping column.
    app = build_agent_graph(store, classify_fn=confident(QuestionType.TWO_GROUP_COMPARISON),
                            draft_fn=fake_draft(["region"]), propose_fn=must_not_propose)
    runs["S2"] = drive(app, start(_dataset(_two_group_df(rng), tmp, store, "s2"), "Region or channel?"), "s2", [
        {"action": "edit", "candidate_predictors": ["channel"]},
        {"action": "approve"},  # M1.3c: an accepted edit re-selects the method and pauses at plan approval again
    ])

    # S3: two-group, the plan is rejected and the run ends at the freeze.
    app = build_agent_graph(store, classify_fn=confident(QuestionType.TWO_GROUP_COMPARISON),
                            draft_fn=fake_draft(["region"]), propose_fn=must_not_propose)
    runs["S3"] = drive(app, start(_dataset(_two_group_df(rng), tmp, store, "s3"), "Did West spend more?"), "s3", [
        {"action": "reject", "reason": "wrong outcome"},
    ])

    # S4: stakeholder mode, duplicates only: the dedupe is auto-approved, then the plan is approved.
    app = build_agent_graph(store, classify_fn=confident(QuestionType.TWO_GROUP_COMPARISON),
                            draft_fn=fake_draft(["region"]), propose_fn=fake_propose)
    ds = _dataset(_with_duplicate(_two_group_df(rng)), tmp, store, "s4")
    runs["S4"] = drive(app, start(ds, "Did West spend more?", autonomy_mode="stakeholder"), "s4",
                       [{"action": "approve"}])

    # S5: multi-group with clearly different means, so the omnibus test is significant.
    mg = pd.DataFrame({"region": [g for g in ("East", "North", "West") for _ in range(60)],
                       "total_spend": np.concatenate([rng.normal(m, 5, 60) for m in (50, 55, 65)])})
    app = build_agent_graph(store, classify_fn=confident(QuestionType.MULTI_GROUP_COMPARISON),
                            draft_fn=fake_draft(["region"]), propose_fn=must_not_propose)
    runs["S5"] = drive(app, start(_dataset(mg, tmp, store, "s5"), "Does spend differ by region?"), "s5",
                       [{"action": "approve"}])

    # S6: regression with a numeric predictor and a 3-level text predictor.
    n = 200
    tier = rng.choice(["basic", "plus", "pro"], n)
    tenure = rng.normal(24, 8, n)
    spend = 20 + 2.0 * tenure + np.array([{"basic": 0, "plus": 5, "pro": 12}[t] for t in tier]) + rng.normal(0, 5, n)
    reg = pd.DataFrame({"tenure": tenure, "tier": tier, "total_spend": spend})
    app = build_agent_graph(store, classify_fn=confident(QuestionType.DRIVER_ANALYSIS),
                            draft_fn=fake_draft(["tenure", "tier"]), propose_fn=must_not_propose)
    runs["S6"] = drive(app, start(_dataset(reg, tmp, store, "s6"), "What drives spend?"), "s6",
                       [{"action": "approve"}])

    # S7: descriptive, stops after eda.
    app = build_agent_graph(store, classify_fn=confident(QuestionType.DESCRIPTIVE),
                            draft_fn=fake_draft([]), propose_fn=must_not_propose)
    runs["S7"] = drive(app, start(_dataset(_two_group_df(rng), tmp, store, "s7"), "Describe this data."), "s7", [])

    # S8: association with two predictors is unsupported and ends at the stage plan, before cleaning (M1.1).
    app = build_agent_graph(store, classify_fn=confident(QuestionType.ASSOCIATION),
                            draft_fn=fake_draft(["region", "channel"]), propose_fn=must_not_propose)
    runs["S8"] = drive(app, start(_dataset(_two_group_df(rng), tmp, store, "s8"), "Are region and channel related to spend?"), "s8", [])

    # S9: a two-group question on a 3-level column is reconciled to multi-group (M0.3).
    app = build_agent_graph(store, classify_fn=confident(QuestionType.TWO_GROUP_COMPARISON),
                            draft_fn=fake_draft(["region"]), propose_fn=must_not_propose)
    runs["S9"] = drive(app, start(_dataset(mg, tmp, store, "s9"), "Did regions differ?"), "s9", [{"action": "approve"}])

    # S10: a two-group question with a 0/1 outcome passes the stage plan and ends at the draft_plan route
    # (unsupported, G44), so planner.route's unsupported path stays covered now that S8 ends earlier (M1.1).
    binary = _two_group_df(rng)
    binary["converted"] = [i % 2 for i in range(len(binary))]

    def binary_draft(request_text, profile):
        return _DraftPlanOut(outcome_name="converted", outcome_type=OutcomeType.BINARY, unit_of_analysis="customer",
                             candidate_predictors=["region"], rationale="fake draft")
    app = build_agent_graph(store, classify_fn=confident(QuestionType.TWO_GROUP_COMPARISON),
                            draft_fn=binary_draft, propose_fn=must_not_propose)
    runs["S10"] = drive(app, start(_dataset(binary, tmp, store, "s10"), "Did West convert more?"), "s10", [])

    # S11: two-group run to the end; the boundary gate is answered "run", so an exploratory regression follows
    # (M1.3b). The reports need a chat model: the fixture is module-scoped, so the conftest guard is not active
    # yet, and the fake is patched in here.
    from tests.test_prompt_canaries import FakeChat
    import core.report as report_module
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(report_module, "init_chat_model", lambda *a, **k: FakeChat([], {}))
        app = build_agent_graph(store, classify_fn=confident(QuestionType.TWO_GROUP_COMPARISON),
                                draft_fn=fake_draft(["region"]), propose_fn=must_not_propose)
        runs["S11"] = drive(app, start(_dataset(_two_group_df(rng), tmp, store, "s11"), "Did West spend more?"), "s11",
                            [{"action": "approve"}, {"action": "approve"}], stage_gate={"action": "run"}, stop=())

    return runs, emitted


@pytest.fixture(scope="module")
def scenario_logs(scenario_runs):
    return scenario_runs[0]


@pytest.fixture(scope="module")
def emitted_by_node(scenario_runs):
    return scenario_runs[1]


# ------------------------------------------------------------------ scenario checks

GATE = "stage_plan_approval"  # M1.3a: every comparison and driver run pauses at the stage-plan gate first
EXPECTED_PAUSES = {
    "S1": ["clarification_needed", GATE, "wrangling_approval", "plan_approval"],
    "S2": [GATE, "plan_approval", "plan_approval"],  # M1.3c: the accepted edit re-selects and pauses again
    "S3": [GATE, "plan_approval"], "S4": [GATE, "plan_approval"],
    "S5": [GATE, "plan_approval"], "S6": [GATE, "plan_approval"], "S7": [], "S8": [], "S9": [GATE, "plan_approval"],
    "S10": [GATE],
    "S11": [GATE, "plan_approval", "stage_gate", "plan_approval"],  # M1.3b: the exploratory round
}
ENDS_WITHOUT_STOP = {"S3", "S8", "S10", "S11"}  # runs that end at END rather than at the stop before report


def test_each_scenario_pauses_in_order_and_stops_where_expected(scenario_logs):
    for name, (result, pauses, next_nodes) in scenario_logs.items():
        assert pauses == EXPECTED_PAUSES[name], name
        # M0.3: a run that ends before the stop (rejected plan, unsupported route) has next ()
        assert next_nodes == (() if name in ENDS_WITHOUT_STOP else ("report",)), name
    assert scenario_logs["S3"][0]["plan"].status.value != "frozen"
    assert scenario_logs["S7"][0]["plan"].question_type.value == "descriptive"


def test_scenarios_emit_exactly_the_registered_points(scenario_logs):
    emitted = {e["decision_point"] for result, _, _ in scenario_logs.values() for e in result["decision_log"]}
    unregistered = {p for result, _, _ in scenario_logs.values() for p in unregistered_points(result["decision_log"])}
    missing = never_emitted(registered_points(), emitted)
    assert not unregistered and not missing, (
        f"unregistered points emitted: {sorted(unregistered)}; registered points never emitted: {sorted(missing)}")


def test_each_scenario_log_links_and_serializes(scenario_logs):
    for name, (result, _, _) in scenario_logs.items():
        log = result["decision_log"]
        assert broken_links(log) == [], name
        json.dumps(log)
        assert find_disallowed_charts(log) == [], name


def test_only_points_declared_for_some_node_appear(scenario_logs):
    for name, (result, _, _) in scenario_logs.items():
        assert unregistered_points(result["decision_log"]) == set(), name


def test_each_node_emits_exactly_its_declared_points(emitted_by_node):
    problems = wrong_node_emissions(emitted_by_node, DECISION_NODES, NON_CHOOSING_NODES)
    assert problems == [], "per-node emission problems:\n" + "\n".join(problems)
    assert "planner.predictors" in emitted_by_node["freeze"]
    assert "planner.predictors" in emitted_by_node["draft_plan"]
    assert "intake.question_type" in emitted_by_node["classify"]
    assert "intake.question_type" in emitted_by_node["reclassify"]


def test_execute_logs_a_record_only_for_multi_group(scenario_logs):
    execute_points = set(DECISION_NODES["execute"])
    counts = {name: sum(e["decision_point"] in execute_points for e in scenario_logs[name][0]["decision_log"])
              for name in ("S1", "S5", "S6")}
    assert counts == {"S1": 0, "S5": 1, "S6": 0}


def test_scenario_specifics(scenario_logs):
    def points(name):
        return [e["decision_point"] for e in scenario_logs[name][0]["decision_log"]]

    assert "intake.clarification" in points("S1")
    assert points("S1").count("stage1.clean.execution") == 2
    s2 = scenario_logs["S2"][0]["decision_log"]
    human_predictors = [e for e in s2 if e["decision_point"] == "planner.predictors" and e["decided_by"] == "human"]
    assert len(human_predictors) == 1 and human_predictors[0]["chosen"] == "channel"
    s4_approval = next(e for e in scenario_logs["S4"][0]["decision_log"] if e["decision_point"] == "stage1.clean.approval")
    assert s4_approval["chosen"] == "auto_approve"
    s5_post_hoc = next(e for e in scenario_logs["S5"][0]["decision_log"] if e["decision_point"] == "stage2.post_hoc")
    assert s5_post_hoc["chosen"] in ("tukey_hsd", "dunn")
    assert "stage3.encoding" in points("S6")
    assert "planner.draft_plan" in points("S7") and "stage2.test_selection" not in points("S7")
    assert points("S8")[-1] == "planner.stage_plan" and "planner.route" not in points("S8")
    # M1.2: the describe step runs after cleaning (S1) and when there is nothing to clean (S2), not for S7
    assert "stage1.describe.execution" in points("S1") and "stage1.describe.execution" in points("S2")
    assert "stage1.describe.execution" not in points("S7")
    # M1.3b: S11 runs the exploratory round
    assert "planner.stage_gate" in points("S11") and "planner.round_start" in points("S11")
    assert len(scenario_logs["S11"][0]["rounds"]) == 2
    s10_route = scenario_logs["S10"][0]["decision_log"][-1]
    assert (s10_route["decision_point"], s10_route["chosen"]) == ("planner.route", "unsupported")
