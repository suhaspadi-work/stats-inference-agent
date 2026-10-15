"""
M0.2 step 8a (G26): the wrangle node is split into wrangle_propose (one model
call) and wrangle_review (the checklist interrupt, then execution), so a
resume never re-runs the proposal call. All tests use a fake proposal
function; none calls the model.
"""
import numpy as np
import pandas as pd
import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from core.agent_state import AgentState
from core.classifier import _ProposedOperationOut
from core.dataset import LocalDiskStore
from core.graph import build_agent_graph
from core.ingestion import load_dataset, profile_dataset
from core.nodes import make_wrangle_propose_node, make_wrangle_review_node, route_after_wrangle_propose


@pytest.fixture
def store(tmp_path):
    return LocalDiskStore(base_dir=tmp_path / "store")


def make_dataset(df, tmp_path, store, name="data"):
    path = tmp_path / f"{name}.csv"
    df.to_csv(path, index=False)
    return load_dataset(path, session_id="session-abc", name=name, store=store)


def build_graph(store, propose_fn):
    graph = StateGraph(AgentState)
    graph.add_node("wrangle_propose", make_wrangle_propose_node(store, propose_fn=propose_fn))
    graph.add_node("wrangle_review", make_wrangle_review_node(store))
    graph.add_node("draft_plan", lambda state: {})
    graph.add_edge(START, "wrangle_propose")
    graph.add_conditional_edges("wrangle_propose", route_after_wrangle_propose,
                                {"wrangle_review": "wrangle_review", "draft_plan": "draft_plan"})
    graph.add_edge("wrangle_review", "draft_plan")
    graph.add_edge("draft_plan", END)
    return graph.compile(checkpointer=InMemorySaver())


def _proposal(index, op_type, params, rationale):
    return _ProposedOperationOut(
        issue_index=index, op_type=op_type, params=params, rationale=rationale,
        alternative_approach="leave the data as is", alternative_rejected_because="it would bias the result",
    )


class CountingProposer:
    """
    Call 1: dedupe for duplicate rows, filter not_null for a missingness issue.
    Call 2 and later: a different operation per issue, so a re-run would be visible.
    """
    def __init__(self):
        self.calls = 0

    def __call__(self, issues):
        self.calls += 1
        proposals = []
        for i, issue in enumerate(issues):
            if self.calls > 1:
                proposals.append(_proposal(i, "filter", {"column": "id", "condition": "equals", "value": 1},
                                           "a different proposal from a second call"))
            elif issue.issue_type == "duplicate_rows":
                proposals.append(_proposal(i, "dedupe", {"subset": []}, "remove exact duplicate rows"))
            else:
                proposals.append(_proposal(i, "filter", {"column": issue.affected_columns[0],
                                                         "condition": "not_null", "value": None},
                                           "drop rows missing this column"))
        return proposals


def _dupes_df():
    return pd.DataFrame({"id": [1, 2, 2, 3], "region": ["East", "West", "West", "East"]})


def _dupes_and_missing_df():
    region = ["East", "West"] * 10
    region[3] = region[7] = region[11] = None  # 3 of 20 rows missing (above the 5% threshold)
    df = pd.DataFrame({"id": range(1, 21), "region": region, "spend": np.arange(20, dtype=float)})
    return pd.concat([df, df.iloc[[0]]], ignore_index=True)  # one exact duplicate


def _start(app, ds, store, thread, **extra):
    state = {"dataset": ds, "profile": profile_dataset(ds, store=store),
             "request_text": "x", "session_id": "session-abc", **extra}
    return app.invoke(state, {"configurable": {"thread_id": thread}})


def _resume(app, thread, decisions):
    return app.invoke(Command(resume={"decisions": decisions}), {"configurable": {"thread_id": thread}})


# --- G26: one proposal call, and the approved operation is the one shown -------

def test_proposal_function_called_once_across_pause_and_resume(tmp_path, store):
    proposer = CountingProposer()
    app = build_graph(store, proposer)
    paused = _start(app, make_dataset(_dupes_df(), tmp_path, store), store, "t1")
    assert "__interrupt__" in paused
    assert proposer.calls == 1
    _resume(app, "t1", [{"index": 0, "action": "approve"}])
    assert proposer.calls == 1


def test_the_operation_that_runs_is_the_one_from_the_first_call(tmp_path, store):
    proposer = CountingProposer()
    app = build_graph(store, proposer)
    paused = _start(app, make_dataset(_dupes_df(), tmp_path, store), store, "t2")
    assert paused["__interrupt__"][0].value["proposed_operations"][0]["op_type"] == "dedupe"
    final = _resume(app, "t2", [{"index": 0, "action": "approve"}])
    cleaned = store.read(final["dataset"].storage_key)
    assert final["dataset"].version == 2
    assert len(cleaned) == 3                      # the dedupe ran
    assert sorted(cleaned["id"]) == [1, 2, 3]     # not the call-2 filter (id == 1)
    assert final["pending_wrangling"] is None


def test_pause_payload_has_the_old_shape(tmp_path, store):
    app = build_graph(store, CountingProposer())
    payload = _start(app, make_dataset(_dupes_df(), tmp_path, store), store, "t3")["__interrupt__"][0].value
    assert payload == {
        "type": "wrangling_approval",
        "question": "Review the proposed data-cleaning operations before they run.",
        "proposed_operations": [{
            "index": 0, "op_type": "dedupe", "params": {"subset": None},
            "rationale": "remove exact duplicate rows", "alternative": "leave the data as is",
        }],
        "allowed_decisions": ["approve", "edit", "reject"],
    }


# --- routing and autonomy ---------------------------------------------------------

def test_no_issues_never_calls_the_proposer_and_skips_review(tmp_path, store):
    proposer = CountingProposer()
    app = build_graph(store, proposer)
    ds = make_dataset(pd.DataFrame({"id": [1, 2, 3], "region": ["East", "West", "East"]}), tmp_path, store)
    result = _start(app, ds, store, "t4")
    assert "__interrupt__" not in result
    assert proposer.calls == 0
    assert result["pending_wrangling"] is None
    assert result["dataset"].handle == ds.handle


def test_stakeholder_mode_dedupes_without_a_pause(tmp_path, store):
    proposer = CountingProposer()
    app = build_graph(store, proposer)
    result = _start(app, make_dataset(_dupes_df(), tmp_path, store), store, "t5", autonomy_mode="stakeholder")
    assert "__interrupt__" not in result
    assert proposer.calls == 1
    assert result["dataset"].version == 2
    assert len(store.read(result["dataset"].storage_key)) == 3


def test_stakeholder_mode_pauses_only_for_missingness(tmp_path, store):
    proposer = CountingProposer()
    app = build_graph(store, proposer)
    ds = make_dataset(_dupes_and_missing_df(), tmp_path, store)
    payload = _start(app, ds, store, "t6", autonomy_mode="stakeholder")["__interrupt__"][0].value
    ops = payload["proposed_operations"]
    assert [(o["index"], o["op_type"]) for o in ops] == [(1, "filter")]
    assert ops[0]["params"]["column"] == "region"
    assert payload["auto_approved_note"] == (
        "1 additional operation(s) were auto-approved under 'stakeholder' autonomy mode "
        "and are not shown here for review.")
    final = _resume(app, "t6", [{"index": 1, "action": "approve"}])
    cleaned = store.read(final["dataset"].storage_key)
    assert len(cleaned) == 17  # 21 rows: the auto-approved dedupe drops 1, the filter drops the 3 missing
    assert cleaned["region"].notna().all()
    assert proposer.calls == 1


# --- decisions --------------------------------------------------------------------

def test_reject_records_the_issue_and_leaves_the_dataset(tmp_path, store):
    app = build_graph(store, CountingProposer())
    ds = make_dataset(_dupes_df(), tmp_path, store)
    _start(app, ds, store, "t7")
    final = _resume(app, "t7", [{"index": 0, "action": "reject", "reason": "they are real repeat orders"}])
    assert final["dataset"].handle == ds.handle
    issues = final["pending_unaddressed_issues"]
    assert len(issues) == 1
    assert issues[0].issue_type == "duplicate_rows"
    assert issues[0].rejection_reason == "they are real repeat orders"
    assert issues[0].proposed_operation == "dedupe({'subset': None})"


def test_edit_runs_the_edited_params(tmp_path, store):
    app = build_graph(store, CountingProposer())
    df = pd.DataFrame({"id": [1, 2, 2, 3], "region": ["East", "West", "West", "West"]})
    _start(app, make_dataset(df, tmp_path, store), store, "t8")
    # Deduping on region alone keeps one row per region, unlike the proposed full-row dedupe.
    final = _resume(app, "t8", [{"index": 0, "action": "edit", "params": {"subset": ["region"]}}])
    cleaned = store.read(final["dataset"].storage_key)
    assert sorted(cleaned["region"]) == ["East", "West"]


def test_stale_pending_wrangling_does_not_misroute(tmp_path, store):
    proposer = CountingProposer()
    app = build_graph(store, proposer)
    paused = _start(app, make_dataset(_dupes_df(), tmp_path, store, name="dupes"), store, "t9")
    assert paused["pending_wrangling"] is not None  # abandoned: never resumed

    clean = make_dataset(pd.DataFrame({"id": [1, 2, 3], "region": ["East", "West", "East"]}),
                         tmp_path, store, name="clean")
    result = _start(app, clean, store, "t9")
    assert "__interrupt__" not in result
    assert result["pending_wrangling"] is None
    assert result["dataset"].handle == clean.handle
    assert proposer.calls == 1


def test_failing_approved_operation_finishes_silently(tmp_path, store):
    """Pins current behaviour: the failed operation is discarded and the failure is not reported anywhere."""
    app = build_graph(store, CountingProposer())
    ds = make_dataset(_dupes_df(), tmp_path, store)
    _start(app, ds, store, "t10")
    final = _resume(app, "t10", [{"index": 0, "action": "edit", "params": {"subset": ["no_such_column"]}}])
    assert "__interrupt__" not in final
    assert final["dataset"].handle == ds.handle
    assert final["pending_wrangling"] is None
    assert "pending_unaddressed_issues" not in final


# --- the real graph ---------------------------------------------------------------

def test_full_graph_wrangle_edges(tmp_path):
    app = build_agent_graph(LocalDiskStore(base_dir=tmp_path))
    edges = {(e.source, e.target): e.conditional for e in app.get_graph().edges}
    assert edges[("profile", "plan_stages")] is False and edges[("plan_stages", "wrangle_propose")] is True  # M1.1
    assert edges[("wrangle_propose", "wrangle_review")] is True
    assert edges[("wrangle_propose", "draft_plan")] is True
    assert edges[("wrangle_review", "draft_plan")] is False
    assert "wrangle" not in {node for pair in edges for node in pair}


# --- cleaning decision records (M0.2 step 8b) ----------------------------------------

def _points(log):
    return [(e["decision_point"], e["chosen"]) for e in log]


def test_records_at_the_pause_are_scan_and_fix(tmp_path, store):
    app = build_graph(store, CountingProposer())
    paused = _start(app, make_dataset(_dupes_df(), tmp_path, store), store, "r1")
    assert _points(paused["decision_log"]) == [
        ("stage1.clean.scan", "propose_fixes"), ("stage1.clean.fix", "dedupe")]


def test_records_after_approve_link_fix_approval_execution_and_dataset(tmp_path, store):
    app = build_graph(store, CountingProposer())
    _start(app, make_dataset(_dupes_df(), tmp_path, store), store, "r2")
    final = _resume(app, "r2", [{"index": 0, "action": "approve"}])
    log = final["decision_log"]
    assert _points(log) == [
        ("stage1.clean.scan", "propose_fixes"), ("stage1.clean.fix", "dedupe"),
        ("stage1.clean.approval", "approve"), ("stage1.clean.execution", "applied")]
    fix, approval, execution = log[1], log[2], log[3]
    assert approval["decided_by"] == "human"
    assert approval["evidence"]["fix_record_id"] == fix["id"]
    assert execution["evidence"]["approval_record_id"] == approval["id"]
    assert execution["evidence"]["operation_id"] == final["dataset"].created_from_operation
    assert execution["evidence"]["output_handle"] == final["dataset"].handle
    assert (execution["evidence"]["rows_before"], execution["evidence"]["rows_after"]) == (4, 3)


def test_records_for_an_edit(tmp_path, store):
    app = build_graph(store, CountingProposer())
    _start(app, make_dataset(_dupes_df(), tmp_path, store), store, "r3")
    final = _resume(app, "r3", [{"index": 0, "action": "edit", "params": {"subset": ["region"]}}])
    log = final["decision_log"]
    approval, execution = log[2], log[3]
    assert (approval["chosen"], approval["decided_by"], approval["overridden_by_human"]) == ("edit", "human", True)
    assert approval["original_choice"] == "approve as proposed: dedupe({'subset': None})"
    assert approval["evidence"]["edited_params"] == {"subset": ["region"]}
    assert execution["chosen"] == "applied"
    assert execution["evidence"]["params"] == {"subset": ["region"]}
    assert execution["evidence"]["operation_id"] == final["dataset"].created_from_operation


def test_records_for_a_reject(tmp_path, store):
    app = build_graph(store, CountingProposer())
    _start(app, make_dataset(_dupes_df(), tmp_path, store), store, "r4")
    final = _resume(app, "r4", [{"index": 0, "action": "reject", "reason": "they are real repeat orders"}])
    assert _points(final["decision_log"]) == [
        ("stage1.clean.scan", "propose_fixes"), ("stage1.clean.fix", "dedupe"),
        ("stage1.clean.approval", "reject")]
    assert final["decision_log"][2]["rationale"] == "they are real repeat orders"
    assert final["decision_log"][2]["evidence"]["carried_forward_as_unaddressed"] is True
    assert len(final["pending_unaddressed_issues"]) == 1


def test_records_when_only_the_second_operation_gets_a_decision(tmp_path, store):
    app = build_graph(store, CountingProposer())
    _start(app, make_dataset(_dupes_and_missing_df(), tmp_path, store), store, "r4b")
    final = _resume(app, "r4b", [{"index": 1, "action": "approve"}])
    log = final["decision_log"]
    fixes = log[1:3]
    assert [f["chosen"] for f in fixes] == ["dedupe", "filter"]
    assert _points(log[3:]) == [
        ("stage1.clean.approval", "no_decision"),
        ("stage1.clean.approval", "approve"),
        ("stage1.clean.execution", "applied")]
    assert log[3]["evidence"]["fix_record_id"] == fixes[0]["id"]
    assert log[4]["evidence"]["fix_record_id"] == fixes[1]["id"]
    assert log[5]["evidence"]["op_type"] == "filter"
    cleaned = store.read(final["dataset"].storage_key)
    assert len(cleaned) == 18                    # 21 rows minus the 3 missing ones; no dedupe
    assert int(cleaned.duplicated().sum()) == 1  # the duplicate row is still there


def test_records_in_stakeholder_mode_follow_index_order(tmp_path, store):
    app = build_graph(store, CountingProposer())
    _start(app, make_dataset(_dupes_and_missing_df(), tmp_path, store), store, "r5", autonomy_mode="stakeholder")
    log = _resume(app, "r5", [{"index": 1, "action": "approve"}])["decision_log"]
    assert _points(log) == [
        ("stage1.clean.scan", "propose_fixes"),
        ("stage1.clean.fix", "dedupe"), ("stage1.clean.fix", "filter"),
        ("stage1.clean.approval", "auto_approve"), ("stage1.clean.execution", "applied"),
        ("stage1.clean.approval", "approve"), ("stage1.clean.execution", "applied")]
    assert log[3]["decided_by"] == "code_rule"
    assert log[3]["rationale"] == "stakeholder mode auto-approves duplicate_rows only"
    assert log[5]["decided_by"] == "human"
    assert log[3]["evidence"]["fix_record_id"] == log[1]["id"]
    assert log[5]["evidence"]["fix_record_id"] == log[2]["id"]


def test_records_with_no_issues_are_only_the_scan(tmp_path, store):
    app = build_graph(store, CountingProposer())
    ds = make_dataset(pd.DataFrame({"id": [1, 2, 3], "region": ["East", "West", "East"]}), tmp_path, store)
    result = _start(app, ds, store, "r6")
    assert _points(result["decision_log"]) == [("stage1.clean.scan", "skipped")]


def test_records_for_a_failing_approved_operation(tmp_path, store):
    app = build_graph(store, CountingProposer())
    ds = make_dataset(_dupes_df(), tmp_path, store)
    _start(app, ds, store, "r7")
    final = _resume(app, "r7", [{"index": 0, "action": "edit", "params": {"subset": ["no_such_column"]}}])
    log = final["decision_log"]
    assert _points(log)[2:] == [("stage1.clean.approval", "edit"), ("stage1.clean.execution", "failed")]
    assert log[3]["evidence"]["failure_reason"] == "Cannot dedupe — columns not found: ['no_such_column']"
    assert final["dataset"].handle == ds.handle


def test_records_for_an_empty_decision_list_pin_g36(tmp_path, store):
    app = build_graph(store, CountingProposer())
    ds = make_dataset(_dupes_df(), tmp_path, store)
    _start(app, ds, store, "r8")
    final = _resume(app, "r8", [])
    assert _points(final["decision_log"])[2:] == [("stage1.clean.approval", "no_decision")]
    assert final["dataset"].handle == ds.handle
    assert "pending_unaddressed_issues" not in final


def test_records_for_an_unrecognized_action_pin_g36(tmp_path, store):
    app = build_graph(store, CountingProposer())
    _start(app, make_dataset(_dupes_df(), tmp_path, store), store, "r9")
    final = _resume(app, "r9", [{"index": 0, "action": "maybe"}])
    approval = final["decision_log"][2]
    assert (approval["chosen"], approval["decided_by"]) == ("approve", "code_rule")
    assert approval["evidence"]["received_action"] == "maybe"
    assert final["decision_log"][3]["chosen"] == "applied"
    assert final["dataset"].version == 2


def test_record_ids_are_unique_across_pause_and_resume(tmp_path, store):
    app = build_graph(store, CountingProposer())
    paused = _start(app, make_dataset(_dupes_and_missing_df(), tmp_path, store), store, "r10")
    paused_ids = [e["id"] for e in paused["decision_log"]]
    final = _resume(app, "r10", [{"index": 0, "action": "approve"}, {"index": 1, "action": "reject"}])
    ids = [e["id"] for e in final["decision_log"]]
    assert len(ids) == len(set(ids))
    assert ids[:len(paused_ids)] == paused_ids  # the pause's records appear once, unchanged
    assert len(ids) == 3 + 2 + 1                # scan, 2 fixes; approve + execution; reject
