from core.agent_state import AgentState
from core.dataset import Dataset
from core.plan import AnalysisPlan, QuestionType, CausalStatus, OutcomeType


def test_state_can_be_built_incrementally():
    """Mirrors how LangGraph nodes actually build state: partial dict, merged in over time."""
    state: AgentState = {
        "request_text": "Did West spend more than East?",
        "session_id": "session-abc",
    }
    assert "dataset" not in state
    assert "plan" not in state

    dataset = Dataset(
        session_id="session-abc", name="customers", version=1,
        storage_key="session-abc/customers_v1.csv",
        parent_version=None, created_from_operation=None,
    )
    state["dataset"] = dataset
    assert state["dataset"].handle == "customers@v1"


def test_clarification_needed_absence_is_the_no_interrupt_signal():
    state: AgentState = {"request_text": "x", "session_id": "y"}
    assert "clarification_needed" not in state

    state["clarification_needed"] = "Could this be read as association or driver_analysis?"
    assert "clarification_needed" in state
    assert state["clarification_needed"].startswith("Could this")


def test_state_holds_a_real_analysis_plan_unchanged():
    """Confirms AgentState doesn't distort or copy the AnalysisPlan -- it holds the real object."""
    plan = AnalysisPlan(
        request_text="test", session_id="session-abc",
        question_type=QuestionType.TWO_GROUP_COMPARISON,
        causal_status=CausalStatus.OBSERVATIONAL,
        outcome_name="total_spend", outcome_type=OutcomeType.CONTINUOUS,
        unit_of_analysis="customer", data_sources=("customers@v1",),
    )
    state: AgentState = {"request_text": "test", "session_id": "session-abc", "plan": plan}
    assert state["plan"] is plan
    assert state["plan"].status.value == "draft"