from dotenv import load_dotenv

load_dotenv()

import pytest  # noqa: E402

_REAL_MODEL_MESSAGE = (
    "a non-live test tried to call the real model (init_chat_model); inject a fake (classify_fn, draft_fn, "
    "propose_fn, report_fn, or patch init_chat_model as tests/test_prompt_canaries.py does) or mark the test live"
)


@pytest.fixture(autouse=True)
def _block_real_model_in_non_live_tests(request, monkeypatch):
    """G48: a test not marked live fails if it tries to create the real chat model. A test that patches
    init_chat_model itself still works, because its patch is applied after this one."""
    if request.node.get_closest_marker("live"):
        return
    import core.classifier
    import core.report

    def refuse(*args, **kwargs):
        raise RuntimeError(_REAL_MODEL_MESSAGE)
    monkeypatch.setattr(core.classifier, "init_chat_model", refuse)
    monkeypatch.setattr(core.report, "init_chat_model", refuse)
