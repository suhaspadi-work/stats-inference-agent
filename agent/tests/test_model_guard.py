"""
G48: the conftest guard makes a test not marked live fail if it tries to create the real chat model.
"""
import langchain.chat_models
import pytest

import core.classifier
import core.report


@pytest.mark.parametrize("module", [core.classifier, core.report], ids=["classifier", "report"])
def test_non_live_test_cannot_create_the_real_model(module):
    with pytest.raises(RuntimeError, match="non-live test tried to call the real model"):
        module.init_chat_model("x")


def test_a_test_that_patches_init_chat_model_gets_its_own_function(monkeypatch):
    def own(*args, **kwargs):
        return "own fake"
    monkeypatch.setattr(core.classifier, "init_chat_model", own)
    assert core.classifier.init_chat_model("x") == "own fake"


@pytest.mark.live
def test_the_guard_does_not_apply_to_live_tests():
    """Makes no call, so it costs no quota."""
    assert core.classifier.init_chat_model is langchain.chat_models.init_chat_model
    assert core.report.init_chat_model is langchain.chat_models.init_chat_model
