"""
G48: pytest collects only tests/, and no test_*.py file may sit outside a tests/ directory, because a
script with a test_ name is collected (and run at import) by a pytest run from agent/.
"""
import configparser
import importlib
import sys
from pathlib import Path

import groq
import langchain.chat_models
import pytest

AGENT = Path(__file__).resolve().parent.parent
SKIPPED_DIRS = {".venv", "venv", ".git", "__pycache__", "node_modules", "site-packages"}
CHECK_SCRIPTS = ("eval.check_connection", "eval.check_groq_limits")


def stray_test_files(root):
    """Every test_*.py or *_test.py file under root that is not inside a directory named tests."""
    root = Path(root)
    found = []
    for path in root.rglob("*.py"):
        parts = path.relative_to(root).parts[:-1]
        if any(p in SKIPPED_DIRS for p in parts) or "tests" in parts:
            continue
        if path.name.startswith("test_") or path.name.endswith("_test.py"):
            found.append(path)
    return sorted(found)


def test_agent_has_no_stray_test_files():
    assert stray_test_files(AGENT) == []


def test_stray_test_files_are_detected_and_ignored_where_they_should_be(tmp_path):
    def touch(rel):
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")
        return path

    strays = [touch("test_x.py"), touch("eval/test_z.py"), touch("y_test.py")]
    for rel in ("tests/test_ok.py", "tests/sub/test_ok2.py", ".venv/lib/test_dep.py", "__pycache__/test_c.py",
                "eval/check_thing.py"):
        touch(rel)
    assert stray_test_files(tmp_path) == sorted(strays)


def test_pytest_ini_collects_only_tests_and_declares_live():
    config = configparser.ConfigParser()
    config.read(AGENT / "pytest.ini")
    assert config["pytest"]["testpaths"].strip() == "tests"
    assert "live:" in config["pytest"]["markers"]


def test_check_scripts_moved_to_eval():
    for old in ("test_connection.py", "test_groq_limits.py"):
        assert not (AGENT / old).exists(), old
    for module in CHECK_SCRIPTS:
        source = (AGENT / (module.replace(".", "/") + ".py")).read_text()
        assert "def main(" in source, module
        assert 'if __name__ == "__main__":' in source, module


@pytest.mark.parametrize("module", CHECK_SCRIPTS)
def test_importing_a_check_script_calls_nothing(module, monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError(f"{module} called the model or Groq at import")
    monkeypatch.setattr(langchain.chat_models, "init_chat_model", refuse)
    monkeypatch.setattr(groq, "Groq", refuse)
    monkeypatch.delitem(sys.modules, module, raising=False)
    imported = importlib.import_module(module)
    assert callable(imported.main)
