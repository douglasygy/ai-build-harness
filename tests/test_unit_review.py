from types import SimpleNamespace

from pipeline import unit_review


class _FakeLLM:
    model_name = "fake"

    def __init__(self):
        self.messages = None

    def invoke(self, messages):
        self.messages = messages
        return SimpleNamespace(content='{"verdict": "pass", "problems": []}',
                               response_metadata={"token_usage": {"cost": 0.01, "prompt_tokens": 1, "completion_tokens": 1}})


def _project(tmp_path):
    (tmp_path / "code").mkdir()
    (tmp_path / "code" / "a.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "SPEC.md").write_text("# T\n\n## One\nfirst\n\n## Two\nsecond\n", encoding="utf-8")
    return tmp_path


def test_unit_review_sends_spec_and_code_and_defaults_reviewer_model(tmp_path, monkeypatch, capsys):
    proj = _project(tmp_path)
    fake = _FakeLLM()
    seen = {}

    def fake_get_chat_model(role, model_override=None, project_dir=None):
        seen["role"], seen["override"] = role, model_override
        return fake

    monkeypatch.setattr("pipeline.model_factory.get_chat_model", fake_get_chat_model)
    rc = unit_review.main(["--project-dir", str(proj), "--unit", "u", "--focus", "f", "--spec-sections", "One", "a.py"])
    assert rc == 0
    assert seen == {"role": "reviewer", "override": unit_review.DEFAULT_REVIEWER_MODEL}
    user = fake.messages[1][1]
    assert "first" in user and "second" not in user and "x = 1" in user
    assert '"verdict": "pass"' in capsys.readouterr().out


def test_unit_review_respects_project_reviewer(tmp_path, monkeypatch):
    proj = _project(tmp_path)
    (proj / "models.yaml").write_text("roles:\n  reviewer:\n    model: custom/model\n", encoding="utf-8")
    seen = {}
    monkeypatch.setattr("pipeline.model_factory.get_chat_model",
                        lambda role, model_override=None, project_dir=None: seen.update(override=model_override) or _FakeLLM())
    assert unit_review.main(["--project-dir", str(proj), "--unit", "u", "--focus", "f", "a.py"]) == 0
    assert seen["override"] is None


def test_unit_review_web_uses_web_role_prompt_and_attachments(tmp_path, monkeypatch):
    proj = _project(tmp_path)
    (proj / "checks.txt").write_text("tsc: 0 errors", encoding="utf-8")
    fake = _FakeLLM()
    seen = {}

    def fake_get_chat_model(role, model_override=None, project_dir=None):
        seen["role"], seen["override"] = role, model_override
        return fake

    monkeypatch.setattr("pipeline.model_factory.get_chat_model", fake_get_chat_model)
    rc = unit_review.main(["--project-dir", str(proj), "--unit", "u", "--focus", "f", "--platform", "web",
                           "--checks-file", str(proj / "checks.txt"), "a.py"])
    assert rc == 0
    assert seen == {"role": "reviewer_web", "override": None}
    assert "WEB FRONTEND UNIT" in fake.messages[0][1]
    assert "VERIFY OUTPUT" in fake.messages[1][1] and "tsc: 0 errors" in fake.messages[1][1]


def test_unit_review_context_files_and_checklist(tmp_path, monkeypatch):
    proj = _project(tmp_path)
    (proj / "code" / "reader.py").write_text("y = 2\n", encoding="utf-8")
    fake = _FakeLLM()
    monkeypatch.setattr("pipeline.model_factory.get_chat_model", lambda role, model_override=None, project_dir=None: fake)
    assert unit_review.main(["--project-dir", str(proj), "--unit", "u", "--focus", "f",
                             "--context-files", "reader.py", "a.py"]) == 0
    system, user = fake.messages[0][1], fake.messages[1][1]
    assert "STANDARD DEFECT CHECKLIST" in system
    assert "CONTEXT FILE (not under review): reader.py" in user and "y = 2" in user
    assert "already reviewed" not in user and "boundary with" in user


def test_unit_review_no_checklist_and_no_context(tmp_path, monkeypatch):
    proj = _project(tmp_path)
    fake = _FakeLLM()
    monkeypatch.setattr("pipeline.model_factory.get_chat_model", lambda role, model_override=None, project_dir=None: fake)
    assert unit_review.main(["--project-dir", str(proj), "--unit", "u", "--focus", "f", "--no-checklist", "a.py"]) == 0
    assert "STANDARD DEFECT CHECKLIST" not in fake.messages[0][1]
    assert "CONTEXT FILE" not in fake.messages[1][1]


def test_unit_review_refuses_anthropic_reviewer(tmp_path, monkeypatch):
    proj = _project(tmp_path)
    fake = _FakeLLM()
    fake.model_name = "anthropic/claude-sonnet-5.5"
    monkeypatch.setattr("pipeline.model_factory.get_chat_model", lambda role, model_override=None, project_dir=None: fake)
    assert unit_review.main(["--project-dir", str(proj), "--unit", "u", "--focus", "f", "a.py"]) == 2
    assert fake.messages is None
