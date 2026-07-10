from lang2action.config import DEFAULT_MODEL, load_settings


def test_defaults(monkeypatch):
    for var in ("LANG2ACTION_MODEL", "LANG2ACTION_PERCEPTION", "LANG2ACTION_SGG_URL"):
        monkeypatch.delenv(var, raising=False)
    s = load_settings()
    assert s.model == DEFAULT_MODEL
    assert s.perception_backend == "sim"
    assert s.sgg_url == "http://localhost:8000"


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("LANG2ACTION_MODEL", "anthropic:claude-haiku-4-5")
    monkeypatch.setenv("LANG2ACTION_PERCEPTION", "sgg")
    monkeypatch.setenv("LANG2ACTION_SGG_URL", "http://sgg:9000")
    s = load_settings()
    assert s.model == "anthropic:claude-haiku-4-5"
    assert s.perception_backend == "sgg"
    assert s.sgg_url == "http://sgg:9000"
