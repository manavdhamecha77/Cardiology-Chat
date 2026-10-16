import pytest


@pytest.fixture(autouse=True)
def disable_external_model_configuration(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    for name in ("OPENAI_API_KEY", "OPENAI_MODEL", "OPENAI_BASE_URL", "OPENAI_TIMEOUT_SECONDS", "OLLAMA_API_KEY", "OLLAMA_MODEL", "OLLAMA_BASE_URL", "OLLAMA_TIMEOUT_SECONDS"):
        monkeypatch.delenv(name, raising=False)
