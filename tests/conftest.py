import pytest


@pytest.fixture(autouse=True)
def disable_external_model_configuration(monkeypatch):
    for name in ("OPENAI_API_KEY", "OPENAI_MODEL", "OPENAI_BASE_URL", "OPENAI_TIMEOUT_SECONDS"):
        monkeypatch.delenv(name, raising=False)
