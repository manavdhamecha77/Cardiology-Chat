import json
from io import BytesIO
from urllib.error import HTTPError

import pytest

from cardio_rag.generation import GenerationError, OpenAICompatibleGenerator, _dotenv_values, generator_from_environment


def test_no_key_selects_local_fallback():
    generator = generator_from_environment()
    assert generator.provider == "extractive"


def test_key_requires_explicit_model(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-secret")
    with pytest.raises(GenerationError, match="OPENAI_MODEL"):
        generator_from_environment()


def test_dotenv_supports_ollama_settings_and_shell_variable_reference(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text(
        'OLLAMA_API_KEY=ollama-test-key\n'
        '$env:OPENAI_API_KEY = $env:OLLAMA_API_KEY\n'
        'OPENAI_BASE_URL="https://ollama.com/v1"\n'
        'OPENAI_MODEL="gemma4:31b\n',
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    values = _dotenv_values(env_path)
    assert values["OPENAI_API_KEY"] == "ollama-test-key"
    generator = generator_from_environment()
    assert generator.provider == "ollama_openai_compatible"
    assert generator.model == "gemma4:31b"


def test_openai_compatible_request_extracts_answer(monkeypatch):
    class FakeResponse(BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.close()

    def fake_urlopen(request, timeout):
        assert request.full_url == "https://example.test/v1/chat/completions"
        assert request.get_header("Authorization") == "Bearer test-secret"
        assert timeout == 7
        body = json.loads(request.data)
        assert body["temperature"] == 0
        return FakeResponse(b'{"choices":[{"message":{"content":"Grounded answer [guide:1]."}}]}')

    monkeypatch.setattr("cardio_rag.generation.urlopen", fake_urlopen)
    generator = OpenAICompatibleGenerator("test-secret", "test-model", "https://example.test/v1", timeout=7)
    result = generator.generate("Question", [])
    assert result.answer == "Grounded answer [guide:1]."
    assert result.model == "test-model"


def test_http_failure_is_not_hidden_as_fallback(monkeypatch):
    def fail_request(_request, timeout):
        raise HTTPError("https://example.test", 429, "rate limit", None, BytesIO())

    monkeypatch.setattr("cardio_rag.generation.urlopen", fail_request)
    generator = OpenAICompatibleGenerator("test-secret", "test-model", "https://example.test/v1")
    with pytest.raises(GenerationError, match="HTTP 429"):
        generator.generate("Question", [])


def test_pipeline_records_provider_error_without_extracting(monkeypatch):
    from cardio_rag.config import Config
    from cardio_rag.pipeline import QAPipeline
    from cardio_rag.retrieval import LexicalIndex
    from cardio_rag.schemas import Chunk

    monkeypatch.setenv("OPENAI_API_KEY", "test-secret")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")

    def fail_request(_request, timeout):
        raise HTTPError("https://example.test", 503, "unavailable", None, BytesIO())

    monkeypatch.setattr("cardio_rag.generation.urlopen", fail_request)
    chunk = Chunk("guide:1", "guide", "Guide", "Org", "url", "date", "license", "Basics", "Heart failure is a condition in which the heart cannot pump blood as well as it should.")
    record = QAPipeline(LexicalIndex([chunk]), Config(min_score=0.0)).answer("What is heart failure?", "baseline")
    assert record.abstained
    assert "HTTP 503" in record.answer
    assert "generator_error" in record.metadata
