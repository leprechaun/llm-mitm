from llm_mitm.config import ModelEntry, OllamaModelConfig


def test_model_entry_id_is_provider_prefixed():
    entry = ModelEntry(
        name="llama3",
        provider="ollama",
        ollama=OllamaModelConfig(model="llama3.1:8b"),
    )
    assert entry.id == "ollama/llama3"
