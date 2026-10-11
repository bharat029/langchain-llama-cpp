"""langchain-tests standard unit tests for ChatLlama (offline)."""

from typing import Any

from langchain_tests.unit_tests import ChatModelUnitTests

from langchain_llama_cpp import ChatLlama


class TestChatLlamaUnit(ChatModelUnitTests):
    @property
    def chat_model_class(self) -> type[ChatLlama]:
        return ChatLlama

    @property
    def chat_model_params(self) -> dict[str, Any]:
        return {"model": "gemma", "base_url": "http://llama.test:8080"}

    @property
    def init_from_env_params(
        self,
    ) -> tuple[dict[str, str], dict[str, Any], dict[str, Any]]:
        return (
            {"LLAMA_BASE_URL": "http://llama.test:8080"},
            {"model": "gemma"},
            {"base_url": "http://llama.test:8080"},
        )

    @property
    def supports_json_mode(self) -> bool:
        return True
