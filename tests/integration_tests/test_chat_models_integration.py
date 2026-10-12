"""langchain-tests standard integration tests for ChatLlama.

Needs a llama-swap server: `uv run --env-file .env pytest tests/integration_tests`.
Each model family runs twice, with thinking off (or as low as it goes) and on.
Classes are grouped by model, so llama-swap swaps entries only between families.
"""

import os
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_tests.integration_tests import ChatModelIntegrationTests

from langchain_llama_cpp import ChatLlama

pytestmark = pytest.mark.skipif(
    not os.environ.get("LLAMA_BASE_URL"), reason="LLAMA_BASE_URL is not set"
)

# Thinking tokens count towards max_tokens, so the cap is higher with thinking on.
# 1000 was too low: qwen used 656-1000+ tokens for test_json_mode and gemma
# 276-1000+ for test_structured_output_optional_param (5 repeats each, 2026-10-11/12).
NO_THINKING_MAX_TOKENS = 300
THINKING_MAX_TOKENS = 2000


class ChatLlamaIntegrationTests(ChatModelIntegrationTests):
    """Shared settings; subclasses set the llama-swap entry and thinking."""

    model_name: str
    extra_params: dict[str, Any]

    @property
    def chat_model_class(self) -> type[ChatLlama]:
        return ChatLlama

    @property
    def chat_model_params(self) -> dict[str, Any]:
        return {"model": self.model_name, **self.extra_params}

    @property
    def supports_json_mode(self) -> bool:
        return True

    @pytest.mark.xfail(
        reason=(
            "llama-server can't force a named tool: tool_choice=<name> is sent as "
            "'required' (b11429). gemma may call the other tool; qwen's handler "
            "allows text before the call and keeps writing to max_tokens."
        )
    )
    def test_tool_choice(self, model: BaseChatModel) -> None:
        super().test_tool_choice(model)


class TestGemmaNoThinking(ChatLlamaIntegrationTests):
    model_name = "gemma"
    extra_params = {
        "chat_template_kwargs": {"enable_thinking": False},
        "max_tokens": NO_THINKING_MAX_TOKENS,
    }


class TestGemmaThinking(ChatLlamaIntegrationTests):
    model_name = "gemma"
    extra_params = {
        "chat_template_kwargs": {"enable_thinking": True},
        "max_tokens": THINKING_MAX_TOKENS,
    }


class TestQwenNoThinking(ChatLlamaIntegrationTests):
    model_name = "qwen"
    extra_params = {
        "chat_template_kwargs": {"enable_thinking": False},
        "max_tokens": NO_THINKING_MAX_TOKENS,
    }


class TestQwenThinking(ChatLlamaIntegrationTests):
    model_name = "qwen"
    extra_params = {
        "chat_template_kwargs": {"enable_thinking": True},
        "max_tokens": THINKING_MAX_TOKENS,
    }


# gpt-oss always reasons: "low" is as close to off as it gets.
class TestGptOssLowEffort(ChatLlamaIntegrationTests):
    model_name = "gpt-oss"
    extra_params = {
        "chat_template_kwargs": {"reasoning_effort": "low"},
        "max_tokens": THINKING_MAX_TOKENS,
    }


class TestGptOssMediumEffort(ChatLlamaIntegrationTests):
    model_name = "gpt-oss"
    extra_params = {
        "chat_template_kwargs": {"reasoning_effort": "medium"},
        "max_tokens": THINKING_MAX_TOKENS,
    }
