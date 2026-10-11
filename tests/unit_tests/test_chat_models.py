import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest
from langchain_core.exceptions import (
    ContextOverflowError,
    ModelAPIError,
    ModelNotFoundError,
)
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from langchain_llama_cpp import (
    ChatLlama,
    LlamaAPIError,
    LlamaConnectionError,
    LlamaContextOverflowError,
    LlamaInvalidRequestError,
    LlamaModelNotFoundError,
)
from langchain_llama_cpp import chat_models as chat_models_module

BASE_URL = "http://llama.test:8080"

# Captured from llama-server b11429 (gemma, thinking on), 2026-10-11.
REPLY: dict[str, Any] = {
    "choices": [
        {
            "finish_reason": "stop",
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "391",
                "reasoning_content": "17 x 23 = 391.",
            },
        }
    ],
    "created": 1791710126,
    "model": "google/gemma-4-26B-A4B-it-qat-q4_0-gguf:Q4_0",
    "system_fingerprint": "b11429-d81235049",
    "object": "chat.completion",
    "usage": {
        "completion_tokens": 217,
        "prompt_tokens": 32,
        "total_tokens": 249,
        "prompt_tokens_details": {"cached_tokens": 27},
    },
    "id": "chatcmpl-pmligs5DcXIiejxeskv7YlviseIhyB8W",
    "timings": {"cache_n": 27, "prompt_n": 5, "predicted_n": 217},
}

OVERFLOW = {
    "error": {
        "code": 400,
        "type": "exceed_context_size_error",
        "message": "request (5012 tokens) exceeds the available context size",
        "n_prompt_tokens": 5012,
        "n_ctx": 4096,
    }
}

Handler = Callable[[httpx.Request], httpx.Response]


class Server:
    """A fake llama-swap: answers with queued responses and records requests."""

    def __init__(self, *responses: httpx.Response | Exception) -> None:
        self.responses = list(responses)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    @property
    def bodies(self) -> list[dict[str, Any]]:
        return [json.loads(request.content) for request in self.requests]


def make_llm(server: Server, **kwargs: Any) -> ChatLlama:
    client = httpx.Client(transport=httpx.MockTransport(server))
    return ChatLlama(model="gemma", base_url=BASE_URL, http_client=client, **kwargs)


def ok(body: dict[str, Any] | None = None) -> httpx.Response:
    return httpx.Response(200, json=body or REPLY)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Record retry delays instead of sleeping."""
    delays: list[float] = []
    monkeypatch.setattr(chat_models_module.time, "sleep", delays.append)
    return delays


# --- request ---------------------------------------------------------------


def test_request_url_and_minimal_body() -> None:
    server = Server(ok())
    make_llm(server).invoke("What is 17 x 23?")
    assert str(server.requests[0].url) == f"{BASE_URL}/v1/chat/completions"
    assert server.bodies[0] == {
        "model": "gemma",
        "messages": [{"role": "user", "content": "What is 17 x 23?"}],
    }


def test_fields_left_at_none_are_left_out_and_set_fields_are_sent() -> None:
    server = Server(ok())
    llm = make_llm(
        server,
        temperature=0.2,
        max_tokens=300,
        top_k=40,
        min_p=0.05,
        chat_template_kwargs={"enable_thinking": True},
    )
    llm.invoke("hi")
    body = server.bodies[0]
    assert body["temperature"] == 0.2
    assert body["max_tokens"] == 300
    assert body["top_k"] == 40
    assert body["min_p"] == 0.05
    assert body["chat_template_kwargs"] == {"enable_thinking": True}
    assert "top_p" not in body
    assert "seed" not in body


def test_invoke_kwargs_and_stop_override_fields() -> None:
    server = Server(ok())
    llm = make_llm(server, max_tokens=300, stop=["A"])
    llm.invoke("hi", stop=["B"], max_tokens=50, cache_prompt=False)
    body = server.bodies[0]
    assert body["stop"] == ["B"]
    assert body["max_tokens"] == 50
    assert body["cache_prompt"] is False


def test_history_sends_reasoning_back() -> None:
    server = Server(ok())
    v0 = AIMessage(content="Paris.", additional_kwargs={"reasoning_content": "R1"})
    v1 = v0.model_copy(update={"content": v0.content_blocks})
    tool_turn = AIMessage(
        content="",
        additional_kwargs={"reasoning_content": "R2"},
        tool_calls=[{"name": "get_weather", "args": {"city": "Paris"}, "id": "call_1"}],
    )
    make_llm(server).invoke(
        [
            SystemMessage("Be brief."),
            HumanMessage("Capital of France?"),
            v0,
            v1,
            tool_turn,
            ToolMessage('{"temp_c": 18}', tool_call_id="call_1"),
        ]
    )
    messages = server.bodies[0]["messages"]
    assert messages[0] == {"role": "system", "content": "Be brief."}
    assert messages[2] == {
        "role": "assistant",
        "content": "Paris.",
        "reasoning_content": "R1",
    }
    assert messages[3] == messages[2]  # v1 blocks become the same wire message
    assert messages[4]["reasoning_content"] == "R2"
    assert messages[4]["tool_calls"][0]["function"] == {
        "name": "get_weather",
        "arguments": '{"city": "Paris"}',
    }
    assert messages[5] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": '{"temp_c": 18}',
    }


def test_assistant_without_reasoning_has_no_reasoning_field() -> None:
    server = Server(ok())
    make_llm(server).invoke([HumanMessage("hi"), AIMessage("hello"), HumanMessage("?")])
    assert server.bodies[0]["messages"][1] == {"role": "assistant", "content": "hello"}


# --- response --------------------------------------------------------------


def test_response_content_reasoning_and_metadata() -> None:
    msg = make_llm(Server(ok())).invoke("What is 17 x 23?")
    assert msg.content == "391"
    assert msg.additional_kwargs == {"reasoning_content": "17 x 23 = 391."}
    assert msg.content_blocks[0] == {"type": "reasoning", "reasoning": "17 x 23 = 391."}
    assert msg.response_metadata == {
        # Core merges llm_output into response_metadata for a single generation.
        "token_usage": REPLY["usage"],
        "finish_reason": "stop",
        "model_name": "google/gemma-4-26B-A4B-it-qat-q4_0-gguf:Q4_0",
        "system_fingerprint": "b11429-d81235049",
        "id": "chatcmpl-pmligs5DcXIiejxeskv7YlviseIhyB8W",
        "timings": {"cache_n": 27, "prompt_n": 5, "predicted_n": 217},
        "model_provider": "llama-cpp",
    }


def test_usage_metadata() -> None:
    msg = make_llm(Server(ok())).invoke("hi")
    assert isinstance(msg, AIMessage)
    assert msg.usage_metadata == {
        "input_tokens": 32,
        "output_tokens": 217,
        "total_tokens": 249,
        "input_token_details": {"cache_read": 27},
        "output_token_details": {},
    }


def test_reasoning_tokens_are_mapped_if_the_server_ever_sends_them() -> None:
    usage = {
        **REPLY["usage"],
        "completion_tokens_details": {"reasoning_tokens": 200},
    }
    msg = make_llm(Server(ok({**REPLY, "usage": usage}))).invoke("hi")
    assert isinstance(msg, AIMessage)
    assert msg.usage_metadata is not None
    assert msg.usage_metadata.get("output_token_details") == {"reasoning": 200}


def test_generate_sums_raw_usage_across_calls() -> None:
    result = make_llm(Server(ok(), ok())).generate(
        [[HumanMessage("hi")], [HumanMessage("again")]]
    )
    assert result.llm_output == {
        "token_usage": {
            "completion_tokens": 434,
            "prompt_tokens": 64,
            "total_tokens": 498,
            "prompt_tokens_details": {"cached_tokens": 54},
        },
        "model_name": "gemma",
        "system_fingerprint": "b11429-d81235049",
    }


def test_finish_reason_length_is_metadata_only() -> None:
    reply = {**REPLY, "choices": [{**REPLY["choices"][0], "finish_reason": "length"}]}
    msg = make_llm(Server(ok(reply))).invoke("hi")
    assert msg.response_metadata["finish_reason"] == "length"


def test_null_content_becomes_empty_string() -> None:
    choice = {**REPLY["choices"][0], "message": {"role": "assistant", "content": None}}
    msg = make_llm(Server(ok({**REPLY, "choices": [choice]}))).invoke("hi")
    assert msg.content == ""
    assert msg.additional_kwargs == {}


def test_tool_calls_are_parsed() -> None:
    message = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
            },
            {
                "id": "call_2",
                "type": "function",
                "function": {"name": "get_weather", "arguments": "{not json"},
            },
        ],
    }
    reply = {**REPLY, "choices": [{**REPLY["choices"][0], "message": message}]}
    msg = make_llm(Server(ok(reply))).invoke("hi")
    assert isinstance(msg, AIMessage)
    assert msg.tool_calls == [
        {
            "name": "get_weather",
            "args": {"city": "Paris"},
            "id": "call_1",
            "type": "tool_call",
        }
    ]
    assert msg.invalid_tool_calls[0]["id"] == "call_2"


# --- errors ----------------------------------------------------------------


def test_context_overflow() -> None:
    server = Server(httpx.Response(400, json=OVERFLOW))
    with pytest.raises(LlamaContextOverflowError) as info:
        make_llm(server).invoke("long prompt")
    error = info.value
    assert isinstance(error, ContextOverflowError)
    assert isinstance(error, httpx.HTTPStatusError)
    assert (error.n_prompt_tokens, error.n_ctx) == (5012, 4096)
    assert "exceeds the available context size" in str(error)
    assert len(server.requests) == 1  # not retried


def test_other_400_is_invalid_request_and_not_retried() -> None:
    body = {"error": {"code": 400, "type": "invalid_request_error", "message": "bad"}}
    server = Server(httpx.Response(400, json=body))
    with pytest.raises(LlamaInvalidRequestError, match="bad"):
        make_llm(server).invoke("hi")
    assert len(server.requests) == 1


def test_unknown_model_is_not_found() -> None:
    server = Server(httpx.Response(404, text="model nope not found"))
    with pytest.raises(LlamaModelNotFoundError, match="model nope not found") as info:
        make_llm(server).invoke("hi")
    assert isinstance(info.value, ModelNotFoundError)
    assert len(server.requests) == 1


def test_5xx_is_retried_then_raised(no_sleep: list[float]) -> None:
    server = Server(*[httpx.Response(500, text="boom")] * 3)
    with pytest.raises(LlamaAPIError) as info:
        make_llm(server).invoke("hi")
    assert isinstance(info.value, ModelAPIError)
    assert len(server.requests) == 3  # 1 + max_retries
    assert len(no_sleep) == 2
    assert 0.375 <= no_sleep[0] <= 0.5  # 0.5 s minus up to 25 % jitter
    assert 0.75 <= no_sleep[1] <= 1.0


def test_retry_succeeds_and_honours_retry_after(no_sleep: list[float]) -> None:
    busy = httpx.Response(503, headers={"Retry-After": "3"}, text="loading")
    server = Server(busy, ok())
    msg = make_llm(server).invoke("hi")
    assert msg.content == "391"
    assert no_sleep == [3.0]


def test_connection_error_is_retried_and_wrapped() -> None:
    server = Server(*[httpx.ConnectError("refused")] * 3)
    with pytest.raises(LlamaConnectionError) as info:
        make_llm(server).invoke("hi")
    assert isinstance(info.value.__cause__, httpx.ConnectError)
    assert len(server.requests) == 3


def test_max_retries_zero_disables_retries() -> None:
    server = Server(httpx.Response(500, text="boom"))
    with pytest.raises(LlamaAPIError):
        make_llm(server, max_retries=0).invoke("hi")
    assert len(server.requests) == 1


# --- construction ----------------------------------------------------------


def test_base_url_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLAMA_BASE_URL", "http://from-env:8080")
    assert ChatLlama(model="gemma").base_url == "http://from-env:8080"


def test_missing_base_url_fails_at_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LLAMA_BASE_URL", raising=False)
    with pytest.raises(ValueError, match="LLAMA_BASE_URL"):
        ChatLlama(model="gemma")


def test_default_client_has_no_timeout() -> None:
    llm = ChatLlama(model="gemma", base_url=BASE_URL)
    assert llm._get_client().timeout == httpx.Timeout(None)


def test_ls_params() -> None:
    llm = ChatLlama(model="gemma", base_url=BASE_URL, temperature=0.2, max_tokens=9)
    params = llm._get_ls_params()
    assert params.get("ls_provider") == "llama-cpp"
    assert params.get("ls_model_name") == "gemma"
    assert params.get("ls_temperature") == 0.2
    assert params.get("ls_max_tokens") == 9


# --- server defaults -------------------------------------------------------

# Trimmed from GET /upstream/gemma/props, llama-server b11429, 2026-10-11.
PROPS = {
    "default_generation_settings": {
        "params": {
            "seed": 4294967295,
            "temperature": 1.0,
            "top_k": 64,
            "top_p": 0.949999988079071,
            "min_p": 0.05000000074505806,
            "max_tokens": -1,
            "samplers": ["penalties", "top_k", "top_p", "min_p", "temperature"],
        },
        "n_ctx": 131072,
    },
    "model_alias": "gemma",
}


def test_get_server_defaults() -> None:
    server = Server(httpx.Response(200, json=PROPS))
    llm = make_llm(server)
    assert llm.get_server_defaults() == {
        "seed": 4294967295,
        "temperature": 1.0,
        "top_k": 64,
        "top_p": 0.95,
        "min_p": 0.05,
        "max_tokens": -1,
        "samplers": ["penalties", "top_k", "top_p", "min_p", "temperature"],
        "n_ctx": 131072,
    }
    request = server.requests[0]
    assert (request.method, str(request.url)) == (
        "GET",
        f"{BASE_URL}/upstream/gemma/props",
    )
    assert llm.temperature is None  # nothing is written back to the fields


def test_get_server_defaults_unknown_model() -> None:
    server = Server(httpx.Response(404, text="model nope not found"))
    with pytest.raises(LlamaModelNotFoundError):
        make_llm(server).get_server_defaults()


def test_ls_model_name_follows_per_call_model() -> None:
    llm = ChatLlama(model="gemma", base_url=BASE_URL)
    assert llm._get_ls_params(model="qwen").get("ls_model_name") == "qwen"


def test_per_call_model_is_sent() -> None:
    server = Server(ok())
    make_llm(server).invoke("hi", model="qwen")
    assert server.bodies[0]["model"] == "qwen"
