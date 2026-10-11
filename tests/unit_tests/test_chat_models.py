import asyncio
import json
from collections.abc import AsyncIterator, Callable, Iterator
from typing import Annotated, Any

import httpx
import pytest
from langchain_core.exceptions import (
    ContextOverflowError,
    ModelAPIError,
    ModelNotFoundError,
    ModelTimeoutError,
    OutputParserException,
)
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.output_parsers.format_instructions import (
    JSON_FORMAT_INSTRUCTIONS,
)
from langchain_core.tools import tool
from pydantic import BaseModel, Field, ValidationError
from typing_extensions import TypedDict

from langchain_llama_cpp import (
    ChatLlama,
    LlamaAPIError,
    LlamaConnectionError,
    LlamaContextOverflowError,
    LlamaInvalidRequestError,
    LlamaModelNotFoundError,
    LlamaTimeoutError,
)
from langchain_llama_cpp import chat_models as chat_models_module
from langchain_llama_cpp.chat_models import request_field_names

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
    transport = httpx.MockTransport(server)
    return ChatLlama(
        model="gemma",
        base_url=BASE_URL,
        http_client=httpx.Client(transport=transport),
        http_async_client=httpx.AsyncClient(transport=transport),
        **kwargs,
    )


def ok(body: dict[str, Any] | None = None) -> httpx.Response:
    return httpx.Response(200, json=body or REPLY)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Record retry delays instead of sleeping (sync and async)."""
    delays: list[float] = []

    async def fake_async_sleep(seconds: float) -> None:
        delays.append(seconds)

    monkeypatch.setattr(chat_models_module.time, "sleep", delays.append)
    monkeypatch.setattr(chat_models_module.asyncio, "sleep", fake_async_sleep)
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


# --- request field coverage ------------------------------------------------

# One value per request field, keyed by wire name (llama.cpp b11429).
ALL_FIELDS: dict[str, Any] = {
    "max_tokens": 300,
    "stop": ["\n\n"],
    "ignore_eos": False,
    "t_max_predict_ms": 5000,
    "n_indent": 4,
    "n": 1,
    "temperature": 0.7,
    "dynatemp_range": 0.1,
    "dynatemp_exponent": 1.0,
    "top_k": 40,
    "top_p": 0.9,
    "min_p": 0.05,
    "typical_p": 1.0,
    "top_n_sigma": -1.0,
    "xtc_probability": 0.0,
    "xtc_threshold": 0.1,
    "mirostat": 0,
    "mirostat_tau": 5.0,
    "mirostat_eta": 0.1,
    "adaptive_target": -1.0,
    "adaptive_decay": 0.9,
    "min_keep": 1,
    "samplers": ["top_k", "min_p", "temperature"],
    "seed": 42,
    "backend_sampling": False,
    "repeat_penalty": 1.1,
    "repeat_last_n": 64,
    "frequency_penalty": 0.1,
    "presence_penalty": 0.1,
    "dry_multiplier": 0.8,
    "dry_base": 1.75,
    "dry_allowed_length": 2,
    "dry_penalty_last_n": 64,
    "dry_sequence_breakers": ["\n"],
    "logit_bias": {"15043": -1.0},
    "grammar": 'root ::= "yes" | "no"',
    "json_schema": {"type": "object"},
    "chat_template_kwargs": {"enable_thinking": False},
    "reasoning_format": "deepseek",
    "reasoning_effort": "low",
    "reasoning_budget_tokens": 200,
    "reasoning_budget_message": "Time to answer.",
    "reasoning_control": True,
    "add_generation_prompt": True,
    "continue_final_message": False,
    "echo": False,
    "logprobs": True,
    "top_logprobs": 5,
    "n_probs": 5,
    "post_sampling_probs": True,
    "return_tokens": True,
    "timings_per_token": True,
    "response_fields": ["choices", "usage"],
    "cache_prompt": True,
    "n_cache_reuse": 256,
    "n_keep": -1,
    "n_discard": 0,
    "id_slot": 0,
    "lora": [{"id": 0, "scale": 1.0}],
    "return_progress": False,
    "sse_ping_interval": 10,
}


def test_request_fields_match_llama_server() -> None:
    assert request_field_names() == list(ALL_FIELDS)


def test_every_request_field_is_sent_under_its_own_name() -> None:
    server = Server(ok())
    make_llm(server, **ALL_FIELDS).invoke("hi")
    body = server.bodies[0]
    assert {key: body[key] for key in ALL_FIELDS} == ALL_FIELDS


def test_core_verbose_is_not_a_request_field() -> None:
    server = Server(ok())
    make_llm(server, verbose=True).invoke("hi")
    assert "verbose" not in server.bodies[0]


# --- streaming -------------------------------------------------------------

CHUNK_BASE = {
    "created": 1791710126,
    "id": "chatcmpl-pmligs5DcXIiejxeskv7YlviseIhyB8W",
    "model": "google/gemma-4-26B-A4B-it-qat-q4_0-gguf:Q4_0",
    "system_fingerprint": "b11429-d81235049",
    "object": "chat.completion.chunk",
}
STREAM_USAGE = {
    "completion_tokens": 5,
    "prompt_tokens": 32,
    "total_tokens": 37,
    "prompt_tokens_details": {"cached_tokens": 27},
}
STREAM_TIMINGS = {"cache_n": 27, "prompt_n": 5, "predicted_n": 5}


def delta(finish_reason: str | None = None, **fields: Any) -> dict[str, Any]:
    choice = {"finish_reason": finish_reason, "index": 0, "delta": fields}
    return {"choices": [choice], **CHUNK_BASE}


# Shaped like llama-server b11429's stream (probe of 2026-10-11).
EVENTS = [
    delta(role="assistant", content=None),
    delta(reasoning_content="17 x 23"),
    delta(reasoning_content=" = 391."),
    delta(content="39"),
    delta(content="1"),
    delta(finish_reason="stop"),
    {"choices": [], **CHUNK_BASE, "usage": STREAM_USAGE, "timings": STREAM_TIMINGS},
]


def sse(*events: dict[str, Any], tail: str = "data: [DONE]\n\n") -> httpx.Response:
    body = "".join(f"data: {json.dumps(event)}\n\n" for event in events) + tail
    headers = {"content-type": "text/event-stream"}
    return httpx.Response(200, content=body.encode(), headers=headers)


def merge(chunks: list[AIMessageChunk]) -> AIMessageChunk:
    merged = chunks[0]
    for chunk in chunks[1:]:
        merged += chunk
    return merged


def test_stream_request_body() -> None:
    server = Server(sse(*EVENTS))
    list(make_llm(server, max_tokens=300).stream("hi"))
    body = server.bodies[0]
    assert body["stream"] is True
    assert body["stream_options"] == {"include_usage": True}
    assert body["max_tokens"] == 300


def test_stream_chunks_and_merged_message() -> None:
    chunks = list(make_llm(Server(sse(*EVENTS))).stream("hi"))
    assert [c.content for c in chunks if c.content] == ["39", "1"]
    msg = merge(chunks)
    assert msg.content == "391"
    assert msg.additional_kwargs == {"reasoning_content": "17 x 23 = 391."}
    assert msg.usage_metadata == {
        "input_tokens": 32,
        "output_tokens": 5,
        "total_tokens": 37,
        "input_token_details": {"cache_read": 27},
        "output_token_details": {},
    }
    meta = msg.response_metadata
    # Each string appears once, so none is concatenated across chunks.
    assert meta["finish_reason"] == "stop"
    assert meta["model_name"] == CHUNK_BASE["model"]
    assert meta["system_fingerprint"] == CHUNK_BASE["system_fingerprint"]
    assert meta["model_provider"] == "llama-cpp"
    assert meta["timings"] == STREAM_TIMINGS


def test_stream_usage_can_be_turned_off() -> None:
    server = Server(sse(*EVENTS[:-1]), sse(*EVENTS[:-1]))
    list(make_llm(server, stream_usage=False).stream("hi"))
    list(make_llm(server).stream("hi", stream_usage=False))
    assert all("stream_options" not in body for body in server.bodies)
    assert all("stream_usage" not in body for body in server.bodies)


def test_pings_and_blank_lines_are_skipped() -> None:
    events = "".join(f"data: {json.dumps(event)}\n\n" for event in EVENTS)
    body = ": ping\n\n:\n\n" + events + "data: [DONE]\n\n"
    response = httpx.Response(200, content=body.encode())
    assert merge(list(make_llm(Server(response)).stream("hi"))).content == "391"


def test_streamed_tool_call_fragments_merge() -> None:
    def tool(**fields: Any) -> dict[str, Any]:
        return delta(tool_calls=[{"index": 0, **fields}])

    events = [
        delta(role="assistant", content=None),
        tool(id="call_1", type="function", function={"name": "get_weather"}),
        tool(function={"arguments": '{"city": '}),
        tool(function={"arguments": '"Paris"}'}),
        delta(finish_reason="tool_calls"),
    ]
    msg = merge(list(make_llm(Server(sse(*events))).stream("weather?")))
    assert msg.tool_calls == [
        {
            "name": "get_weather",
            "args": {"city": "Paris"},
            "id": "call_1",
            "type": "tool_call",
        }
    ]
    assert msg.response_metadata["finish_reason"] == "tool_calls"


def test_error_before_streaming_is_a_normal_http_error() -> None:
    server = Server(httpx.Response(400, json=OVERFLOW))
    with pytest.raises(LlamaContextOverflowError) as info:
        list(make_llm(server).stream("long prompt"))
    assert (info.value.n_prompt_tokens, info.value.n_ctx) == (5012, 4096)
    assert len(server.requests) == 1


def test_retry_before_streaming_starts(no_sleep: list[float]) -> None:
    server = Server(httpx.Response(503, text="loading"), sse(*EVENTS))
    assert merge(list(make_llm(server).stream("hi"))).content == "391"
    assert len(server.requests) == 2
    assert len(no_sleep) == 1


def test_error_event_mid_stream_is_raised() -> None:
    error = {"code": 500, "message": "decode failed", "type": "server_error"}
    # No [DONE] after an error event, as in llama-server.
    server = Server(sse(*EVENTS[:5], tail=f"data: {json.dumps({'error': error})}\n\n"))
    received: list[str] = []
    with pytest.raises(LlamaAPIError, match="decode failed"):
        received.extend(chunk.text for chunk in make_llm(server).stream("hi"))
    assert "".join(received) == "391"  # chunks before the error were delivered
    assert len(server.requests) == 1  # not retried


def test_streaming_field_routes_invoke_through_stream() -> None:
    server = Server(sse(*EVENTS))
    msg = make_llm(server, streaming=True).invoke("hi")
    assert server.bodies[0]["stream"] is True
    assert msg.content == "391"
    assert msg.additional_kwargs == {"reasoning_content": "17 x 23 = 391."}


# --- async -----------------------------------------------------------------


def test_ainvoke() -> None:
    server = Server(ok())
    msg = asyncio.run(make_llm(server).ainvoke("hi"))
    assert msg.content == "391"
    assert "stream" not in server.bodies[0]


def test_astream() -> None:
    async def collect(llm: ChatLlama) -> list[AIMessageChunk]:
        return [chunk async for chunk in llm.astream("hi")]

    msg = merge(asyncio.run(collect(make_llm(Server(sse(*EVENTS))))))
    assert msg.content == "391"
    assert msg.additional_kwargs == {"reasoning_content": "17 x 23 = 391."}
    assert msg.usage_metadata is not None
    assert msg.response_metadata["timings"] == STREAM_TIMINGS


def test_async_retry_uses_async_sleep(no_sleep: list[float]) -> None:
    server = Server(httpx.Response(500, text="boom"), ok())
    assert asyncio.run(make_llm(server).ainvoke("hi")).content == "391"
    assert len(no_sleep) == 1


def test_async_errors_are_mapped() -> None:
    server = Server(httpx.Response(400, json=OVERFLOW))
    with pytest.raises(LlamaContextOverflowError):
        asyncio.run(make_llm(server).ainvoke("long prompt"))


# --- structured output -----------------------------------------------------


class Age(BaseModel):
    """A person's age."""

    age: int = Field(description="The person's age in MONTHS, not years.")


class AgeDict(TypedDict):
    """A person's age."""

    age: Annotated[int, ..., "The person's age in MONTHS, not years."]


AGE_PROPERTY = {
    "description": "The person's age in MONTHS, not years.",
    "type": "integer",
}
AGE_PARAMETERS = {
    "type": "object",
    "properties": {"age": AGE_PROPERTY},
    "required": ["age"],
}


def reply_with(message: dict[str, Any]) -> httpx.Response:
    return ok({**REPLY, "choices": [{**REPLY["choices"][0], "message": message}]})


def tool_call_reply(arguments: str, name: str = "Age") -> httpx.Response:
    return reply_with(
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "2 years is 24 months.",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": name, "arguments": arguments},
                }
            ],
        }
    )


def content_reply(content: str) -> httpx.Response:
    return reply_with(
        {"role": "assistant", "content": content, "reasoning_content": "Thinking."}
    )


def test_function_calling_is_the_default() -> None:
    server = Server(tool_call_reply('{"age": 24}'))
    result = make_llm(server).with_structured_output(Age).invoke("Alice is 2.")
    assert result == Age(age=24)
    body = server.bodies[0]
    assert body["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "Age",
                "description": "A person's age.",
                "parameters": AGE_PARAMETERS,
            },
        }
    ]
    assert body["tool_choice"] == "required"  # a named-tool object becomes auto
    assert body["parallel_tool_calls"] is False
    assert "response_format" not in body
    assert "ls_structured_output_format" not in body  # core strips it


@pytest.mark.parametrize(
    "schema",
    [AgeDict, {"title": "AgeDict", **AGE_PARAMETERS}],
    ids=["typeddict", "json-schema"],
)
def test_function_calling_with_dict_schemas_returns_a_dict(schema: Any) -> None:
    server = Server(tool_call_reply('{"age": 24}', name="AgeDict"))
    result = make_llm(server).with_structured_output(schema).invoke("Alice is 2.")
    assert result == {"age": 24}
    assert server.bodies[0]["tools"][0]["function"]["name"] == "AgeDict"


def test_strict_is_sent_only_when_set() -> None:
    server = Server(
        tool_call_reply('{"age": 24}'),
        tool_call_reply('{"age": 24}'),
        content_reply('{"age": 2}'),
    )
    llm = make_llm(server)
    llm.with_structured_output(Age).invoke("x")
    llm.with_structured_output(Age, strict=True).invoke("x")
    llm.with_structured_output(Age, method="json_schema", strict=True).invoke("x")
    assert "strict" not in server.bodies[0]["tools"][0]["function"]
    assert server.bodies[1]["tools"][0]["function"]["strict"] is True
    assert server.bodies[2]["response_format"]["json_schema"]["strict"] is True


def test_json_schema_request_and_parse() -> None:
    server = Server(content_reply('{"age": 2}'))
    structured = make_llm(server).with_structured_output(Age, method="json_schema")
    assert structured.invoke("Alice is 2.") == Age(age=2)
    body = server.bodies[0]
    assert body["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "Age",
            "description": "A person's age.",
            "schema": AGE_PARAMETERS,
        },
    }
    assert "tools" not in body
    assert body["messages"] == [{"role": "user", "content": "Alice is 2."}]


def test_json_mode_without_schema_returns_a_dict() -> None:
    server = Server(content_reply('{"age": 2, "name": "Alice"}'))
    structured = make_llm(server).with_structured_output(method="json_mode")
    assert structured.invoke("Alice is 2.") == {"age": 2, "name": "Alice"}
    assert server.bodies[0]["response_format"] == {"type": "json_object"}


def test_extra_kwargs_are_sent() -> None:
    server = Server(content_reply('{"age": 2}'))
    llm = make_llm(server)
    llm.with_structured_output(Age, method="json_schema", temperature=0).invoke("x")
    assert server.bodies[0]["temperature"] == 0


def test_format_instructions_add_a_system_message() -> None:
    server = Server(content_reply('{"age": 24}'))
    structured = make_llm(server).with_structured_output(
        Age, method="json_schema", include_format_instructions=True
    )
    assert structured.invoke("Alice is 2.") == Age(age=24)
    instructions = PydanticOutputParser(pydantic_object=Age).get_format_instructions()
    assert "MONTHS" in instructions
    assert server.bodies[0]["messages"] == [
        {"role": "system", "content": instructions},
        {"role": "user", "content": "Alice is 2."},
    ]


def test_format_instructions_go_after_an_existing_system_message() -> None:
    server = Server(content_reply('{"age": 24}'))
    messages = [SystemMessage("Be brief."), HumanMessage("Alice is 2.")]
    structured = make_llm(server).with_structured_output(
        Age, method="json_mode", include_format_instructions=True
    )
    structured.invoke(messages)
    instructions = PydanticOutputParser(pydantic_object=Age).get_format_instructions()
    assert server.bodies[0]["messages"] == [
        {"role": "system", "content": f"Be brief.\n\n{instructions}"},
        {"role": "user", "content": "Alice is 2."},
    ]
    assert messages[0].content == "Be brief."  # the caller's messages are untouched


def test_format_instructions_for_a_dict_schema() -> None:
    server = Server(content_reply('{"age": 24}'))
    structured = make_llm(server).with_structured_output(
        AgeDict, method="json_schema", include_format_instructions=True
    )
    assert structured.invoke("Alice is 2.") == {"age": 24}
    reduced = {
        "description": "A person's age.",
        "properties": {"age": AGE_PROPERTY},
        "required": ["age"],
    }
    assert server.bodies[0]["messages"][0] == {
        "role": "system",
        "content": JSON_FORMAT_INSTRUCTIONS.format(schema=json.dumps(reduced)),
    }


@pytest.mark.parametrize(
    ("schema", "kwargs", "match"),
    [
        (Age, {"include_format_instructions": True}, "function_calling"),
        (None, {}, "schema is required"),
        (None, {"method": "json_schema"}, "schema is required"),
        (
            None,
            {"method": "json_mode", "include_format_instructions": True},
            "schema is required",
        ),
        (Age, {"method": "tools"}, "method must be"),
    ],
)
def test_invalid_arguments(schema: Any, kwargs: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        make_llm(Server()).with_structured_output(schema, **kwargs)


def test_include_raw_keeps_the_message_and_the_parsing_error() -> None:
    server = Server(content_reply("not json"))
    structured = make_llm(server).with_structured_output(
        Age, method="json_schema", include_raw=True
    )
    result = structured.invoke("Alice is 2.")
    assert isinstance(result, dict)
    assert isinstance(result["raw"], AIMessage)
    assert result["raw"].additional_kwargs["reasoning_content"] == "Thinking."
    assert result["parsed"] is None
    assert isinstance(result["parsing_error"], OutputParserException)


def test_include_raw_on_success() -> None:
    server = Server(tool_call_reply('{"age": 24}'))
    structured = make_llm(server).with_structured_output(Age, include_raw=True)
    result = structured.invoke("Alice is 2.")
    assert isinstance(result, dict)
    assert result["parsed"] == Age(age=24)
    assert result["parsing_error"] is None


def test_structured_ainvoke() -> None:
    server = Server(content_reply('{"age": 2}'))
    structured = make_llm(server).with_structured_output(
        Age, method="json_schema", include_format_instructions=True
    )
    assert asyncio.run(structured.ainvoke("Alice is 2.")) == Age(age=2)
    assert server.bodies[0]["messages"][0]["role"] == "system"


# --- tools -----------------------------------------------------------------


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"18C and sunny in {city}"


@tool
def get_time(city: str) -> str:
    """Get the local time in a city."""
    return "09:00"


WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
            "type": "object",
        },
    },
}


def weather_call_reply(call_id: str = "call_1") -> httpx.Response:
    return reply_with(
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city":"Paris"}',
                    },
                }
            ],
        }
    )


def test_bind_tools_sends_openai_tools_only() -> None:
    server = Server(weather_call_reply())
    msg = make_llm(server).bind_tools([get_weather]).invoke("Weather in Paris?")
    assert msg.tool_calls[0]["args"] == {"city": "Paris"}
    body = server.bodies[0]
    assert body["tools"] == [WEATHER_TOOL]
    assert "tool_choice" not in body
    assert "parallel_tool_calls" not in body


@pytest.mark.parametrize(
    ("tool_choice", "sent"),
    [
        ("auto", "auto"),
        ("none", "none"),
        ("required", "required"),
        ("any", "required"),
        (True, "required"),
        ("get_weather", "required"),  # the server can't force one tool
        ({"type": "function", "function": {"name": "get_time"}}, "required"),
    ],
)
def test_tool_choice_is_sent_as_a_server_string(tool_choice: Any, sent: str) -> None:
    server = Server(weather_call_reply())
    llm = make_llm(server).bind_tools([get_weather, get_time], tool_choice=tool_choice)
    llm.invoke("Weather in Paris?")
    assert server.bodies[0]["tool_choice"] == sent
    assert server.bodies[0]["tools"][0] == WEATHER_TOOL  # all tools are still sent
    assert len(server.bodies[0]["tools"]) == 2


@pytest.mark.parametrize("tool_choice", [None, False])
def test_no_tool_choice_is_left_out(tool_choice: Any) -> None:
    server = Server(weather_call_reply())
    make_llm(server).bind_tools([get_weather], tool_choice=tool_choice).invoke("x")
    assert "tool_choice" not in server.bodies[0]


@pytest.mark.parametrize(
    "tool_choice",
    [
        "get_time",  # not bound
        {"type": "function", "function": {"name": "get_time"}},
        {"type": "function"},
        "always",
    ],
)
def test_invalid_tool_choice(tool_choice: Any) -> None:
    with pytest.raises(ValueError, match="tool_choice must be"):
        make_llm(Server()).bind_tools([get_weather], tool_choice=tool_choice)


def test_strict_parallel_tool_calls_and_kwargs_are_sent() -> None:
    server = Server(weather_call_reply())
    llm = make_llm(server).bind_tools(
        [get_weather], strict=True, parallel_tool_calls=False, temperature=0
    )
    llm.invoke("x")
    body = server.bodies[0]
    assert body["tools"][0]["function"]["strict"] is True
    assert body["parallel_tool_calls"] is False
    assert body["temperature"] == 0


def test_agent_loop_sends_the_tool_call_and_result_back() -> None:
    final = content_reply("It's 18C and sunny in Paris.")
    server = Server(weather_call_reply("call_7"), final)
    llm = make_llm(server).bind_tools([get_weather])
    messages: list[BaseMessage] = [HumanMessage("Weather in Paris?")]
    while True:
        ai = llm.invoke(messages)
        messages.append(ai)
        if not ai.tool_calls:
            break
        messages.extend(get_weather.invoke(call) for call in ai.tool_calls)

    assert messages[-1].content == "It's 18C and sunny in Paris."
    assert len(server.requests) == 2
    assert server.bodies[1]["messages"] == [
        {"role": "user", "content": "Weather in Paris?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "id": "call_7",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city": "Paris"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "name": "get_weather",
            "tool_call_id": "call_7",
            "content": "18C and sunny in Paris",
        },
    ]
    assert server.bodies[1]["tools"] == [WEATHER_TOOL]


# --- v1 content blocks while streaming ---------------------------------------


def tool_call_delta(
    index: int, *, call_id: str | None = None, name: str | None = None, args: str = ""
) -> dict[str, Any]:
    function: dict[str, Any] = {"arguments": args}
    if name is not None:
        function["name"] = name
    call: dict[str, Any] = {"index": index, "function": function}
    if call_id is not None:
        call |= {"id": call_id, "type": "function"}
    return delta(tool_calls=[call])


def stream_v1(*events: dict[str, Any]) -> AIMessageChunk:
    llm = make_llm(Server(sse(*events)), output_version="v1")
    return merge(list(llm.stream("hi")))


MAGIC_CALL = {
    "type": "tool_call",
    "name": "magic_function",
    "args": {"input": 3},
    "id": "call_1",
}


def test_v1_stream_keeps_a_tool_call_after_reasoning() -> None:
    msg = stream_v1(
        delta(role="assistant", content=None),
        delta(reasoning_content="Use"),
        delta(reasoning_content=" the tool."),
        tool_call_delta(0, call_id="call_1", name="magic_function"),
        tool_call_delta(0, args='{"input": 3}'),
        delta(finish_reason="tool_calls"),
    )
    assert msg.content == [
        {"type": "reasoning", "reasoning": "Use the tool.", "index": 0},
        MAGIC_CALL,
    ]
    assert msg.content_blocks == msg.content


def test_v1_stream_keeps_a_tool_call_after_text() -> None:
    msg = stream_v1(
        delta(content="Let me"),
        delta(content=" check."),
        tool_call_delta(0, call_id="call_1", name="magic_function"),
        tool_call_delta(0, args='{"input": 3}'),
        delta(finish_reason="tool_calls"),
    )
    assert msg.content == [
        {"type": "text", "text": "Let me check.", "index": 0},
        MAGIC_CALL,
    ]


def test_v1_stream_keeps_parallel_tool_calls_apart() -> None:
    msg = stream_v1(
        delta(reasoning_content="Both."),
        tool_call_delta(0, call_id="call_1", name="get_weather"),
        tool_call_delta(0, args='{"city": "Paris"}'),
        tool_call_delta(1, call_id="call_2", name="get_time"),
        tool_call_delta(1, args='{"city": "Paris"}'),
        delta(finish_reason="tool_calls"),
    )
    assert [block["type"] for block in msg.content_blocks] == [
        "reasoning",
        "tool_call",
        "tool_call",
    ]
    assert [call["name"] for call in msg.tool_calls] == ["get_weather", "get_time"]


def test_v1_astream_keeps_a_tool_call_after_reasoning() -> None:
    llm = make_llm(
        Server(
            sse(
                delta(reasoning_content="Use the tool."),
                tool_call_delta(0, call_id="call_1", name="magic_function"),
                tool_call_delta(0, args='{"input": 3}'),
                delta(finish_reason="tool_calls"),
            )
        ),
        output_version="v1",
    )

    async def collect() -> list[AIMessageChunk]:
        return [chunk async for chunk in llm.astream("hi")]

    msg = merge(asyncio.run(collect()))
    assert [block["type"] for block in msg.content_blocks] == ["reasoning", "tool_call"]


def test_v0_content_blocks_are_core_defaults() -> None:
    msg = make_llm(Server(tool_call_reply('{"age": 24}'))).invoke("x")
    assert msg.response_metadata["model_provider"] == "llama-cpp"
    assert msg.content_blocks == [
        {"type": "reasoning", "reasoning": "2 years is 24 months."},
        {"type": "tool_call", "id": "call_1", "name": "Age", "args": {"age": 24}},
    ]


# --- review fixes ----------------------------------------------------------


def test_invoke_with_stream_kwarg_streams() -> None:
    server = Server(sse(*EVENTS))
    assert make_llm(server).invoke("hi", stream=True).content == "391"
    assert server.bodies[0]["stream"] is True


def test_ainvoke_with_stream_kwarg_streams() -> None:
    server = Server(sse(*EVENTS))
    msg = asyncio.run(make_llm(server).ainvoke("hi", stream=True))
    assert msg.content == "391"
    assert server.bodies[0]["stream"] is True


def test_stream_false_kwarg_is_not_sent() -> None:
    server = Server(ok())
    make_llm(server).invoke("hi", stream=False)
    assert "stream" not in server.bodies[0]


def test_async_client_is_replaced_on_a_new_event_loop() -> None:
    llm = ChatLlama(model="gemma", base_url=BASE_URL)

    async def client_twice() -> tuple[httpx.AsyncClient, httpx.AsyncClient]:
        return llm._get_async_client(), llm._get_async_client()

    first, same = asyncio.run(client_twice())
    second, _ = asyncio.run(client_twice())
    assert first is same  # reused within one loop
    assert second is not first  # not carried over to a closed loop


class BrokenStream(httpx.SyncByteStream, httpx.AsyncByteStream):
    """One chunk, then the connection fails."""

    first = f"data: {json.dumps(delta(content='39'))}\n\n".encode()

    def __iter__(self) -> Iterator[bytes]:
        yield self.first
        msg = "read timed out"
        raise httpx.ReadTimeout(msg)

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield self.first
        msg = "read timed out"
        raise httpx.ReadTimeout(msg)


def test_transport_error_mid_stream_is_wrapped() -> None:
    llm = make_llm(Server(httpx.Response(200, stream=BrokenStream())))
    with pytest.raises(LlamaTimeoutError) as info:
        list(llm.stream("hi"))
    assert isinstance(info.value, ModelTimeoutError)
    assert isinstance(info.value.__cause__, httpx.ReadTimeout)


def test_transport_error_mid_astream_is_wrapped() -> None:
    llm = make_llm(Server(httpx.Response(200, stream=BrokenStream())))

    async def collect() -> list[AIMessageChunk]:
        return [chunk async for chunk in llm.astream("hi")]

    with pytest.raises(LlamaTimeoutError):
        asyncio.run(collect())


def test_streaming_with_n_above_1_is_refused() -> None:
    with pytest.raises(ValueError, match="n must be 1 when streaming"):
        list(make_llm(Server(), n=2).stream("hi"))
    per_call: dict[str, Any] = {"n": 3}
    with pytest.raises(ValueError, match="n must be 1 when streaming"):
        list(make_llm(Server()).stream("hi", **per_call))


def test_content_blocks_do_not_modify_the_message() -> None:
    chunk = AIMessageChunk(
        content=[{"type": "text", "text": "hi"}],
        tool_call_chunks=[
            {
                "name": "f",
                "args": "{}",
                "id": "c1",
                "index": 0,
                "type": "tool_call_chunk",
            }
        ],
        response_metadata={"model_provider": "llama-cpp"},
    )
    blocks = chunk.content_blocks
    assert chunk.content == [{"type": "text", "text": "hi"}]
    assert blocks[-1].get("index") == "lc_tc_0"


def test_v1_stream_keeps_text_after_a_tool_call_in_order() -> None:
    msg = stream_v1(
        delta(content="Checking."),
        tool_call_delta(0, call_id="call_1", name="magic_function"),
        tool_call_delta(0, args='{"input": 3}'),
        delta(content="Done."),
        delta(finish_reason="tool_calls"),
    )
    assert [block["type"] for block in msg.content_blocks] == [
        "text",
        "tool_call",
        "text",
    ]
    texts = [b.get("text") for b in msg.content_blocks if b["type"] == "text"]
    assert texts == ["Checking.", "Done."]


def test_v1_stream_text_and_tool_call_in_one_delta() -> None:
    first = tool_call_delta(0, call_id="call_1", name="magic_function", args='{"in')
    first["choices"][0]["delta"]["content"] = "Calling."
    msg = stream_v1(
        first,
        tool_call_delta(0, args='put": 3}'),
        delta(finish_reason="tool_calls"),
    )
    assert [block["type"] for block in msg.content_blocks] == ["text", "tool_call"]
    assert msg.content_blocks[1] == MAGIC_CALL


def test_bind_tools_with_a_non_function_tool() -> None:
    server = Server(weather_call_reply())
    tools = [get_weather, {"type": "web_search"}]
    make_llm(server).bind_tools(tools, tool_choice="get_weather").invoke("x")
    assert server.bodies[0]["tool_choice"] == "required"
    assert server.bodies[0]["tools"][1] == {"type": "web_search"}


def test_empty_tool_choice_is_left_out() -> None:
    server = Server(weather_call_reply())
    make_llm(server).bind_tools([get_weather], tool_choice="").invoke("x")
    assert "tool_choice" not in server.bodies[0]


INNER_SCHEMA = {"name": "Age", "schema": AGE_PARAMETERS, "strict": True}


@pytest.mark.parametrize(
    "schema",
    [INNER_SCHEMA, {"type": "json_schema", "json_schema": INNER_SCHEMA}],
    ids=["name-schema", "response-format"],
)
def test_json_schema_accepts_openai_response_format_shapes(schema: Any) -> None:
    server = Server(content_reply('{"age": 24}'))
    structured = make_llm(server).with_structured_output(
        schema, method="json_schema", include_format_instructions=True
    )
    assert structured.invoke("Alice is 2.") == {"age": 24}
    body = server.bodies[0]
    assert body["response_format"] == {
        "type": "json_schema",
        "json_schema": INNER_SCHEMA,
    }
    reduced = {"properties": {"age": AGE_PROPERTY}, "required": ["age"]}
    assert body["messages"][0]["content"] == JSON_FORMAT_INSTRUCTIONS.format(
        schema=json.dumps(reduced)
    )


def test_structured_output_kwargs_override_defaults() -> None:
    server = Server(tool_call_reply('{"age": 24}'))
    llm = make_llm(server)
    llm.with_structured_output(
        Age, parallel_tool_calls=True, tool_choice="auto"
    ).invoke("x")
    assert server.bodies[0]["tool_choice"] == "auto"
    assert server.bodies[0]["parallel_tool_calls"] is True


def test_structured_output_rejects_extra_tools() -> None:
    with pytest.raises(ValueError, match="only tool"):
        make_llm(Server()).with_structured_output(Age, tools=[get_weather])


def test_negative_max_retries_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ChatLlama(model="gemma", base_url=BASE_URL, max_retries=-1)
