"""ChatLlama: a LangChain chat model for llama.cpp's llama-server behind llama-swap."""

import asyncio
import json
import random
import time
from collections.abc import AsyncIterator, Iterator, Mapping
from typing import Any, Literal

import httpx
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.exceptions import ModelError, OutputParserException
from langchain_core.language_models import BaseChatModel, LangSmithParams
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    InvalidToolCall,
    ToolCall,
    convert_to_openai_messages,
)
from langchain_core.messages.ai import (
    InputTokenDetails,
    OutputTokenDetails,
    UsageMetadata,
)
from langchain_core.messages.tool import ToolCallChunk
from langchain_core.messages.tool import tool_call_chunk as create_tool_call_chunk
from langchain_core.output_parsers.openai_tools import (
    make_invalid_tool_call,
    parse_tool_call,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.utils import from_env
from pydantic import Field, PrivateAttr

from langchain_llama_cpp.exceptions import (
    error_from_response,
    error_from_stream_event,
    error_from_transport,
)

CHAT_COMPLETIONS = "/v1/chat/completions"
MODEL_PROVIDER = "llama-cpp"

# Backoff between retries, as in the openai and groq SDKs.
INITIAL_RETRY_DELAY = 0.5
MAX_RETRY_DELAY = 8.0
MAX_RETRY_AFTER = 60.0


class ChatLlama(BaseChatModel):
    """Chat model for llama.cpp's `llama-server`, reached through llama-swap.

    Example:
        ```python
        from langchain_llama_cpp import ChatLlama

        llm = ChatLlama(model="gemma", max_tokens=300)  # base_url from LLAMA_BASE_URL
        msg = llm.invoke("What is 17 * 23?")
        msg.content  # "391"
        msg.additional_kwargs["reasoning_content"]  # the model's thinking, if any
        msg.response_metadata["timings"]  # llama-server's own timings
        ```
    """

    model: str
    """llama-swap entry to use, e.g. `"gemma"`."""

    base_url: str = Field(
        default_factory=from_env(
            "LLAMA_BASE_URL",
            error_message="Pass base_url= or set LLAMA_BASE_URL.",
        )
    )
    """Server root, e.g. `http://localhost:8080`. Defaults to `LLAMA_BASE_URL`."""

    # --- Request fields -----------------------------------------------------
    # Every field accepted by /v1/chat/completions in llama.cpp b11429
    # (tools/server/server-common.cpp oaicompat_chat_params_parse,
    # server-schema.cpp make_llama_cmpl_schema, server-context.cpp id_slot).
    # Each is sent under its own name; None (the default) leaves it out, so the
    # server's per-model default applies (see `get_server_defaults`).

    # Length and stopping
    max_tokens: int | None = None
    """Maximum tokens to generate (-1 = no limit). Thinking tokens count too."""

    stop: list[str] | None = None
    """Stop strings. Generation ends at the first one, which is not returned."""

    ignore_eos: bool | None = None
    """Ignore the end-of-sequence token and keep generating."""

    t_max_predict_ms: int | None = None
    """Stop generating after this many ms from the first token, at a newline."""

    n_indent: int | None = None
    """Minimum line indentation, in spaces, for generated text (code completion)."""

    n: int | None = None
    """Number of completions to generate (1 to the server's slot count)."""

    # Sampling
    temperature: float | None = None
    """Randomness (0 = greedy)."""

    dynatemp_range: float | None = None
    """Dynamic temperature: final value in `temperature ± range` (0 = off)."""

    dynatemp_exponent: float | None = None
    """Dynamic temperature: how entropy maps to temperature."""

    top_k: int | None = None
    """Keep only the k most likely tokens (0 = off)."""

    top_p: float | None = None
    """Keep the smallest set of tokens with this total probability (1.0 = off)."""

    min_p: float | None = None
    """Drop tokens below this fraction of the top token's probability (0 = off)."""

    typical_p: float | None = None
    """Locally typical sampling (1.0 = off)."""

    top_n_sigma: float | None = None
    """Keep tokens within n standard deviations of the top logit (< 0 = off)."""

    xtc_probability: float | None = None
    """Chance that the XTC sampler removes top tokens (0 = off)."""

    xtc_threshold: float | None = None
    """Minimum probability for XTC removal (> 0.5 turns XTC off)."""

    mirostat: Literal[0, 1, 2] | None = None
    """Mirostat sampling: 0 = off, 1 = Mirostat, 2 = Mirostat 2.0."""

    mirostat_tau: float | None = None
    """Mirostat target entropy (tau)."""

    mirostat_eta: float | None = None
    """Mirostat learning rate (eta)."""

    adaptive_target: float | None = None
    """Adaptive sampling target entropy, 0.0 to 1.0 (negative = off)."""

    adaptive_decay: float | None = None
    """Adaptive sampling EMA decay, 0.0 to 0.99."""

    min_keep: int | None = None
    """Make every sampler keep at least this many tokens."""

    samplers: list[str] | str | None = None
    """Sampler order: names (`["top_k", "min_p", ...]`) or a string of sampler chars."""

    seed: int | None = None
    """Random seed (-1 = random)."""

    backend_sampling: bool | None = None
    """Sample on the backend instead of in llama.cpp's sampler."""

    # Repetition penalties
    repeat_penalty: float | None = None
    """Penalty for repeated tokens (1.0 = off)."""

    repeat_last_n: int | None = None
    """How many recent tokens the repeat penalty looks at (0 = off)."""

    frequency_penalty: float | None = None
    """Penalty that grows with how often a token appeared (0 = off)."""

    presence_penalty: float | None = None
    """Penalty for any token that already appeared (0 = off)."""

    dry_multiplier: float | None = None
    """DRY repetition penalty multiplier (0 = off)."""

    dry_base: float | None = None
    """DRY penalty base (>= 1.0; smaller values fall back to the server default)."""

    dry_allowed_length: int | None = None
    """Repeats longer than this get the DRY penalty."""

    dry_penalty_last_n: int | None = None
    """How many tokens DRY scans for repeats (0 = off)."""

    dry_sequence_breakers: list[str] | None = None
    """Strings that reset DRY's repeat matching (non-empty list)."""

    logit_bias: dict[str, float | bool] | list[list[Any]] | None = None
    """Token bias: `{"token_or_id": bias}` or `[[token_or_id, bias], ...]`.

    A bias of `False` bans the token.
    """

    # Constrained output (with_structured_output builds on these)
    grammar: str | None = None
    """A GBNF grammar the output must follow. Can't be combined with tools."""

    json_schema: dict[str, Any] | None = None
    """A JSON schema the output must follow (llama.cpp's own field)."""

    # Chat template and reasoning
    chat_template_kwargs: dict[str, Any] | None = None
    """Variables for the chat template, e.g. `{"enable_thinking": False}`."""

    reasoning_format: str | None = None
    """Where thinking goes: `"auto"`/`"deepseek"` → `reasoning_content`.

    `"none"` leaves it in the content text; `"deepseek-legacy"` does both.
    """

    reasoning_effort: str | None = None
    """OpenAI-style effort (`"low"`, `"medium"`, `"high"`); `"none"` turns thinking off.

    Passed to the template as `reasoning_effort` (gpt-oss).
    """

    reasoning_budget_tokens: int | None = None
    """Cap on thinking tokens (-1 = no cap); then the end-of-thinking tag is forced."""

    reasoning_budget_message: str | None = None
    """Text inserted before the forced end-of-thinking tag."""

    reasoning_control: bool | None = None
    """Create the reasoning-budget sampler so thinking can be ended at runtime."""

    add_generation_prompt: bool | None = None
    """Whether the template appends the assistant turn's opening (default true)."""

    continue_final_message: bool | str | None = None
    """Continue the last assistant message instead of starting a new turn."""

    echo: bool | None = None
    """Echo the input tokens in the output."""

    # Probabilities and debugging output
    logprobs: bool | None = None
    """Return token probabilities (llama.cpp's format, not exactly OpenAI's)."""

    top_logprobs: int | None = None
    """How many top tokens to return per position (needs `logprobs`; default 20)."""

    n_probs: int | None = None
    """llama.cpp's native form of `top_logprobs` (no `logprobs` flag needed)."""

    post_sampling_probs: bool | None = None
    """Report probabilities after the sampler chain instead of before."""

    return_tokens: bool | None = None
    """Also return the generated token ids."""

    timings_per_token: bool | None = None
    """Include timing information in every response chunk."""

    response_fields: list[str] | None = None
    """Only return these reply fields. Can remove fields ChatLlama reads."""

    # Server resources and caching
    cache_prompt: bool | None = None
    """Reuse the KV cache for a shared prompt prefix (server default: on)."""

    n_cache_reuse: int | None = None
    """Minimum chunk size to reuse from the cache via KV shifting."""

    n_keep: int | None = None
    """Prompt tokens to keep when the context is shifted (-1 = all)."""

    n_discard: int | None = None
    """Tokens after `n_keep` that a context shift may drop (0 = half)."""

    id_slot: int | None = None
    """Run on this server slot (-1 = any free slot)."""

    lora: list[dict[str, Any]] | None = None
    """LoRA adapters for this request: `[{"id": 0, "scale": 1.0}, ...]`."""

    # Streaming only (used from milestone 2)
    return_progress: bool | None = None
    """Send prompt-processing progress events while streaming."""

    sse_ping_interval: int | None = None
    """Seconds between keep-alive pings on a silent stream (-1 = off)."""

    # --- Client settings (not sent) ----------------------------------------

    timeout: float | httpx.Timeout | None = None
    """Request timeout in seconds. `None` (the default) means no timeout."""

    max_retries: int = 2
    """Retries for connection errors, timeouts, 429 and 5xx. 0 turns them off."""

    streaming: bool = False
    """Route `invoke` through streaming (core reads this field, as for ChatOpenAI)."""

    stream_usage: bool = True
    """Ask for the final usage chunk (`stream_options.include_usage`) when streaming.

    llama-server supports it, so it's on by default; the chunk also carries
    `timings`. Can be overridden per call with `stream_usage=`.
    """

    http_client: httpx.Client | None = Field(default=None, exclude=True)
    """Your own `httpx.Client`. If given, `timeout` is not applied to it."""

    http_async_client: httpx.AsyncClient | None = Field(default=None, exclude=True)
    """Your own `httpx.AsyncClient` for async calls. `timeout` isn't applied to it."""

    _client: httpx.Client | None = PrivateAttr(default=None)
    _async_client: httpx.AsyncClient | None = PrivateAttr(default=None)

    @property
    def _llm_type(self) -> str:
        return "llama-cpp-chat"

    @property
    def _params(self) -> dict[str, Any]:
        """Request fields set on the model; fields left at `None` are left out."""
        params = {name: getattr(self, name) for name in request_field_names()}
        return {key: value for key, value in params.items() if value is not None}

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model": self.model, "base_url": self.base_url, **self._params}

    def _get_ls_params(
        self, stop: list[str] | None = None, **kwargs: Any
    ) -> LangSmithParams:
        params = super()._get_ls_params(stop=stop, **kwargs)
        params["ls_provider"] = MODEL_PROVIDER
        params["ls_model_name"] = kwargs.get("model", self.model)  # per-call override
        if self.temperature is not None:
            params["ls_temperature"] = self.temperature
        if self.max_tokens is not None:
            params["ls_max_tokens"] = self.max_tokens
        if ls_stop := stop or self.stop:
            params["ls_stop"] = ls_stop
        return params

    def _combine_llm_outputs(
        self, llm_outputs: list[dict[str, Any] | None]
    ) -> dict[str, Any]:
        """Sum `token_usage` across calls, as ChatOpenAI does (used by `generate`)."""
        token_usage: dict[str, Any] = {}
        system_fingerprint = None
        for output in llm_outputs:
            if output is None:
                continue
            _add_usage(token_usage, output.get("token_usage") or {})
            system_fingerprint = system_fingerprint or output.get("system_fingerprint")
        combined: dict[str, Any] = {
            "token_usage": token_usage,
            "model_name": self.model,
        }
        if system_fingerprint:
            combined["system_fingerprint"] = system_fingerprint
        return combined

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,  # noqa: ARG002
        **kwargs: Any,
    ) -> ChatResult:
        payload = self._build_payload(messages, stop, **kwargs)
        response = self._request("POST", CHAT_COMPLETIONS, payload)
        return _create_chat_result(response.json())

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,  # noqa: ARG002
        **kwargs: Any,
    ) -> ChatResult:
        payload = self._build_payload(messages, stop, **kwargs)
        response = await self._arequest("POST", CHAT_COMPLETIONS, payload)
        return _create_chat_result(response.json())

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,  # noqa: ARG002
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        # Core calls run_manager.on_llm_new_token for each chunk itself.
        payload = self._build_payload(messages, stop, stream=True, **kwargs)
        request = self._build_request("POST", CHAT_COMPLETIONS, payload)
        response = self._send(self._get_client(), request, stream=True)
        try:
            for line in response.iter_lines():
                data = _sse_data(line)
                if data is None:
                    continue
                if data == "[DONE]":
                    break
                if chunk := _convert_chunk(_sse_event(data, response)):
                    yield chunk
        finally:
            response.close()  # also when the caller stops early

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,  # noqa: ARG002
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        payload = self._build_payload(messages, stop, stream=True, **kwargs)
        request = self._build_request("POST", CHAT_COMPLETIONS, payload)
        client = self._get_async_client()
        response = await self._asend(client, request, stream=True)
        try:
            async for line in response.aiter_lines():
                data = _sse_data(line)
                if data is None:
                    continue
                if data == "[DONE]":
                    break
                if chunk := _convert_chunk(_sse_event(data, response)):
                    yield chunk
        finally:
            await response.aclose()

    def get_server_defaults(self) -> dict[str, Any]:
        """Fetch the sampling defaults the server uses for this model.

        These are what applies to every field you leave at `None`. Reads
        `GET /upstream/<model>/props` on each call (which loads the model in
        llama-swap). Nothing is stored on the model or sent with requests.

        Returns:
            `default_generation_settings.params` from `/props`, plus `n_ctx`.
            Floats are rounded to 7 significant digits, since the server stores
            them as 32-bit (it reports `top_p` 0.95 as 0.949999988079071).

        Example:
            ```python
            ChatLlama(model="gemma").get_server_defaults()
            # {"temperature": 1.0, "top_k": 64, "top_p": 0.95, "max_tokens": -1, ...}
            ```
        """
        response = self._request("GET", f"/upstream/{self.model}/props")
        settings: dict[str, Any] = response.json()["default_generation_settings"]
        defaults = {
            key: _round_float32(value)
            for key, value in settings.get("params", {}).items()
        }
        defaults["n_ctx"] = settings.get("n_ctx")
        return defaults

    def _build_payload(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None,
        *,
        stream: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        stream_usage = kwargs.pop("stream_usage", self.stream_usage)
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [_convert_message(message) for message in messages],
            **self._params,
            **kwargs,
        }
        if stop is not None:
            payload["stop"] = stop
        if stream:
            payload["stream"] = True
            if stream_usage:
                payload["stream_options"] = {"include_usage": True}
        return payload

    def _build_request(
        self, method: str, path: str, payload: dict[str, Any] | None = None
    ) -> httpx.Request:
        url = self.base_url.rstrip("/") + path
        return httpx.Request(method, url, json=payload)

    def _get_client(self) -> httpx.Client:
        if self.http_client is not None:
            return self.http_client
        if self._client is None:
            # httpx's own default is a 5 s timeout, so always pass ours.
            self._client = httpx.Client(timeout=self.timeout)
        return self._client

    def _get_async_client(self) -> httpx.AsyncClient:
        if self.http_async_client is not None:
            return self.http_async_client
        if self._async_client is None:
            self._async_client = httpx.AsyncClient(timeout=self.timeout)
        return self._async_client

    def _request(
        self, method: str, path: str, payload: dict[str, Any] | None = None
    ) -> httpx.Response:
        request = self._build_request(method, path, payload)
        return self._send(self._get_client(), request, stream=False)

    async def _arequest(
        self, method: str, path: str, payload: dict[str, Any] | None = None
    ) -> httpx.Response:
        request = self._build_request(method, path, payload)
        return await self._asend(self._get_async_client(), request, stream=False)

    def _send(
        self, client: httpx.Client, request: httpx.Request, *, stream: bool
    ) -> httpx.Response:
        """Send, retrying on errors core marks `is_retryable`.

        With `stream=True` this returns once the headers arrive, so retries only
        cover the request itself; a failure mid-stream is raised, not retried.
        """
        for attempt in range(self.max_retries + 1):
            try:
                response = client.send(request, stream=stream)
            except httpx.TransportError as exc:
                error, cause, retry_after = error_from_transport(exc), exc, None
            else:
                if response.is_success:
                    return response
                response.read()  # a streamed error body isn't read yet
                response.close()
                error, cause = error_from_response(response), None
                retry_after = _retry_after(response)
            if not _is_retryable(error) or attempt == self.max_retries:
                raise error from cause
            time.sleep(_retry_delay(attempt, retry_after))
        msg = "unreachable"  # the loop always returns or raises
        raise AssertionError(msg)

    async def _asend(
        self, client: httpx.AsyncClient, request: httpx.Request, *, stream: bool
    ) -> httpx.Response:
        """Async twin of `_send`."""
        for attempt in range(self.max_retries + 1):
            try:
                response = await client.send(request, stream=stream)
            except httpx.TransportError as exc:
                error, cause, retry_after = error_from_transport(exc), exc, None
            else:
                if response.is_success:
                    return response
                await response.aread()
                await response.aclose()
                error, cause = error_from_response(response), None
                retry_after = _retry_after(response)
            if not _is_retryable(error) or attempt == self.max_retries:
                raise error from cause
            await asyncio.sleep(_retry_delay(attempt, retry_after))
        msg = "unreachable"
        raise AssertionError(msg)


# ChatLlama fields that configure the client rather than the request.
_CLIENT_FIELDS = frozenset(
    {
        "model",
        "base_url",
        "timeout",
        "max_retries",
        "streaming",
        "stream_usage",
        "http_client",
        "http_async_client",
    }
)


def _is_retryable(error: Exception) -> bool:
    return isinstance(error, ModelError) and error.is_retryable


def _sse_data(line: str) -> str | None:
    """The payload of an SSE `data:` line; None for anything else.

    Anything else is a blank separator, a `:` keep-alive ping, or another field.
    """
    if not line.startswith("data:"):
        return None
    return line[len("data:") :].strip()


def _sse_event(data: str, response: httpx.Response) -> dict[str, Any]:
    """Decode one event. An `error` event (failure mid-stream) is raised."""
    event: dict[str, Any] = json.loads(data)
    if "error" in event:
        raise error_from_stream_event(event["error"], response)
    return event


def request_field_names() -> list[str]:
    """Names of ChatLlama's request fields, in declaration order.

    Everything ChatLlama declares itself, minus the client settings. `model` is
    sent too, but always (it's required), so it isn't listed here.
    """
    inherited = BaseChatModel.model_fields
    return [
        name
        for name in ChatLlama.model_fields
        if name not in inherited and name not in _CLIENT_FIELDS
    ]


def _convert_message(message: BaseMessage) -> dict[str, Any]:
    """Convert a message to the OpenAI shape llama-server expects."""
    converted = convert_to_openai_messages(message)
    if isinstance(message, AIMessage):
        # Core leaves v1 reasoning blocks in `content`; send text as content and
        # reasoning as `reasoning_content` (M1-c: the chat template decides).
        converted["content"] = message.text
        reasoning = "".join(
            block.get("reasoning", "")
            for block in message.content_blocks
            if block["type"] == "reasoning"
        )
        if reasoning:
            converted["reasoning_content"] = reasoning
    return converted


def _create_chat_result(data: Mapping[str, Any]) -> ChatResult:
    usage = data.get("usage")
    response_metadata = {
        key: value
        for key, value in {
            "model_name": data.get("model"),
            "system_fingerprint": data.get("system_fingerprint"),
            "id": data.get("id"),
            "timings": data.get("timings"),
            "model_provider": MODEL_PROVIDER,
        }.items()
        if value is not None
    }
    generations: list[ChatGeneration] = []
    choices: list[dict[str, Any]] = data.get("choices") or []
    for choice in choices:
        message = _convert_response_message(choice["message"])
        message.response_metadata = dict(response_metadata)
        if usage:
            message.usage_metadata = _create_usage_metadata(usage)
        generation_info: dict[str, Any] = {"finish_reason": choice.get("finish_reason")}
        if choice.get("logprobs"):
            generation_info["logprobs"] = choice["logprobs"]
        generations.append(
            ChatGeneration(message=message, generation_info=generation_info)
        )
    llm_output = {
        "token_usage": usage,
        "model_name": data.get("model"),
        "system_fingerprint": data.get("system_fingerprint"),
        "id": data.get("id"),
    }
    return ChatResult(generations=generations, llm_output=llm_output)


def _convert_response_message(message: Mapping[str, Any]) -> AIMessage:
    additional_kwargs: dict[str, Any] = {}
    if reasoning := message.get("reasoning_content"):
        additional_kwargs["reasoning_content"] = reasoning
    tool_calls: list[ToolCall] = []
    invalid_tool_calls: list[InvalidToolCall] = []
    raw_tool_calls: list[dict[str, Any]] = message.get("tool_calls") or []
    for raw in raw_tool_calls:
        try:
            parsed = parse_tool_call(raw, return_id=True)
        except OutputParserException as exc:  # arguments aren't valid JSON
            invalid_tool_calls.append(make_invalid_tool_call(raw, str(exc)))
            continue
        if parsed is not None:
            tool_calls.append(
                ToolCall(
                    name=parsed["name"],
                    args=parsed["args"],
                    id=parsed.get("id"),
                    type="tool_call",
                )
            )
    return AIMessage(
        content=message.get("content") or "",
        additional_kwargs=additional_kwargs,
        tool_calls=tool_calls,
        invalid_tool_calls=invalid_tool_calls,
    )


def _convert_chunk(event: Mapping[str, Any]) -> ChatGenerationChunk | None:
    """Convert one streamed `chat.completion.chunk` to a LangChain chunk.

    Core merges chunks by concatenating string metadata (except `model_provider`
    and `id`), so `finish_reason`, `model_name` and `system_fingerprint` go on
    the finish chunk only.
    """
    choices: list[dict[str, Any]] = event.get("choices") or []
    if not choices:
        # The last chunk with `stream_options.include_usage`: usage and timings.
        usage = event.get("usage")
        timings = event.get("timings")
        if usage is None and timings is None:
            return None
        response_metadata: dict[str, Any] = {"model_provider": MODEL_PROVIDER}
        if timings is not None:
            response_metadata["timings"] = timings
        message = AIMessageChunk(
            content="",
            usage_metadata=_create_usage_metadata(usage) if usage else None,
            response_metadata=response_metadata,
        )
        return ChatGenerationChunk(message=message)

    choice = choices[0]
    delta: Mapping[str, Any] = choice.get("delta") or {}
    additional_kwargs: dict[str, Any] = {}
    if reasoning := delta.get("reasoning_content"):
        additional_kwargs["reasoning_content"] = reasoning
    raw_tool_calls: list[dict[str, Any]] = delta.get("tool_calls") or []
    tool_call_chunks: list[ToolCallChunk] = []
    for raw in raw_tool_calls:
        # The id and name arrive in the first fragment, the arguments in pieces.
        function: Mapping[str, Any] = raw.get("function") or {}
        tool_call_chunks.append(
            create_tool_call_chunk(
                name=function.get("name"),
                args=function.get("arguments"),
                id=raw.get("id"),
                index=raw.get("index"),
            )
        )
    message = AIMessageChunk(
        content=delta.get("content") or "",
        additional_kwargs=additional_kwargs,
        tool_call_chunks=tool_call_chunks,
        response_metadata={"model_provider": MODEL_PROVIDER},
    )
    generation_info: dict[str, Any] | None = None
    if finish_reason := choice.get("finish_reason"):
        generation_info = {
            key: value
            for key, value in {
                "finish_reason": finish_reason,
                "model_name": event.get("model"),
                "system_fingerprint": event.get("system_fingerprint"),
                "id": event.get("id"),
            }.items()
            if value is not None
        }
    if choice.get("logprobs"):
        generation_info = {**(generation_info or {}), "logprobs": choice["logprobs"]}
    return ChatGenerationChunk(message=message, generation_info=generation_info)


def _create_usage_metadata(usage: Mapping[str, Any]) -> UsageMetadata:
    """Map llama-server's `usage` the way ChatOpenAI does."""
    input_tokens: int = usage.get("prompt_tokens") or 0
    output_tokens: int = usage.get("completion_tokens") or 0
    total_tokens: int | None = usage.get("total_tokens")
    prompt_details: Mapping[str, Any] = usage.get("prompt_tokens_details") or {}
    completion_details: Mapping[str, Any] = usage.get("completion_tokens_details") or {}
    cache_read: int | None = prompt_details.get("cached_tokens")
    reasoning: int | None = completion_details.get("reasoning_tokens")
    input_details = InputTokenDetails()
    if cache_read is not None:
        input_details["cache_read"] = cache_read
    output_details = OutputTokenDetails()
    if reasoning is not None:
        output_details["reasoning"] = reasoning
    return UsageMetadata(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=(
            total_tokens if total_tokens is not None else input_tokens + output_tokens
        ),
        input_token_details=input_details,
        output_token_details=output_details,
    )


def _add_usage(total: dict[str, Any], usage: Mapping[str, Any]) -> None:
    """Add `usage` into `total`, recursing into nested detail dicts."""
    for key, value in usage.items():
        if value is None:
            continue
        if isinstance(value, Mapping):
            _add_usage(total.setdefault(key, {}), value)  # pyright: ignore[reportUnknownArgumentType]
        elif isinstance(value, int | float):
            total[key] = total.get(key, 0) + value


def _round_float32(value: Any) -> Any:
    """Undo float32 noise: 0.949999988079071 -> 0.95. Other values pass through."""
    return float(f"{value:.7g}") if isinstance(value, float) else value


def _retry_after(response: httpx.Response) -> float | None:
    """Seconds from a numeric `Retry-After` header, if it's present and sane."""
    try:
        seconds = float(response.headers.get("retry-after", ""))
    except ValueError:
        return None
    return seconds if 0 <= seconds <= MAX_RETRY_AFTER else None


def _retry_delay(attempt: int, retry_after: float | None) -> float:
    if retry_after is not None:
        return retry_after
    delay = min(INITIAL_RETRY_DELAY * 2**attempt, MAX_RETRY_DELAY)
    return delay * (1 - 0.25 * random.random())  # noqa: S311 (jitter, not crypto)
