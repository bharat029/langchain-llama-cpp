"""ChatLlama: a LangChain chat model for llama.cpp's llama-server behind llama-swap."""

import asyncio
import json
import random
import time
import weakref
from collections.abc import AsyncIterator, Callable, Iterator, Mapping, Sequence
from operator import itemgetter
from typing import Any, Literal, cast

import httpx
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.exceptions import ModelError, OutputParserException
from langchain_core.language_models import (
    BaseChatModel,
    LangSmithParams,
    LanguageModelInput,
)
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    InvalidToolCall,
    SystemMessage,
    ToolCall,
    convert_to_openai_messages,
)
from langchain_core.messages.ai import (
    InputTokenDetails,
    OutputTokenDetails,
    UsageMetadata,
)
from langchain_core.messages.tool import ToolCallChunk
from langchain_core.messages.tool import invalid_tool_call as create_invalid_tool_call
from langchain_core.messages.tool import tool_call_chunk as create_tool_call_chunk
from langchain_core.output_parsers import (
    JsonOutputKeyToolsParser,
    JsonOutputParser,
    PydanticOutputParser,
    PydanticToolsParser,
)
from langchain_core.output_parsers.format_instructions import (
    JSON_FORMAT_INSTRUCTIONS,
)
from langchain_core.output_parsers.openai_tools import (
    make_invalid_tool_call,
    parse_tool_call,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import (
    Runnable,
    RunnableLambda,
    RunnableMap,
    RunnablePassthrough,
)
from langchain_core.tools import BaseTool
from langchain_core.utils import from_env
from langchain_core.utils.function_calling import (
    convert_to_json_schema,
    convert_to_openai_function,
    convert_to_openai_tool,
)
from langchain_core.utils.pydantic import is_basemodel_subclass
from pydantic import BaseModel, Field, PrivateAttr

from langchain_llama_cpp.content_blocks import MODEL_PROVIDER
from langchain_llama_cpp.exceptions import (
    error_from_cut_off_stream,
    error_from_response,
    error_from_stream_event,
    error_from_transport,
)

CHAT_COMPLETIONS = "/v1/chat/completions"

_AsyncClients = weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, httpx.AsyncClient]


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

    max_retries: int = Field(default=2, ge=0)
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
    # An AsyncClient's pooled connections belong to the event loop that opened
    # them, so there is one client per loop (langchain-ai/langchain#35783 is
    # this bug in ChatOpenAI). Threads running their own loops don't evict each
    # other; clients of closed loops are dropped in `_get_async_client`.
    _async_clients: _AsyncClients = PrivateAttr(default_factory=_AsyncClients)

    @property
    def _llm_type(self) -> str:
        return "llama-cpp-chat"

    @property
    def _params(self) -> dict[str, Any]:
        """Request fields set on the model; fields left at `None` are left out."""
        return _without_none(**{name: getattr(self, name) for name in _REQUEST_FIELDS})

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model": self.model, "base_url": self.base_url, **self._params}

    def _get_ls_params(
        self, stop: list[str] | None = None, **kwargs: Any
    ) -> LangSmithParams:
        # Core fills the model name, temperature and max_tokens (per-call first).
        params = super()._get_ls_params(stop=stop or self.stop, **kwargs)
        params["ls_provider"] = MODEL_PROVIDER
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
        response = self._send("POST", CHAT_COMPLETIONS, payload)
        return _create_chat_result(response.json())

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,  # noqa: ARG002
        **kwargs: Any,
    ) -> ChatResult:
        payload = self._build_payload(messages, stop, **kwargs)
        response = await self._asend("POST", CHAT_COMPLETIONS, payload)
        return _create_chat_result(response.json())

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,  # noqa: ARG002
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        # Core calls run_manager.on_llm_new_token for each chunk itself.
        payload = self._build_payload(messages, stop, streaming=True, **kwargs)
        response = self._send("POST", CHAT_COMPLETIONS, payload, stream=True)
        try:
            for line in response.iter_lines():
                chunk = _chunk_from_line(line, response)
                if chunk is _DONE:
                    return
                if chunk is not None:
                    yield chunk
        except httpx.TransportError as exc:  # the connection broke mid-stream
            raise error_from_transport(exc) from exc
        finally:
            response.close()  # also when the caller stops early
        # Only [DONE] returns early: the stream was cut off.
        raise error_from_cut_off_stream(response)

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,  # noqa: ARG002
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        payload = self._build_payload(messages, stop, streaming=True, **kwargs)
        response = await self._asend("POST", CHAT_COMPLETIONS, payload, stream=True)
        try:
            async for line in response.aiter_lines():
                chunk = _chunk_from_line(line, response)
                if chunk is _DONE:
                    return
                if chunk is not None:
                    yield chunk
        except httpx.TransportError as exc:
            raise error_from_transport(exc) from exc
        finally:
            await response.aclose()
        # Only [DONE] returns early: the stream was cut off.
        raise error_from_cut_off_stream(response)

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
        response = self._send("GET", f"/upstream/{self.model}/props")
        settings: dict[str, Any] = response.json()["default_generation_settings"]
        defaults = {
            key: _round_float32(value)
            for key, value in settings.get("params", {}).items()
        }
        defaults["n_ctx"] = settings.get("n_ctx")
        return defaults

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: dict[str, Any] | str | bool | None = None,
        strict: bool | None = None,
        parallel_tool_calls: bool | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        """Bind tools to the model, sent as OpenAI-format `tools`.

        Args:
            tools: Anything `convert_to_openai_tool` accepts: `@tool` functions,
                `BaseTool`s, Pydantic classes, `TypedDict`s, plain functions or
                OpenAI tool dicts.
            tool_choice: llama-server accepts only the strings `"auto"`,
                `"none"` and `"required"`, which are sent as they are.
                `"any"` and `True` become `"required"`. A bound tool's name, or
                `{"type": "function", "function": {"name": ...}}`, also becomes
                `"required"`: the server can't force one particular tool (it
                turns the object form into `"auto"`), so the model must call a
                tool but may pick another one. `None`, `False` and `""` leave
                it out.

                How strictly `"required"` holds depends on the chat handler
                llama-server picks for the model's template. For Qwen3.5/3.6
                (the handler llama.cpp names "qwen3-coder") the grammar lets
                the model write text before the call and forbids ending without
                one, so a model that doesn't want a tool keeps writing until
                `max_tokens`, with no tool call (llama.cpp issue #27217, open
                as of b11429). Keep `max_tokens` set when forcing tools.
            strict: Sent as `strict` on each tool when set. llama-server
                ignores it: its grammar always enforces the schema.
            parallel_tool_calls: Sent when set.
            **kwargs: Passed through to the request, like `bind`.

        Returns:
            A runnable that takes the model's input and returns an `AIMessage`.

        Raises:
            ValueError: If `tool_choice` names a tool that isn't bound, or isn't
                a value described above.
        """
        formatted_tools = [
            convert_to_openai_tool(tool, strict=strict) for tool in tools
        ]
        if tool_choice:  # None, False and "" leave it out, as in ChatOpenAI
            tool_names = [
                tool["function"]["name"]
                for tool in formatted_tools
                if "function" in tool
            ]
            kwargs["tool_choice"] = _tool_choice(
                tool_choice=tool_choice, tool_names=tool_names
            )
        if parallel_tool_calls is not None:
            kwargs["parallel_tool_calls"] = parallel_tool_calls
        return self.bind(tools=formatted_tools, **kwargs)

    def with_structured_output(
        self,
        schema: dict[str, Any] | type | None = None,
        *,
        method: Literal[
            "function_calling", "json_schema", "json_mode"
        ] = "function_calling",
        include_raw: bool = False,
        strict: bool | None = None,
        include_format_instructions: bool = False,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, dict[str, Any] | BaseModel]:
        """Return a runnable whose output is parsed into `schema`.

        Args:
            schema: A Pydantic class, a `TypedDict`, a JSON schema or an OpenAI
                tool/function dict. Optional only for `method="json_mode"`.
            method: How the server is asked for the structure:

                - `"function_calling"` (default): the schema is the only tool,
                  sent with `tool_choice: "required"` and
                  `parallel_tool_calls: false`. The chat template shows the
                  schema (with its descriptions) to the model, and the
                  server's grammar constrains the arguments. With Qwen3.5/3.6,
                  a prompt the model doesn't answer with the tool runs to
                  `max_tokens` (see `bind_tools`).
                - `"json_schema"`: `response_format: {"type": "json_schema"}`.
                  Grammar only: the model never sees the schema or its
                  descriptions unless `include_format_instructions=True`.
                - `"json_mode"`: `response_format: {"type": "json_object"}`,
                  any JSON object.
            include_raw: Return `{"raw", "parsed", "parsing_error"}` instead of
                raising on a parsing error.
            strict: Sent as `strict` in the tool or `json_schema` when set.
                llama-server ignores it: its grammar always enforces the schema.
            include_format_instructions: For `json_schema` and `json_mode`, add
                core's format instructions for the schema to the system message
                (after a leading system message's content, else as a new first
                system message). Your messages are not modified.
            **kwargs: Passed through to the request, like `bind`.

        Returns:
            A runnable that takes the model's input and returns an instance of
            the Pydantic class, a dict, or the `include_raw` dict.

        Raises:
            ValueError: If `schema` is missing where it's needed, `method` is
                unknown, or `include_format_instructions` is combined with
                `function_calling`.
        """
        _check_structured_output_args(
            schema,
            method=method,
            include_format_instructions=include_format_instructions,
            kwargs=kwargs,
        )
        pydantic_schema = (
            cast("type[BaseModel]", schema)
            if isinstance(schema, type) and is_basemodel_subclass(schema)
            else None
        )
        ls_kwargs: dict[str, Any] = {"method": method}
        if method != "json_mode":
            ls_kwargs["strict"] = strict
        ls_format = {"kwargs": ls_kwargs, "schema": schema}
        llm: Runnable[LanguageModelInput, BaseMessage]
        output_parser: Runnable[Any, Any]
        if schema is not None and method == "function_calling":
            given = _given_json_schema(schema)
            tool = (
                _tool_from_json_schema(given, strict=strict)
                if given is not None
                else convert_to_openai_tool(schema, strict=strict)
            )
            # The caller's kwargs win over our defaults, as in ChatOpenAI.
            bind_kwargs: dict[str, Any] = {
                "tool_choice": "required",
                "parallel_tool_calls": False,
                "ls_structured_output_format": ls_format,
                **kwargs,
            }
            llm = self.bind_tools([tool], **bind_kwargs)
            output_parser = (
                PydanticToolsParser(tools=[pydantic_schema], first_tool_only=True)
                if pydantic_schema
                else JsonOutputKeyToolsParser(
                    key_name=tool["function"]["name"], first_tool_only=True
                )
            )
        else:
            response_format = (
                _json_schema_response_format(schema, strict=strict)
                if schema is not None and method == "json_schema"
                else {"type": "json_object"}
            )
            llm = self.bind(
                **{
                    "response_format": response_format,
                    "ls_structured_output_format": ls_format,
                    **kwargs,
                }
            )
            if schema is not None and include_format_instructions:
                instructions = _format_instructions(schema)

                def add_format_instructions(
                    model_input: LanguageModelInput,
                ) -> list[BaseMessage]:
                    messages = self._convert_input(model_input).to_messages()
                    return _add_to_system_message(messages, instructions)

                llm = RunnableLambda(add_format_instructions) | llm
            output_parser = (
                PydanticOutputParser(pydantic_object=pydantic_schema)
                if pydantic_schema
                else JsonOutputParser()
            )

        if include_raw:
            parser_assign = RunnablePassthrough.assign(
                parsed=itemgetter("raw") | output_parser, parsing_error=lambda _: None
            )
            parser_none = RunnablePassthrough.assign(parsed=lambda _: None)
            parser_with_fallback = parser_assign.with_fallbacks(
                [parser_none], exception_key="parsing_error"
            )
            return RunnableMap(raw=llm) | parser_with_fallback
        return llm | output_parser

    def _build_payload(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None,
        *,
        streaming: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        stream_usage = kwargs.pop("stream_usage", self.stream_usage)
        # Core leaves `invoke(..., stream=True)` in kwargs; `streaming` decides.
        kwargs.pop("stream", None)
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [_convert_message(message) for message in messages],
            **self._params,
            **kwargs,
        }
        if stop is not None:
            payload["stop"] = stop
        if streaming:
            # Chunks would interleave completions; ChatOpenAI refuses this too.
            # `n_cmpl` is llama-server's alias for `n`.
            if max(payload.get("n") or 1, payload.get("n_cmpl") or 1) > 1:
                msg = "n must be 1 when streaming."
                raise ValueError(msg)
            payload["stream"] = True
            if stream_usage:
                payload["stream_options"] = {"include_usage": True}
        return payload

    def _build_request(
        self, method: str, path: str, payload: dict[str, Any] | None
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
        loop = asyncio.get_running_loop()
        client = self._async_clients.get(loop)
        if client is None:
            # Pooled connections keep their loop alive, so a weak key never
            # dies on its own: drop clients of loops that have closed (each
            # asyncio.run). They can't be aclose()d without their loop.
            for old_loop in [old for old in self._async_clients if old.is_closed()]:
                del self._async_clients[old_loop]
            client = httpx.AsyncClient(timeout=self.timeout)
            self._async_clients[loop] = client
        return client

    def _send(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        *,
        stream: bool = False,
    ) -> httpx.Response:
        """Send a request to `path`, retrying on errors core marks `is_retryable`.

        With `stream=True` this returns once the headers arrive, so retries only
        cover the request itself; a failure mid-stream is raised, not retried.
        """
        client = self._get_client()
        request = self._build_request(method, path, payload)
        attempt = 0
        while True:
            try:
                response = client.send(request, stream=stream)
            except httpx.TransportError as exc:
                delay = self._retry_delay_or_raise(
                    attempt, error_from_transport(exc), cause=exc
                )
            else:
                if response.is_success:
                    return response
                response.read()  # a streamed error body isn't read yet
                response.close()
                delay = self._retry_delay_or_raise(
                    attempt, error_from_response(response)
                )
            time.sleep(delay)
            attempt += 1

    async def _asend(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        *,
        stream: bool = False,
    ) -> httpx.Response:
        """Async twin of `_send`."""
        client = self._get_async_client()
        request = self._build_request(method, path, payload)
        attempt = 0
        while True:
            try:
                response = await client.send(request, stream=stream)
            except httpx.TransportError as exc:
                delay = self._retry_delay_or_raise(
                    attempt, error_from_transport(exc), cause=exc
                )
            else:
                if response.is_success:
                    return response
                await response.aread()
                await response.aclose()
                delay = self._retry_delay_or_raise(
                    attempt, error_from_response(response)
                )
            await asyncio.sleep(delay)
            attempt += 1

    def _retry_delay_or_raise(
        self, attempt: int, error: Exception, cause: BaseException | None = None
    ) -> float:
        """Seconds to wait before retrying `error`, or raise it if it's final.

        An HTTP error's response supplies `Retry-After`.
        """
        if not _is_retryable(error) or attempt >= self.max_retries:
            raise error from cause
        retry_after = (
            _retry_after(error.response)
            if isinstance(error, httpx.HTTPStatusError)
            else None
        )
        return _retry_delay(attempt, retry_after)


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


# What `_chunk_from_line` returns for `data: [DONE]`, the end of a finished stream.
_DONE = ChatGenerationChunk(message=AIMessageChunk(content=""))


def _chunk_from_line(line: str, response: httpx.Response) -> ChatGenerationChunk | None:
    """The chunk for one SSE line: None to skip the line, `_DONE` at the end."""
    data = _sse_data(line)
    if data is None:
        return None
    if data == "[DONE]":
        return _DONE
    return _convert_chunk(_sse_event(data, response))


def _sse_event(data: str, response: httpx.Response) -> dict[str, Any]:
    """Decode one event. An `error` event (failure mid-stream) is raised.

    So is data that isn't a JSON object, as an API error.
    """
    try:
        event: object = json.loads(data)
    except ValueError as exc:
        detail = {"message": f"invalid JSON in stream event: {data[:200]!r}"}
        raise error_from_stream_event(detail, response) from exc
    if not isinstance(event, dict):
        detail = {"message": f"stream event isn't a JSON object: {data[:200]!r}"}
        raise error_from_stream_event(detail, response)
    event = cast("dict[str, Any]", event)
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


_REQUEST_FIELDS = tuple(request_field_names())  # fixed once the class exists


def _tool_choice(
    *, tool_choice: dict[str, Any] | str | bool, tool_names: list[str]
) -> str:
    """Map a LangChain `tool_choice` to one of the strings llama-server accepts.

    A named tool becomes `"required"`: llama-server turns the object form into
    `"auto"` (b11429), so `"required"` is the closest it can do.
    """
    if tool_choice is True or tool_choice == "any":
        return "required"
    if isinstance(tool_choice, str) and tool_choice in {"auto", "none", "required"}:
        return tool_choice
    name: object = tool_choice
    if isinstance(tool_choice, dict):
        function: object = tool_choice.get("function")
        if isinstance(function, dict):
            name = cast("dict[str, Any]", function).get("name")
    if isinstance(name, str) and name in tool_names:
        return "required"
    msg = (
        "tool_choice must be 'auto', 'none', 'required', 'any', True, or the "
        f"name of a bound tool ({', '.join(tool_names)}); got {tool_choice!r}"
    )
    raise ValueError(msg)


def _check_structured_output_args(
    schema: dict[str, Any] | type | None,
    *,
    method: str,
    include_format_instructions: bool,
    kwargs: Mapping[str, Any],
) -> None:
    """Raise `ValueError` for argument combinations `with_structured_output` rejects."""
    if method not in {"function_calling", "json_schema", "json_mode"}:
        msg = (
            "method must be 'function_calling', 'json_schema' or 'json_mode', "
            f"got {method!r}"
        )
        raise ValueError(msg)
    if include_format_instructions and method == "function_calling":
        msg = (
            "include_format_instructions applies to json_schema and json_mode; "
            "with function_calling the chat template already shows the schema "
            "to the model."
        )
        raise ValueError(msg)
    if schema is None and (method != "json_mode" or include_format_instructions):
        msg = f"schema is required for method={method!r}"
        if include_format_instructions:
            msg += " with include_format_instructions=True"
        raise ValueError(msg)
    if method == "function_calling" and "tools" in kwargs:
        msg = "with_structured_output binds the schema as the only tool"
        raise ValueError(msg)


def _json_schema_response_format(
    schema: dict[str, Any] | type, *, strict: bool | None
) -> dict[str, Any]:
    """`response_format` for `json_schema`: `{name, description?, schema, strict?}`."""
    if (given := _given_json_schema(schema)) is not None:
        json_schema = dict(given)
    else:
        function = convert_to_openai_function(schema)
        json_schema = {"name": function["name"]}
        if description := function.get("description"):
            json_schema["description"] = description
        json_schema["schema"] = function["parameters"]
    if strict is not None:
        json_schema["strict"] = strict
    return {"type": "json_schema", "json_schema": json_schema}


def _tool_from_json_schema(
    given: dict[str, Any], *, strict: bool | None
) -> dict[str, Any]:
    """An OpenAI tool from a `{name, description?, schema, strict?}` dict.

    `convert_to_openai_tool` doesn't know this shape: it would keep the name
    and drop the schema, leaving a tool without parameters.
    """
    function: dict[str, Any] = {"name": given["name"]}
    if description := given.get("description"):
        function["description"] = description
    function["parameters"] = given["schema"]
    if strict is None:
        strict = given.get("strict")
    if strict is not None:
        function["strict"] = strict
    return {"type": "function", "function": function}


def _given_json_schema(schema: dict[str, Any] | type) -> dict[str, Any] | None:
    """The `{name, schema, ...}` part of a schema already in OpenAI's shape.

    Accepts the two shapes ChatOpenAI does: a whole `response_format`
    (`{"type": "json_schema", "json_schema": {...}}`) and its inner
    `{"name": ..., "schema": ...}`. Anything else returns None.

    Raises:
        ValueError: If a `response_format` has no `json_schema.schema` object.
    """
    if not isinstance(schema, dict):
        return None
    if schema.get("type") == "json_schema":
        inner: object = schema.get("json_schema")
        if not (
            isinstance(inner, dict)
            and "name" in inner
            and isinstance(cast("dict[str, Any]", inner).get("schema"), dict)
        ):
            msg = (
                "a json_schema response_format needs "
                '{"json_schema": {"name": ..., "schema": {...}}}'
            )
            raise ValueError(msg)
        return cast("dict[str, Any]", inner)
    if "name" in schema and isinstance(schema.get("schema"), dict):
        return schema
    return None


def _format_instructions(schema: dict[str, Any] | type) -> str:
    """Core's format instructions for `schema`.

    Pydantic classes get `PydanticOutputParser.get_format_instructions()`. Other
    schemas get the text `JsonOutputParser.get_format_instructions()` builds,
    which it only does for Pydantic classes itself: the schema minus `title`
    and `type`, in `JSON_FORMAT_INSTRUCTIONS`.
    """
    if isinstance(schema, type) and is_basemodel_subclass(schema):
        return PydanticOutputParser(pydantic_object=schema).get_format_instructions()
    if (given := _given_json_schema(schema)) is not None:
        reduced = dict(given["schema"])
    else:
        reduced = convert_to_json_schema(schema)
    reduced.pop("title", None)
    reduced.pop("type", None)
    return JSON_FORMAT_INSTRUCTIONS.format(
        schema=json.dumps(reduced, ensure_ascii=False)
    )


def _add_to_system_message(
    messages: list[BaseMessage], instructions: str
) -> list[BaseMessage]:
    """Append `instructions` to a leading system message, or insert one first.

    Returns a new list; the caller's messages are left untouched.
    """
    if messages and isinstance(messages[0], SystemMessage):
        system = messages[0]
        content: str | list[str | dict[str, Any]]
        if isinstance(system.content, str):
            content = f"{system.content}\n\n{instructions}"
        else:
            content = [*system.content, {"type": "text", "text": instructions}]
        return [system.model_copy(update={"content": content}), *messages[1:]]
    return [SystemMessage(content=instructions), *messages]


def _convert_message(message: BaseMessage) -> dict[str, Any]:
    """Convert a message to the OpenAI shape llama-server expects."""
    converted = convert_to_openai_messages(message)
    if isinstance(message, AIMessage):
        # Core leaves v1 reasoning blocks in `content`; send text as content and
        # reasoning as `reasoning_content` (M1-c: the chat template decides).
        converted["content"] = message.text
        if isinstance(message.content, str):
            # No blocks to look through; this is where core's best-effort
            # parsing finds reasoning for string content.
            kwarg = message.additional_kwargs.get("reasoning_content")
            reasoning = kwarg if isinstance(kwarg, str) else ""
        else:
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
    response_metadata = _without_none(
        model_name=data.get("model"),
        system_fingerprint=data.get("system_fingerprint"),
        id=data.get("id"),
        timings=data.get("timings"),
        model_provider=MODEL_PROVIDER,
    )
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
        if not isinstance(raw.get("function"), dict):  # parse_tool_call drops these
            invalid_tool_calls.append(
                create_invalid_tool_call(
                    name=None,
                    args=None,
                    id=raw.get("id"),
                    error="tool call has no 'function' object",
                )
            )
            continue
        try:
            parsed = parse_tool_call(raw, return_id=True)
        except OutputParserException as exc:  # arguments aren't valid JSON
            invalid_tool_calls.append(make_invalid_tool_call(raw, str(exc)))
            continue
        if parsed is not None:  # with return_id, already a complete ToolCall
            tool_calls.append(cast("ToolCall", parsed))
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
    generation_info: dict[str, Any] = {}
    if finish_reason := choice.get("finish_reason"):
        generation_info = _without_none(
            finish_reason=finish_reason,
            model_name=event.get("model"),
            system_fingerprint=event.get("system_fingerprint"),
            id=event.get("id"),
        )
    if logprobs := choice.get("logprobs"):
        generation_info["logprobs"] = logprobs
    return ChatGenerationChunk(message=message, generation_info=generation_info or None)


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


def _without_none(**items: Any) -> dict[str, Any]:
    """`items` minus the ones that are None."""
    return {key: value for key, value in items.items() if value is not None}


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
