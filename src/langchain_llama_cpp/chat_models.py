"""ChatLlama: a LangChain chat model for llama.cpp's llama-server behind llama-swap."""

import random
import time
from collections.abc import Mapping
from typing import Any

import httpx
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.exceptions import ModelError, OutputParserException
from langchain_core.language_models import BaseChatModel, LangSmithParams
from langchain_core.messages import (
    AIMessage,
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
from langchain_core.output_parsers.openai_tools import (
    make_invalid_tool_call,
    parse_tool_call,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils import from_env
from pydantic import Field, PrivateAttr

from langchain_llama_cpp.exceptions import error_from_response, error_from_transport

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

    temperature: float | None = None
    """Sampling temperature."""

    max_tokens: int | None = None
    """Maximum tokens to generate. Thinking tokens count towards it."""

    top_p: float | None = None
    """Nucleus sampling: keep the smallest set of tokens with this total probability."""

    top_k: int | None = None
    """Keep only the k most likely tokens (llama.cpp extension)."""

    min_p: float | None = None
    """Drop tokens below this fraction of the top token's probability (llama.cpp)."""

    seed: int | None = None
    """Random seed for sampling."""

    stop: list[str] | None = None
    """Stop sequences."""

    chat_template_kwargs: dict[str, Any] | None = None
    """Variables for the chat template, e.g. `{"enable_thinking": False}`."""

    timeout: float | httpx.Timeout | None = None
    """Request timeout in seconds. `None` (the default) means no timeout."""

    max_retries: int = 2
    """Retries for connection errors, timeouts, 429 and 5xx. 0 turns them off."""

    http_client: httpx.Client | None = Field(default=None, exclude=True)
    """Your own `httpx.Client`. If given, `timeout` is not applied to it."""

    _client: httpx.Client | None = PrivateAttr(default=None)

    @property
    def _llm_type(self) -> str:
        return "llama-cpp-chat"

    @property
    def _params(self) -> dict[str, Any]:
        """Request fields set on the model; fields left at `None` are left out."""
        params = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "min_p": self.min_p,
            "seed": self.seed,
            "stop": self.stop,
            "chat_template_kwargs": self.chat_template_kwargs,
        }
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
        self, messages: list[BaseMessage], stop: list[str] | None, **kwargs: Any
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [_convert_message(message) for message in messages],
            **self._params,
            **kwargs,
        }
        if stop is not None:
            payload["stop"] = stop
        return payload

    def _get_client(self) -> httpx.Client:
        if self.http_client is not None:
            return self.http_client
        if self._client is None:
            # httpx's own default is a 5 s timeout, so always pass ours.
            self._client = httpx.Client(timeout=self.timeout)
        return self._client

    def _request(
        self, method: str, path: str, payload: dict[str, Any] | None = None
    ) -> httpx.Response:
        """Send a request, retrying on errors core marks `is_retryable`."""
        url = self.base_url.rstrip("/") + path
        client = self._get_client()
        for attempt in range(self.max_retries + 1):
            retry_after = None
            try:
                response = client.request(method, url, json=payload)
            except httpx.TransportError as exc:
                error: Exception = error_from_transport(exc)
                cause: Exception | None = exc
            else:
                if response.is_success:
                    return response
                error, cause = error_from_response(response), None
                retry_after = _retry_after(response)
            retryable = isinstance(error, ModelError) and error.is_retryable
            if not retryable or attempt == self.max_retries:
                raise error from cause
            time.sleep(_retry_delay(attempt, retry_after))
        msg = "unreachable"  # the loop always returns or raises
        raise AssertionError(msg)


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
