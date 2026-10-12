"""Errors raised by ChatLlama.

Each class inherits from both the httpx exception it replaces and the matching
`langchain_core.exceptions.ModelError` subclass, so `except httpx.HTTPStatusError`
and `except ContextOverflowError` both work. `is_retryable` comes from core.
"""

from typing import Any, cast

import httpx
from langchain_core.exceptions import (
    ContextOverflowError,
    ModelAPIError,
    ModelAuthenticationError,
    ModelConnectionError,
    ModelInvalidRequestError,
    ModelNotFoundError,
    ModelPermissionDeniedError,
    ModelRateLimitError,
    ModelTimeoutError,
)


class LlamaContextOverflowError(httpx.HTTPStatusError, ContextOverflowError):
    """The prompt is longer than the server's context window (HTTP 400).

    Attributes:
        n_prompt_tokens: Tokens in the rejected prompt, as counted by the server.
        n_ctx: The server's context window.
    """

    def __init__(
        self,
        message: str,
        *,
        request: httpx.Request,
        response: httpx.Response,
        n_prompt_tokens: int | None = None,
        n_ctx: int | None = None,
    ) -> None:
        """Create the error with the server's token counts."""
        super().__init__(message, request=request, response=response)
        self.n_prompt_tokens = n_prompt_tokens
        self.n_ctx = n_ctx


class LlamaInvalidRequestError(httpx.HTTPStatusError, ModelInvalidRequestError):
    """The server rejected the request (HTTP 400 and other 4xx)."""


class LlamaAuthenticationError(httpx.HTTPStatusError, ModelAuthenticationError):
    """The server needs an API key, or rejected the one sent (HTTP 401)."""


class LlamaPermissionDeniedError(httpx.HTTPStatusError, ModelPermissionDeniedError):
    """The server refused the request (HTTP 403)."""


class LlamaModelNotFoundError(httpx.HTTPStatusError, ModelNotFoundError):
    """llama-swap has no entry with this `model` name (HTTP 404)."""


class LlamaRateLimitError(httpx.HTTPStatusError, ModelRateLimitError):
    """The server has no free slot (HTTP 429)."""


class LlamaAPIError(httpx.HTTPStatusError, ModelAPIError):
    """The server failed (HTTP 5xx)."""


class LlamaConnectionError(httpx.TransportError, ModelConnectionError):
    """The server could not be reached, or the connection broke."""


class LlamaTimeoutError(httpx.TimeoutException, ModelTimeoutError):
    """The request took longer than `timeout`."""


_STATUS_ERRORS: dict[int, type[httpx.HTTPStatusError]] = {
    401: LlamaAuthenticationError,
    403: LlamaPermissionDeniedError,
    404: LlamaModelNotFoundError,
    429: LlamaRateLimitError,
}


def error_from_response(response: httpx.Response) -> httpx.HTTPStatusError:
    """Build the right error for an HTTP error response."""
    # llama-server: {"error": {"type", "message", ...}}; llama-swap: plain text.
    try:
        body: object = response.json()
    except ValueError:
        body = None
    raw_detail = _json_object(body).get("error")
    detail = _json_object(raw_detail)
    reason = detail.get("message") or raw_detail or response.text
    message = f"HTTP {response.status_code} from {response.request.url}: {reason}"
    return _build_error(
        message,
        detail,
        response.status_code,
        response,
        fallback=LlamaInvalidRequestError,  # e.g. a 3xx that wasn't followed
    )


def error_from_stream_event(
    detail: object, response: httpx.Response
) -> httpx.HTTPStatusError:
    """Build the error for a `data: {"error": ...}` event in a running stream.

    The HTTP status was already 200, so the class comes from the event's
    `type` and `code` (llama-server's `format_error_response`).
    """
    fields = _json_object(detail)
    reason = fields.get("message") or detail or "error during streaming"
    message = f"Error during streaming from {response.request.url}: {reason}"
    code = fields.get("code")
    status = code if isinstance(code, int) else None
    return _build_error(message, fields, status, response, fallback=LlamaAPIError)


def error_from_cut_off_stream(response: httpx.Response) -> LlamaConnectionError:
    """The error for a stream that ended without `data: [DONE]` or an error event.

    llama-server always ends a finished stream with `[DONE]`, so the
    connection was closed early (a crash, a restart, or a proxy in between).
    """
    message = f"Stream from {response.request.url} ended without data: [DONE]"
    return LlamaConnectionError(message, request=response.request)


def _build_error(
    message: str,
    detail: dict[str, Any],
    status: int | None,
    response: httpx.Response,
    *,
    fallback: type[httpx.HTTPStatusError],
) -> httpx.HTTPStatusError:
    """Pick the error class from llama-server's error `type` and the status.

    Overflow is recognised by `type`, then the status decides: a mapped code,
    any other 4xx (invalid request), 5xx (API error), else `fallback`.
    """
    request = response.request
    if detail.get("type") == "exceed_context_size_error":
        return LlamaContextOverflowError(
            message,
            request=request,
            response=response,
            n_prompt_tokens=detail.get("n_prompt_tokens"),
            n_ctx=detail.get("n_ctx"),
        )
    cls = fallback
    if status in _STATUS_ERRORS:
        cls = _STATUS_ERRORS[status]
    elif status is not None and 400 <= status < 500:  # noqa: PLR2004
        cls = LlamaInvalidRequestError
    elif status is not None and status >= 500:  # noqa: PLR2004
        cls = LlamaAPIError
    return cls(message, request=request, response=response)


def _json_object(value: object) -> dict[str, Any]:
    """`value` if it's a JSON object, else an empty dict."""
    return cast("dict[str, Any]", value) if isinstance(value, dict) else {}


def error_from_transport(exc: httpx.TransportError) -> httpx.TransportError:
    """Wrap an httpx transport failure in the matching ChatLlama error."""
    try:
        request: httpx.Request | None = exc.request
    except RuntimeError:  # httpx raises this when no request is attached
        request = None
    cls = (
        LlamaTimeoutError
        if isinstance(exc, httpx.TimeoutException)
        else LlamaConnectionError
    )
    return cls(str(exc) or type(exc).__name__, request=request)
