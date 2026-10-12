"""The JSON llama-server sends back from `/v1/chat/completions` (b11429).

Types for the type checker only: replies are plain dicts from `json.loads`,
never validated, so a server that sends something else isn't rejected here.
Every class is `total=False` because the code reads each key with `.get()`
and copes when it's missing; the docstrings say which keys the server always
sends.

Sources (llama.cpp b11429): `tools/server/server-task.cpp`
(`to_json_oaicompat_chat`, `to_json_oaicompat_chat_stream`,
`server_task_result_cmpl_partial::to_json_oaicompat_chat`,
`probs_vector_to_json`, `result_prompt_progress::to_json`),
`server-chat.cpp` (`server_chat_msg_diff_to_json_oaicompat`),
`server-common.cpp` (`server_slot_stats::to_json`, `format_error_response`),
`common/chat.cpp` (`common_chat_msg::to_json_oaicompat`).

A reply (`stream` off):

    {"id": "chatcmpl-...", "object": "chat.completion", "created": 1760000000,
     "model": "google/gemma-4-26B-A4B-it-qat-q4_0-gguf:Q4_0",
     "system_fingerprint": "b11429-d81235049",
     "choices": [{
         "index": 0, "finish_reason": "tool_calls",
         "message": {
             "role": "assistant", "content": "", "reasoning_content": "...",
             "tool_calls": [{
                 "type": "function", "id": "...",
                 "function": {"name": "get_weather", "arguments": "<JSON text>"}}]}}],
     "usage": {"prompt_tokens": 32, "completion_tokens": 5, "total_tokens": 37,
               "prompt_tokens_details": {"cached_tokens": 27}},
     "timings": {"cache_n": 27, "prompt_n": 5, ...}}

A stream (`stream` on) is SSE `data:` lines, each a `LlamaChunk`, then
`data: [DONE]`:

    {"choices": [{"index": 0, "finish_reason": null,
                  "delta": {"role": "assistant", "content": null}}], ...}  # first
    {"choices": [{"index": 0, "finish_reason": null,
                  "delta": {"reasoning_content": "Let"}}], ...}
    {"choices": [{"index": 0, "finish_reason": null,
                  "delta": {"tool_calls": [{
                      "index": 0, "id": "...", "type": "function",
                      "function": {"name": "get_weather"}}]}}], ...}
    {"choices": [{"index": 0, "finish_reason": null,
                  "delta": {"tool_calls": [{
                      "index": 0, "function": {"arguments": "<JSON piece>"}}]}}],
     ...}
    {"choices": [{"index": 0, "finish_reason": "tool_calls", "delta": {}}], ...}
    {"choices": [], "usage": {...}, "timings": {...}, ...}  # with include_usage

A failure after the first chunk arrives as `data: {"error": LlamaError}` and
the stream ends without `[DONE]`.
"""

from typing import Literal, TypedDict

FinishReason = Literal["stop", "length", "tool_calls"]
"""`length` when `max_tokens` or the context window ran out (b11429 sends
`length` for every stop that isn't an end-of-sequence token or a stop string)."""


class LlamaFunction(TypedDict, total=False):
    """A tool call's function. In a reply both keys are sent.

    In a stream, `name` comes in the first fragment and `arguments` in pieces
    after it; a fragment leaves out whichever it doesn't carry.
    """

    name: str
    arguments: str
    """JSON text, not an object: `'{"city": "Paris"}'`."""


class LlamaToolCall(TypedDict, total=False):
    """One tool call.

    In a reply: `type`, `function`, and `id` (left out if the template made
    none). In a stream fragment: always `index`; `id` and `type` only in the
    first fragment of each call; `function` when there's a name or arguments.
    """

    index: int
    """Which call a stream fragment belongs to (0, 1, ...). Stream only."""
    id: str
    type: Literal["function"]
    function: LlamaFunction


class LlamaContentPart(TypedDict, total=False):
    """A content part.

    `common_chat_msg::to_json_oaicompat` writes `content` as a list of these
    when the message holds parts instead of text; the reply parsers write
    text, so a list hasn't been seen in a reply.
    """

    type: str
    text: str


class LlamaMessage(TypedDict, total=False):
    """`choices[].message` in a reply. Always `role` and `content`."""

    role: Literal["assistant"]
    content: str | list[LlamaContentPart]
    """`""` when the model only called tools."""
    reasoning_content: str
    """The thinking, separate from `content`. Left out when there was none."""
    tool_calls: list[LlamaToolCall]
    """Left out when there were no calls."""


class LlamaDelta(TypedDict, total=False):
    """`choices[].delta` in a stream chunk. Each key only when it changed.

    The first chunk is `{"role": "assistant", "content": None}`; the finish
    chunk is `{}`. `tool_calls` holds one fragment at a time.
    """

    role: Literal["assistant"]
    content: str | None
    reasoning_content: str
    tool_calls: list[LlamaToolCall]


class LlamaTopLogprob(TypedDict, total=False):
    """One candidate token. `logprob`, or `prob` with `post_sampling_probs`."""

    id: int
    token: str
    bytes: list[int]
    logprob: float
    prob: float


class LlamaTokenLogprob(LlamaTopLogprob, total=False):
    """One generated token and its candidates.

    `top_logprobs`, or `top_probs` with `post_sampling_probs`.
    """

    top_logprobs: list[LlamaTopLogprob]
    top_probs: list[LlamaTopLogprob]


class LlamaLogprobs(TypedDict, total=False):
    """`choices[].logprobs`, only when `logprobs` or `n_probs` asked for it.

    In a stream, each chunk carries the tokens it generated.
    """

    content: list[LlamaTokenLogprob]


class LlamaChoice(TypedDict, total=False):
    """`choices[]` in a reply. Always `index`, `finish_reason` and `message`."""

    index: int
    finish_reason: FinishReason
    message: LlamaMessage
    logprobs: LlamaLogprobs


class LlamaChunkChoice(TypedDict, total=False):
    """`choices[]` in a stream chunk. Always `index`, `finish_reason`, `delta`.

    `finish_reason` is None until the finish chunk.
    """

    index: int
    finish_reason: FinishReason | None
    delta: LlamaDelta
    logprobs: LlamaLogprobs


class LlamaPromptTokensDetails(TypedDict, total=False):
    """`usage.prompt_tokens_details`."""

    cached_tokens: int
    """Prompt tokens taken from the KV cache. Part of `prompt_tokens`."""


class LlamaCompletionTokensDetails(TypedDict, total=False):
    """`usage.completion_tokens_details`, as OpenAI sends it.

    b11429 never sends it (reasoning tokens are counted in
    `completion_tokens` but not reported apart); read in case a build does.
    """

    reasoning_tokens: int


class LlamaUsage(TypedDict, total=False):
    """`usage`. b11429 always sends the four keys below except the last."""

    prompt_tokens: int
    completion_tokens: int
    """Includes thinking tokens."""
    total_tokens: int
    prompt_tokens_details: LlamaPromptTokensDetails
    completion_tokens_details: LlamaCompletionTokensDetails


class LlamaTimings(TypedDict, total=False):
    """`timings`, llama-server's own.

    The `draft_*` keys come only with speculative decoding.
    """

    cache_n: int
    """Prompt tokens reused from the KV cache."""
    prompt_n: int
    """Prompt tokens processed for this request."""
    prompt_ms: float
    prompt_per_token_ms: float
    prompt_per_second: float
    predicted_n: int
    """Tokens generated, thinking included."""
    predicted_ms: float
    predicted_per_token_ms: float
    predicted_per_second: float
    draft_n: int
    draft_n_accepted: int


class LlamaPromptProgress(TypedDict, total=False):
    """`prompt_progress`, in stream chunks with `return_progress`."""

    total: int
    cache: int
    processed: int
    time_ms: int


class LlamaResponse(TypedDict, total=False):
    """A `chat.completion` reply. Always every key but `timings`."""

    id: str
    """`chatcmpl-...`."""
    object: Literal["chat.completion"]
    created: int
    """Unix time, in seconds."""
    model: str
    """The file llama-swap loaded, not the entry name that was requested."""
    system_fingerprint: str
    """The server build, e.g. `b11429-d81235049`."""
    choices: list[LlamaChoice]
    """One per completion: `n` of them."""
    usage: LlamaUsage
    timings: LlamaTimings
    """Sent when the slot recorded timings (in every reply seen so far)."""


class LlamaChunk(TypedDict, total=False):
    """A `chat.completion.chunk` stream event.

    Always `id`, `object`, `created`, `model`, `system_fingerprint` and
    `choices`. `choices` is empty only in the usage chunk at the end (with
    `stream_options.include_usage`), which carries `usage`; `timings` goes on
    the last chunk sent, and on others with `timings_per_token`.
    """

    id: str
    object: Literal["chat.completion.chunk"]
    created: int
    model: str
    system_fingerprint: str
    choices: list[LlamaChunkChoice]
    usage: LlamaUsage
    timings: LlamaTimings
    prompt_progress: LlamaPromptProgress


class LlamaError(TypedDict, total=False):
    """The `error` object of an HTTP error reply or a stream `error` event.

    Always `code` (the HTTP status), `message` and `type`. llama-swap's own
    errors (e.g. 404 for an unknown model) are plain text, not this.
    """

    code: int
    message: str
    type: Literal[
        "invalid_request_error",
        "authentication_error",
        "not_found_error",
        "server_error",
        "permission_error",
        "not_supported_error",
        "unavailable_error",
        "exceed_context_size_error",
    ]
    n_prompt_tokens: int
    """With `exceed_context_size_error`: tokens in the rejected prompt."""
    n_ctx: int
    """With `exceed_context_size_error`: the context window."""
