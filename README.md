# langchain-llama-cpp

[![PyPI - Version](https://img.shields.io/pypi/v/langchain-llama-cpp?label=%20)](https://pypi.org/project/langchain-llama-cpp/)
[![PyPI - License](https://img.shields.io/pypi/l/langchain-llama-cpp)](https://opensource.org/licenses/MIT)

A LangChain chat model, `ChatLlama`, for llama.cpp's `llama-server`. It's built
on `langchain-core` and `httpx` only and talks to `/v1/chat/completions`
directly, so nothing llama-server sends back is lost on the way.

**Why not `ChatOpenAI`?** llama-server speaks the OpenAI format, but with extras
that `ChatOpenAI` drops or gets wrong:

- it drops `reasoning_content` (the model's thinking) and llama-server's `timings`;
- it sends a named `tool_choice` as an object, which llama-server silently turns
  into `"auto"`;
- it can't send llama.cpp's own sampling fields (`top_k`, `min_p`, `samplers`,
  DRY, XTC, ...) as typed parameters.

## Install

```bash
pip install langchain-llama-cpp
```

or `uv add langchain-llama-cpp`. Python 3.10+.

## Tested setup

- **llama-swap** in front of llama-server. `model` is the llama-swap entry to
  use. A bare llama-server (no llama-swap) is not tested yet.
- llama-server build **b11429**, with gemma 4, Qwen3.6 and gpt-oss, on
  langchain-core 1.6.

## Quick start

```python
from langchain_llama_cpp import ChatLlama

llm = ChatLlama(model="gemma", max_tokens=2000)  # base_url from LLAMA_BASE_URL
msg = llm.invoke("What is 17 * 23? Answer with the number only.")

msg.content  # '391'
msg.additional_kwargs["reasoning_content"]  # 'The user wants to know the product...'
msg.usage_metadata  # {'input_tokens': 32, 'output_tokens': 223, ...}
msg.response_metadata["timings"]  # {'prompt_n': 25, 'predicted_n': 223, ...}
```

Set `max_tokens` with room for thinking. Reasoning length varies from run to
run: gemma answered this in 223 tokens once, and in another run used all of
`max_tokens=300` and ended mid-thought, with an empty `content` and
`finish_reason: "length"`.

`base_url` defaults to the `LLAMA_BASE_URL` environment variable, e.g.
`http://localhost:8080`. Pass `base_url=` to override it.

Reasoning is stored where ChatDeepSeek and ChatOllama put it,
`additional_kwargs["reasoning_content"]`, so `msg.content_blocks` shows it as a
standard `{"type": "reasoning"}` block. `langchain_llama_cpp.utils.pretty_print(msg)`
prints a message with its reasoning.

## Usage

### Thinking

The switch lives in the chat template, and its name differs per model family:

```python
ChatLlama(model="gemma", chat_template_kwargs={"enable_thinking": False})  # gemma, qwen
ChatLlama(model="gpt-oss", chat_template_kwargs={"reasoning_effort": "low"})  # gpt-oss
```

Thinking tokens count towards `max_tokens` (see Quick start).

When you send a conversation back, an `AIMessage`'s reasoning goes back too, and
the model's chat template decides whether to show it to the model.

### Streaming and async

```python
for chunk in llm.stream("Count to five."):
    print(chunk.content, end="")

msg = await llm.ainvoke("Hello!")
async for chunk in llm.astream("Hello!"):
    ...
```

Reasoning arrives in `chunk.additional_kwargs["reasoning_content"]`. Usage and
timings come in the last chunk (`stream_usage=True`, the default).

### Tool calling

```python
from langchain_core.tools import tool


@tool
def get_weather(city: str) -> str:
    """Get the current weather in a city."""
    return f"Sunny in {city}"


llm_with_tools = llm.bind_tools([get_weather])
msg = llm_with_tools.invoke("What's the weather in Paris?")
msg.tool_calls  # [{'name': 'get_weather', 'args': {'city': 'Paris'}, 'id': '...', ...}]
```

llama-server accepts only the strings `"auto"`, `"none"` and `"required"` as
`tool_choice`. A tool's name (or `"any"`, or `True`) is sent as `"required"`:
the model must call a tool but may pick a different one, because the server
can't force a particular tool.

### Structured output

```python
from pydantic import BaseModel, Field


class Person(BaseModel):
    name: str
    age: int = Field(description="Age in years")


structured = llm.with_structured_output(Person)
structured.invoke("Alice is 30.")  # Person(name='Alice', age=30)
```

Three methods:

| `method` | What's sent | The model sees the schema? |
|---|---|---|
| `"function_calling"` (default) | the schema as the only tool, `tool_choice: "required"` | yes, through the chat template |
| `"json_schema"` | `response_format: {"type": "json_schema", ...}` | no: it's enforced by the grammar only |
| `"json_mode"` | `response_format: {"type": "json_object"}` | no |

`function_calling` is the default because with `json_schema` the model never
sees the schema's field descriptions: asked for an age described as "in months",
it fills in years. Pass `include_format_instructions=True` with `json_schema` or
`json_mode` to add the schema to the system message.

### Request parameters

Every field `/v1/chat/completions` accepts in b11429 is a typed parameter:
`temperature`, `top_k`, `top_p`, `min_p`, `samplers`, `repeat_penalty`, the DRY
and XTC samplers, `grammar`, `json_schema`, `reasoning_budget_tokens`,
`cache_prompt`, `id_slot`, `lora` and more. Set them on the model or per call:

```python
llm = ChatLlama(model="gemma", temperature=0.2, top_k=40)
llm.invoke("Hi", max_tokens=500)
```

They all default to `None`, which leaves them out of the request so the
server's per-model defaults apply. To see those defaults:

```python
llm.get_server_defaults()
# {'temperature': 1.0, 'top_k': 64, 'top_p': 0.95, 'max_tokens': -1, ..., 'n_ctx': 131072}
```

### Errors and retries

Errors are subclasses of both the httpx exception and LangChain's `ModelError`
classes, so either `except` works:

```python
from langchain_llama_cpp import LlamaContextOverflowError

try:
    llm.invoke(very_long_prompt)
except LlamaContextOverflowError as error:
    print(error.n_prompt_tokens, error.n_ctx)
```

There are also `LlamaModelNotFoundError` (an unknown llama-swap entry),
`LlamaInvalidRequestError`, `LlamaRateLimitError`, `LlamaAPIError` (5xx),
`LlamaConnectionError` and `LlamaTimeoutError`.

Connection errors, timeouts, 429 and 5xx are retried `max_retries` times
(default 2) with exponential backoff, obeying `Retry-After`. `timeout` is in
seconds and defaults to `None` (no timeout), since local generation can be slow.

## Known limitations

- **Forcing a tool with Qwen3.5/3.6.** With `tool_choice="required"` (also used
  by `with_structured_output`'s default method), llama-server's grammar for these
  models lets the model write text before the call and won't let it stop
  without one. A model that doesn't want a tool keeps writing until
  `max_tokens`, and there's no tool call. This is
  [ggml-org/llama.cpp#27217](https://github.com/ggml-org/llama.cpp/issues/27217);
  keep `max_tokens` set.
- **A named `tool_choice`** is sent as `"required"` (see Tool calling).
- **JSON schema bounds on numbers** (`minimum`, `maximum`, `multipleOf` on
  `number`) aren't enforced by llama-server's grammar. Integer bounds are.
- **Bare llama-server** (without llama-swap) isn't tested; `get_server_defaults`
  needs llama-swap's `/upstream/<model>/props`.

## Development

Bug reports and pull requests are welcome on
[GitHub](https://github.com/bharat029/langchain-llama-cpp/issues).

```bash
uv sync
uv run pytest tests/unit_tests          # offline
uv run ruff check && uv run ruff format --check && uv run pyright
```

The integration tests (LangChain's standard `langchain-tests` suite, on gemma,
qwen and gpt-oss) need a llama-swap server. Copy `.env.example` to `.env` and
set `LLAMA_BASE_URL`, then:

```bash
uv run --env-file .env pytest tests/integration_tests
```

The JSON llama-server sends back is described in
`src/langchain_llama_cpp/_wire.py`.

## License

MIT
