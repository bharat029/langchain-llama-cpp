# langchain-llama-cpp

A LangChain chat model (`ChatLlama`) for llama.cpp's `llama-server`, used directly or behind llama-swap. It's built on `langchain-core` alone, with no `langchain-openai` or `openai` dependency, and talks to `/v1/chat/completions` (plus the native endpoints) over `httpx`.

**Why it exists:** `ChatOpenAI` drops `reasoning_content` and `timings`, and it sends `tool_choice` as an object that llama-server silently turns into `"auto"`. `ChatDeepSeek` rewrites `json_schema` into tool calling. `langchain-llama-server` (PyPI) subclasses `BaseChatOpenAI` and crashes on `json_schema`/`json_mode`.

## Decisions so far
- **Name:** distribution `langchain-llama-cpp`, import `langchain_llama_cpp`, class `ChatLlama`.
- **Servers (v0): llama-swap only.** `model` picks the entry; native endpoints live at `/upstream/<model>/...`. No `server` field and no detection.
  - Later: bare llama-server and auto-detection, maybe via llama-server's own router mode replacing llama-swap. The plan then: a lazy `GET /running` probe on the first call, cached, with no network at construction, matching ChatOpenAI, ChatOllama and ChatDeepSeek; optionally an opt-in eager check like ChatOllama's `validate_model_on_init`.
- **Parameters:** typed fields for every llama.cpp option we support. A field left at `None` is left out of the request.
- **Python** 3.10+. Dependencies: `langchain-core>=1.0,<2`, `httpx>=0.27,<1`.
- **Milestones**, each ending with passing tests and a review:
  1. `invoke`: message conversion, reasoning, usage, `response_metadata`, clear errors for 400 overflow and 404.
  2. `stream` and real async (`_stream`, `_agenerate`, `_astream`).
  3. Structured output: `json_schema`, `function_calling`, `json_mode`, plus an opt-in flag that injects the schema into the prompt.
  4. Tools: `bind_tools`, streamed tool-call chunks, `tool_choice` limited to what the server accepts. A small agent loop as the tool test.
  5. `langchain-tests` (unit and integration), run on several model families.

## Server behaviour the code depends on (llama-server b11429, verified)
- **Request:** OpenAI shape plus llama.cpp extras: `top_k`, `min_p`, `samplers` (order), `chat_template_kwargs`, `cache_prompt`, `post_sampling_probs`, `n` (1–4).
  - Unknown fields are silently ignored (HTTP 200).
  - `max_tokens` or `max_completion_tokens` both work.
- **`tool_choice`:** strings only, `auto`/`none`/`required`. An object logs "Wrong type supplied for parameter 'tool_choice'" and falls back to `auto`. With `none`, the model still sees the tools and may write a call as plain text.
- **`response_format`:** `json_object` or `json_schema` (`{"type":"json_schema","json_schema":{"name","schema","strict"}}`).
  - It's **grammar-only**: the schema and its descriptions never reach the prompt.
  - Enforced: integer bounds, string length/pattern/format, array length, enum/const, required, additionalProperties.
  - Not enforced: `number` bounds, `multipleOf`.
  - With thinking on, the reasoning comes first and the grammar applies only to `content`.
- **Response:** `message.reasoning_content` (separate from `content`) and `tool_calls[].function.arguments` (a JSON string).
  - `usage`: `prompt_tokens`, `completion_tokens`, `prompt_tokens_details.cached_tokens`.
  - `timings`: `cache_n`, `prompt_n`, `prompt_ms`, `predicted_n`, `predicted_ms`, ...
- **Streaming:** SSE `data: {...}` and a final `data: [DONE]`.
  - Each `delta` carries `content`, `reasoning_content` or `tool_calls` fragments (`index`; the id and name come first, the arguments in pieces).
  - With `stream_options.include_usage`, a final chunk with `choices: []` carries `usage` and `timings`.
- **Errors:**
  - Prompt over the window: plain JSON (even when streaming) HTTP 400 `{"error":{"type":"exceed_context_size_error","message","n_prompt_tokens","n_ctx"}}`.
  - A reply that runs into the window: `finish_reason: "length"`. Nothing is truncated silently.
  - llama-swap with an unknown model: 404.
- **Native endpoints:** `/props` (`default_generation_settings.n_ctx`, defaults, `chat_template`, `chat_template_caps`), `/apply-template` (`{messages, tools?, chat_template_kwargs?}` → `{prompt}`), `/tokenize`. Through llama-swap, these load the model.
- **Thinking switches** (`chat_template_kwargs`) differ per model family: gemma/qwen `enable_thinking: bool`; gpt-oss `reasoning_effort: low|medium|high` (it always reasons). Thinking tokens count towards `max_tokens`.
- **Chat templates** ignore `name` on system/user/assistant messages.

## langchain-core facts the code depends on (1.6.9)
- **Abstract methods:** `_generate(messages, stop, run_manager, **kwargs) -> ChatResult` and the `_llm_type` property. Optional: `_stream`, `_agenerate`, `_astream`, `bind_tools`, `with_structured_output`, `_identifying_params`, `_get_ls_params`.
- **The base class does some things itself:**
  - It calls `run_manager.on_llm_new_token` for each streamed chunk. Don't call it in `_stream`.
  - It adds the final `chunk_position="last"` chunk.
  - It routes `invoke` through `_stream` when `stream=True`, when a `streaming=True` field is set, or when a streaming callback is attached (`astream_events` and LangGraph).
- **Reasoning convention:** `additional_kwargs["reasoning_content"]`. `msg.content_blocks` turns it into a `{"type": "reasoning"}` block. `convert_to_openai_messages` drops it.

## Conventions
- `uv` for everything: `uv run pytest`, `uv run ruff check`, `uv run ruff format`, `uv run pyright`. Ruff: `select = ["ALL"]` with the LangChain partner ignores and Google docstrings. Pyright: strict on `src/`, standard on `tests/`, configured in `pyproject.toml` so Pylance uses the same settings.
- Unit tests run offline (`httpx.MockTransport`). Integration tests need a server: `LLAMA_BASE_URL` (see `.env.example`); keep every call capped with `max_tokens`.
