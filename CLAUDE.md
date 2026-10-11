# langchain-llama-cpp

A LangChain chat model (`ChatLlama`) for llama.cpp's `llama-server`, used directly or behind llama-swap. It's built on `langchain-core` alone, with no `langchain-openai` or `openai` dependency, and talks to `/v1/chat/completions` (plus the native endpoints) over `httpx`.

**Why it exists:** `ChatOpenAI` drops `reasoning_content` and `timings`, and it sends `tool_choice` as an object that llama-server silently turns into `"auto"`. `ChatDeepSeek` rewrites `json_schema` into tool calling. `langchain-llama-server` (PyPI) subclasses `BaseChatOpenAI` and crashes on `json_schema`/`json_mode`.

## Decisions so far
- **Name:** distribution `langchain-llama-cpp`, import `langchain_llama_cpp`, class `ChatLlama`.
- **Servers (v0): llama-swap only.** `model` picks the entry; native endpoints live at `/upstream/<model>/...`. No `server` field and no detection.
  - Later: bare llama-server and auto-detection, maybe via llama-server's own router mode replacing llama-swap. The plan then: a lazy `GET /running` probe on the first call, cached, with no network at construction, matching ChatOpenAI, ChatOllama and ChatDeepSeek; optionally an opt-in eager check like ChatOllama's `validate_model_on_init`.
- **Parameters:** a typed field for **every** request field `/v1/chat/completions` accepts. The list comes from the llama.cpp source for the server's build (b11429: `tools/server/server-common.cpp` `oaicompat_chat_params_parse`, `server-schema.cpp` `make_llama_cmpl_schema`, `server-context.cpp` `id_slot`). Each field is sent under its own name; `None` (the default) is left out, so the server's per-model defaults apply. `test_request_fields_match_llama_server` pins the list (60 fields); update it when the server build changes.
  - Not typed, because another API owns them: `messages`; `tools`, `tool_choice`, `parallel_tool_calls` (milestone 4, `bind_tools`); `response_format` (milestone 3, `with_structured_output`); `stream`, `stream_options` (milestone 2).
  - Not typed, because the chat layer overwrites them (template plumbing): `prompt`, `chat_format`, `generation_prompt`, `chat_parser`, `grammar_lazy`, `grammar_triggers`, `preserved_tokens`, `parse_tool_calls`, `message_delimiters`, `reasoning_budget_start_tag`, `reasoning_budget_end_tags`.
  - Aliases, not separate fields: `n_predict`/`max_completion_tokens` → `max_tokens`, `n_cmpl` → `n`, `thinking_budget_tokens` → `reasoning_budget_tokens`.
  - `verbose` clashes with core's own `verbose` field (callback verbosity), so it isn't typed; pass it per call with `invoke(..., verbose=True)` if you need it.
  - Speculative-decoding fields are compiled out (`#if 0`) in b11429.
- **Python** 3.10+. Dependencies: `langchain-core>=1.0,<2`, `httpx>=0.27,<1`.
- **Milestones**, each ending with passing tests and a review:
  1. `invoke`: message conversion, reasoning, usage, `response_metadata`, clear errors for 400 overflow and 404.
  2. `stream` and real async (`_stream`, `_agenerate`, `_astream`), including streamed tool-call chunks (moved here from milestone 4 so streaming never drops tool calls).
  3. Structured output: `json_schema`, `function_calling`, `json_mode`, plus an opt-in flag that injects the schema into the prompt.
  4. Tools: `bind_tools`, `tool_choice` limited to what the server accepts. A small agent loop as the tool test.
  5. `langchain-tests` (unit and integration), run on several model families.

## Server behaviour the code depends on (llama-server b11429, verified)
- **Request:** OpenAI shape plus llama.cpp extras: `top_k`, `min_p`, `samplers` (order), `chat_template_kwargs`, `cache_prompt`, `post_sampling_probs`, `n` (1–4).
  - Unknown fields are silently ignored (HTTP 200).
  - `max_tokens` or `max_completion_tokens` both work.
- **`tool_choice`:** strings only, `auto`/`none`/`required`. An object logs "Wrong type supplied for parameter 'tool_choice'" and falls back to `auto`. With `none`, the model still sees the tools and may write a call as plain text.
  - **`required` differs per chat handler** (b11429 `common/parsers/*.cpp`; gemma and qwen use specialized handlers, chosen in `chat.cpp:1200-1226`, not the autoparser): gemma4 allows reasoning then the tool call immediately (`start + thought + tool_call`); gpt-oss allows analysis messages then the call. **The `qwen` entry (Qwen3.6-35B-A3B, a general model) goes through the handler llama.cpp names "qwen3-coder"** (chosen because its template has `<tool_call>`, `<function=`, `<parameter=`; also used by Qwen3.5, Nemotron Nano 3, StepFun). That handler **allows free text before the call** (`content(until "<tool_call>") << tool_calls`, at least one call) and so bans end-of-turn until a call appears. A qwen that doesn't want a tool keeps writing until `max_tokens` (verified: "Hello!" → emoji loop, `finish_reason: length`, no call).
    - Upstream: llama.cpp **#27217** (open) reports exactly this on Qwen3.5/3.6, and a contributor traced it to the same line. #27767 (same symptom, Qwen3.6) was closed as stale. gemma4 had the same flaw and was fixed in **#29115** (merged 2026-09-19, in b11429), which is why gemma's `required` now goes straight to the call. Documented in `bind_tools`/`with_structured_output`; no client workaround (user's decision).
    - Named `tool_choice` upstream: PR #26849 proposed narrowing `tools` to the named function plus `required` (our M4 option 1); it was cut down to a Muse Glimmer fix (#26879, merged) and the named part did not land.
  - **Decision (M4):** `bind_tools` sends `auto`/`none`/`required` as they are; `"any"`, `True`, a bound tool's name and the `{"type":"function","function":{"name"}}` object all become `"required"`, with every bound tool still sent. Forcing one particular tool is the server's job, not ours. **Revisit when the server build changes:** if llama-server starts honouring the object form, send it instead (`_tool_choice` in `chat_models.py`).
  - Probed 2026-10-11 on gemma, `required` with `test_tool_choice`'s two tools: "Hello!" gave `magic_function` 10/10 in either list order (the reasoning said no tool was needed; the grammar forced a call with throwaway input). "What's the weather in Paris?" gave `get_weather` 5/5, so a named `magic_function` would not be honoured there.
- **`response_format`:** `json_object` or `json_schema` (`{"type":"json_schema","json_schema":{"name","schema","strict"}}`).
  - It's **grammar-only**: the schema and its descriptions never reach the prompt.
  - Enforced: integer bounds, string length/pattern/format, array length, enum/const, required, additionalProperties.
  - Not enforced: `number` bounds, `multipleOf`.
  - With thinking on, the reasoning comes first and the grammar applies only to `content`.
- **Response:** `message.reasoning_content` (separate from `content`) and `tool_calls[].function.arguments` (a JSON string).
  - `usage`: `prompt_tokens`, `completion_tokens`, `total_tokens`, `prompt_tokens_details.cached_tokens`.
    - Probed 2026-10-11 on gemma with thinking on: there is **no `completion_tokens_details`**. Reasoning tokens are included in `completion_tokens` (212 for a 1-token answer) but not reported separately.
    - `cached_tokens` is part of `prompt_tokens`: a repeat of a 32-token prompt gave `cached_tokens: 27`, `cache_n: 27`, `prompt_n: 5`.
  - `timings`: `cache_n`, `prompt_n`, `prompt_ms`, `prompt_per_token_ms`, `prompt_per_second`, `predicted_n`, `predicted_ms`, `predicted_per_token_ms`, `predicted_per_second`.
  - `model` is the file llama-swap loaded (e.g. `google/gemma-4-26B-A4B-it-qat-q4_0-gguf:Q4_0`), not the alias requested (`gemma`). `system_fingerprint` is the build (`b11429-d81235049`). `id` is `chatcmpl-...`.
- **Streaming:** SSE `data: {...}` and a final `data: [DONE]`.
  - Each `delta` carries `content`, `reasoning_content` or `tool_calls` fragments (`index`; the id and name come first, the arguments in pieces).
  - With `stream_options.include_usage`, a final chunk with `choices: []` carries `usage` and `timings`.
  - Errors (b11429 `server-context.cpp:4731-4868`): an error before the first chunk is a normal HTTP error response; a later one arrives as `data: {"error": {...}}` and the stream ends without `[DONE]`. Lines starting with `:` are keep-alive pings (`sse_ping_interval`).
  - Core concatenates string metadata when merging chunks (except `model_provider`, `id`, `output_version`), so `finish_reason`, `model_name` and `system_fingerprint` go on the finish chunk only.
- **Errors:**
  - Prompt over the window: plain JSON (even when streaming) HTTP 400 `{"error":{"type":"exceed_context_size_error","message","n_prompt_tokens","n_ctx"}}`.
  - A reply that runs into the window: `finish_reason: "length"`. Nothing is truncated silently.
  - llama-swap with an unknown model: 404.
- **Native endpoints:** `/props` (`default_generation_settings.n_ctx`, defaults, `chat_template`, `chat_template_caps`), `/apply-template` (`{messages, tools?, chat_template_kwargs?}` → `{prompt}`), `/tokenize`. Through llama-swap, these load the model.
- **Thinking switches** (`chat_template_kwargs`) differ per model family: gemma/qwen `enable_thinking: bool`; gpt-oss `reasoning_effort: low|medium|high` (it always reasons). Thinking tokens count towards `max_tokens`.
- **Chat templates** ignore `name` on system/user/assistant messages.
- **Reasoning in history** (probed 2026-10-11 with `/apply-template`): llama-server passes an assistant message's `reasoning_content` to the template, and the template decides what to render.
  - Every model keeps the reasoning of the current tool-call turn: gemma `<|channel>thought`, qwen `<think>`, gpt-oss `<|channel|>analysis`.
  - gemma and gpt-oss drop reasoning from earlier turns.
  - qwen keeps earlier turns by default (`chat_template_caps.supports_preserve_reasoning: true`); `chat_template_kwargs: {"preserve_thinking": false}` drops them.
  - **Decision (M1-c):** always send `reasoning_content` when an `AIMessage` has it, converting v1 reasoning blocks back. Never strip it on our side. This is what ChatOllama does (sent back as `thinking`); ChatOpenAI, ChatDeepSeek and ChatGroq drop it.

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
