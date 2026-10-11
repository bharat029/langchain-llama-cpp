"""Content-block translator for llama-server messages (`model_provider` "llama-cpp").

Core builds `content_blocks` for providers without a translator by best-effort
parsing, and that is what we want, except when streaming with
`output_version="v1"`. Core then numbers blocks that have no `index` with a
counter that goes up whenever the block type changes (reasoning 0, text 1, ...),
while a tool-call chunk keeps the server's tool-call index (0, 1, ...). Equal
indexes are merged as one block, so the first tool call was folded into the
reasoning or text block before it and vanished from `content_blocks`.

For streamed chunks the translator returns core's blocks with two changes:

- tool-call chunks get `lc_tc_<i>` indexes, which can't equal a counter value
  (`merge_lists` merges `lc_` indexes only with the same string). Reasoning and
  text keep core's counter, so text after a tool call stays a separate block.
- every chunk that carries tool-call fragments gets them as `tool_call_chunk`
  blocks. Core does that only when the chunk has no text; otherwise it adds a
  partially parsed `tool_call` that would survive next to the finished call.

The blocks are copies: reading `content_blocks` never changes the message.
"""

from typing import TYPE_CHECKING, Any, TypeVar

from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.messages.block_translators import register_translator

if TYPE_CHECKING:
    from langchain_core.messages import content as types

MODEL_PROVIDER = "llama-cpp"

_M = TypeVar("_M", AIMessage, AIMessageChunk)

_TOOL_CALL_TYPES = {"tool_call", "invalid_tool_call", "tool_call_chunk"}


def _default_blocks(message: _M) -> list["types.ContentBlock"]:
    """Core's best-effort blocks: a copy without our provider skips the registry."""
    metadata = {
        key: value
        for key, value in message.response_metadata.items()
        if key != "model_provider"
    }
    return message.model_copy(update={"response_metadata": metadata}).content_blocks


def translate_content(message: AIMessage) -> list["types.ContentBlock"]:
    """Blocks for a whole message: core's, unchanged (nothing is merged)."""
    return _default_blocks(message)


def translate_content_chunk(message: AIMessageChunk) -> list["types.ContentBlock"]:
    """Blocks for a streamed chunk: core's, with tool-call chunks re-indexed."""
    blocks: list[dict[str, Any]] = [dict(block) for block in _default_blocks(message)]
    if message.tool_call_chunks and message.chunk_position != "last":
        blocks = [block for block in blocks if block["type"] not in _TOOL_CALL_TYPES]
        for chunk in message.tool_call_chunks:
            block: dict[str, Any] = {
                "type": "tool_call_chunk",
                "id": chunk.get("id"),
                "name": chunk.get("name"),
                "args": chunk.get("args"),
            }
            if (index := chunk.get("index")) is not None:
                block["index"] = f"lc_tc_{index}"
            blocks.append(block)
    return blocks  # pyright: ignore[reportReturnType]


register_translator(MODEL_PROVIDER, translate_content, translate_content_chunk)
