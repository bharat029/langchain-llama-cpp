"""Content-block translator for llama-server messages (`model_provider` "llama-cpp").

Core builds `content_blocks` for providers without a translator by best-effort
parsing, and that is what we want, except for one thing. When streaming with
`output_version="v1"`, core numbers blocks that have no `index` from 0 (the
reasoning or text block), while a tool-call chunk keeps the server's tool-call
index (also 0). Merging chunks then treats the two as one block: the tool call
is folded into the reasoning block and disappears from `content_blocks`.

The translator returns core's own blocks with collision-free indexes, the
scheme core's OpenAI translator uses: `lc_rs` for reasoning, `lc_txt` for text
and `lc_tc_<i>` for tool-call chunks. `merge_lists` only merges `lc_` indexes
that are equal.
"""

from typing import TYPE_CHECKING, Any, TypeVar, cast

from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.messages.block_translators import register_translator

if TYPE_CHECKING:
    from langchain_core.messages import content as types

MODEL_PROVIDER = "llama-cpp"

_M = TypeVar("_M", AIMessage, AIMessageChunk)

_INDEX_BY_TYPE = {"reasoning": "lc_rs", "text": "lc_txt"}


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
    """Blocks for a streamed chunk: core's, with collision-free indexes."""
    blocks = _default_blocks(message)
    for block in blocks:
        mutable = cast("dict[str, Any]", block)
        block_type = mutable.get("type")
        if block_type in _INDEX_BY_TYPE:
            mutable["index"] = _INDEX_BY_TYPE[block_type]
        elif block_type == "tool_call_chunk" and isinstance(mutable.get("index"), int):
            mutable["index"] = f"lc_tc_{mutable['index']}"
    return blocks


register_translator(MODEL_PROVIDER, translate_content, translate_content_chunk)
