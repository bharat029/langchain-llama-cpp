"""Helpers for working with messages from reasoning models."""

from textwrap import indent

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.utils.interactive_env import is_interactive_env


def pretty_repr(message: BaseMessage, *, html: bool = False) -> str:
    """Like `message.pretty_repr()`, but shows an AI message's reasoning.

    Reasoning is read from `content_blocks`, so it works for reasoning stored in
    `additional_kwargs["reasoning_content"]` (ChatLlama, ChatDeepSeek, ChatOllama,
    ChatGroq) and for reasoning blocks in `content` (`output_version="v1"`,
    ChatAnthropic). Non-AI messages are passed to their own `pretty_repr`.
    """
    if not isinstance(message, AIMessage):
        return message.pretty_repr(html=html)
    parts = [
        block.get("reasoning", "")
        for block in message.content_blocks
        if block["type"] == "reasoning"
    ]
    reasoning = "\n\n".join(part for part in parts if part)
    # Core prints list content as a raw repr, so hand it text only, with the
    # reasoning section in front; core still adds the title, name and tool calls.
    section = f"Reasoning:\n{indent(reasoning, '  ')}" if reasoning else ""
    content = "\n\n".join(part for part in (section, message.text) if part)
    return message.model_copy(update={"content": content}).pretty_repr(html=html)


def pretty_print(message: BaseMessage) -> None:
    """Print `pretty_repr(message)`, with a bold title in Jupyter."""
    print(pretty_repr(message, html=is_interactive_env()))  # noqa: T201
