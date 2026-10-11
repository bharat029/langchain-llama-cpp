from langchain_core.messages import AIMessage, HumanMessage

from langchain_llama_cpp.utils import pretty_repr

TITLE = (
    "================================== Ai Message =================================="
)
REASONING = {"reasoning_content": "User asks capital of France..."}


def test_reasoning_from_additional_kwargs() -> None:
    msg = AIMessage(content="Paris.", additional_kwargs=REASONING)
    assert pretty_repr(msg) == (
        f"{TITLE}\n\nReasoning:\n  User asks capital of France...\n\nParis."
    )


def test_v1_content_blocks_render_like_v0() -> None:
    v0 = AIMessage(content="Paris.", additional_kwargs=REASONING)
    # What the base class does for output_version="v1": kwargs keep reasoning too.
    v1 = v0.model_copy(update={"content": v0.content_blocks})
    # Blocks only, as from a provider that puts reasoning in content.
    blocks_only = v1.model_copy(update={"additional_kwargs": {}})
    assert pretty_repr(v1) == pretty_repr(v0)
    assert pretty_repr(blocks_only) == pretty_repr(v0)


def test_no_reasoning_matches_core() -> None:
    msg = AIMessage(content="Paris.")
    assert pretty_repr(msg) == msg.pretty_repr()


def test_multiline_reasoning_is_indented() -> None:
    msg = AIMessage(content="4", additional_kwargs={"reasoning_content": "2+2\n= 4"})
    assert "Reasoning:\n  2+2\n  = 4\n\n4" in pretty_repr(msg)


def test_tool_calls_and_name_are_kept() -> None:
    msg = AIMessage(
        content="",
        name="agent",
        additional_kwargs=REASONING,
        tool_calls=[{"name": "get_weather", "args": {"city": "Paris"}, "id": "1"}],
    )
    assert pretty_repr(msg).startswith(
        f"{TITLE}\nName: agent\n\nReasoning:\n"
        "  User asks capital of France...\nTool Calls:\n  get_weather (1)"
    )


def test_non_ai_message_uses_core() -> None:
    msg = HumanMessage(content="What is the capital of France?")
    assert pretty_repr(msg) == msg.pretty_repr()
