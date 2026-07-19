"""Persistence and streaming — the two features that make agents feel real.

PERSISTENCE. The API is stateless, so every agent so far has kept its memory
in a Python list that dies with the process. A LangGraph *checkpointer*
saves the graph's state (the message list) to a database after every node,
keyed by a `thread_id`. Same thread_id -> the conversation continues where
it left off, even across process restarts (with a file/server-backed DB).
Different thread_id -> a completely fresh conversation. One compiled graph
can serve any number of independent threads — that's one agent process
serving many users.

The demo below asks about SF weather, then "What about in LA?", then "Which
one is warmer?" on the SAME thread — each turn only makes sense because the
checkpointer restored the previous ones. Then it asks "Which one is warmer?"
on a NEW thread, where the agent has never heard of SF or LA.

STREAMING. `graph.astream_events()` surfaces every event inside the run,
including each token as the model generates it — print them as they arrive
instead of making the user stare at a spinner for the whole response.

Run:  python agent_persistence.py
"""

import asyncio
import sqlite3

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

import tools as registry
from agent_class import MODEL, SYSTEM_PROMPT, Agent

load_dotenv()


def make_agent(checkpointer) -> Agent:
    return Agent(
        model=ChatAnthropic(model=MODEL, max_tokens=16000),
        tool_schemas=registry.tool_schemas(),
        system=SYSTEM_PROMPT,
        checkpointer=checkpointer,
    )


# ---------------------------------------------------------------------------
# Part 1: persistence — same thread remembers, new thread doesn't
# ---------------------------------------------------------------------------

def demo_persistence() -> None:
    # In-memory SQLite: perfect for a demo, gone when the process exits.
    # Point it at a file (sqlite3.connect("checkpoints.db")) and the
    # conversations survive restarts too.
    checkpointer = SqliteSaver(sqlite3.connect(":memory:", check_same_thread=False))
    agent = make_agent(checkpointer)

    conversation = [
        ("1", "What's the weather in SF?"),
        ("1", "What about in LA?"),          # "in LA" only parses with turn 1
        ("1", "Which one is warmer?"),       # needs BOTH previous answers
        ("2", "Which one is warmer?"),       # fresh thread: no idea what "one" is
    ]
    for thread_id, question in conversation:
        print(f"\n=== thread {thread_id} | user: {question} " + "=" * 20)
        answer = agent.run(question, thread_id=thread_id)
        print(f"[answer] {answer}")


# ---------------------------------------------------------------------------
# Part 2: streaming — print tokens as the model generates them
# ---------------------------------------------------------------------------

def _chunk_text(chunk) -> str:
    """Anthropic stream chunks carry a list of content blocks; extract text."""
    if isinstance(chunk.content, str):
        return chunk.content
    return "".join(
        block.get("text", "")
        for block in chunk.content
        if isinstance(block, dict) and block.get("type") == "text"
    )


async def demo_streaming() -> None:
    print("\n=== streaming | user: What's the weather in SF? " + "=" * 20)
    # astream_events drives the graph async, so it needs the async saver.
    async with AsyncSqliteSaver.from_conn_string(":memory:") as checkpointer:
        agent = make_agent(checkpointer)
        config = {"configurable": {"thread_id": "streaming-demo"}}
        async for event in agent.graph.astream_events(
            {"messages": [("user", "What's the weather in SF?")]}, config
        ):
            if event["event"] == "on_chat_model_stream":
                print(_chunk_text(event["data"]["chunk"]), end="", flush=True)
    print()  # final newline after the streamed text


if __name__ == "__main__":
    demo_persistence()
    asyncio.run(demo_streaming())
