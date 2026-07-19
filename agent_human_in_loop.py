"""Human in the loop — pausing, editing, and rewinding an agent.

Because the checkpointer snapshots state after every node, the graph can be
stopped and restarted at any checkpoint. That enables three things no
fire-and-forget loop can do:

  1. APPROVE:     interrupt_before=["action"] pauses the run just before any
                  tool executes; a human inspects the pending call, then
                  resumes with graph.stream(None, thread).
  2. EDIT:        while paused, replace the pending tool call's arguments
                  via graph.update_state() — the agent executes the human's
                  version, not its own.
  3. TIME TRAVEL: graph.get_state_history() lists every past checkpoint;
                  replaying from an earlier one forks a NEW branch of the
                  conversation (the original path is preserved).

The demos use the calculator so the "consequential action" being approved,
edited, or replayed is concrete and its result is checkable.

Run:  python agent_human_in_loop.py
"""

import sqlite3
from collections import Counter

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage
from langgraph.checkpoint.sqlite import SqliteSaver

import tools as registry
from agent_class import MODEL, SYSTEM_PROMPT, Agent

load_dotenv()


def separator(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def stream_events(agent: Agent, graph_input, thread) -> None:
    """Stream the graph, printing what each node produced.

    graph_input=None means "resume from the checkpoint" rather than
    starting a new turn.
    """
    for event in agent.graph.stream(graph_input, thread):
        for node, update in event.items():
            # the pause itself arrives as a '__interrupt__' pseudo-event
            if not isinstance(update, dict) or "messages" not in update:
                continue
            for msg in update["messages"]:
                if isinstance(msg, AIMessage) and msg.tool_calls:
                    calls = ", ".join(
                        f"{c['name']}({c['args']})" for c in msg.tool_calls
                    )
                    print(f"[{node}] wants to run: {calls}")
                elif msg.text:
                    print(f"[{node}] {msg.text}")


def show_pause(agent: Agent, thread):
    """Print where the paused graph stopped and what it wants to do next."""
    state = agent.graph.get_state(thread)
    print(f"[paused] get_state(thread).next = {state.next}")
    for call in state.values["messages"][-1].tool_calls:
        print(f"[pending] {call['name']}({call['args']})")
    return state


# ---------------------------------------------------------------------------
# Demo 1: manual approval
# ---------------------------------------------------------------------------

def demo_approval(agent: Agent) -> None:
    separator("DEMO 1: manual approval (interrupt_before=['action'])")
    thread = {"configurable": {"thread_id": "approve"}}

    stream_events(agent, {"messages": [("user", "What is 1234 * 5678?")]}, thread)
    show_pause(agent, thread)

    print("\n[human] looks right — approving. resuming with stream(None, thread)")
    stream_events(agent, None, thread)


# ---------------------------------------------------------------------------
# Demo 2: modify the pending tool call, then resume
# ---------------------------------------------------------------------------

def demo_modify(agent: Agent) -> None:
    separator("DEMO 2: edit the pending action before resuming")
    thread = {"configurable": {"thread_id": "modify"}}

    stream_events(agent, {"messages": [("user", "What is 999 * 999?")]}, thread)
    state = show_pause(agent, thread)

    # Build a replacement AIMessage with the SAME id but edited args. The
    # add_messages reducer sees the matching id and replaces the original
    # message instead of appending — that's why agent_class.py uses it.
    original = state.values["messages"][-1]
    edited_call = {**original.tool_calls[0], "args": {"expression": "111 * 111"}}
    edited = AIMessage(
        content=original.content, tool_calls=[edited_call], id=original.id
    )
    agent.graph.update_state(thread, {"messages": [edited]})

    print("\n[human] changing the expression to '111 * 111', then resuming")
    show_pause(agent, thread)
    stream_events(agent, None, thread)
    print(
        "\n(the tool ran the HUMAN's expression: 12321 = 111*111, not 998001.\n"
        " note the model may spot that the observation doesn't answer the\n"
        " user's original 999*999 — an edit is only as coherent as the whole\n"
        " history you leave behind. To fully redirect the agent, edit the\n"
        " user message too, not just the tool call.)"
    )


# ---------------------------------------------------------------------------
# Demo 3: time travel — replay from an earlier checkpoint
# ---------------------------------------------------------------------------

def demo_time_travel(agent: Agent) -> None:
    separator("DEMO 3: time travel (get_state_history + replay)")
    thread = {"configurable": {"thread_id": "timetravel"}}

    stream_events(agent, {"messages": [("user", "What is 42 * 42?")]}, thread)
    print("[human] approving")
    stream_events(agent, None, thread)

    history = list(agent.graph.get_state_history(thread))  # newest first
    print(f"\nhistory: {len(history)} checkpoints (newest first)")
    for i, st in enumerate(history):
        n_msgs = len(st.values.get("messages", []))
        print(f"  [{i}] next={st.next or '(end)'}  messages={n_msgs}")

    # Rewind to the moment BEFORE the tool originally ran and replay from
    # there. Streaming from a past checkpoint's config forks the thread —
    # the original path above is untouched.
    past = next(st for st in reversed(history) if st.next == ("action",))
    print(f"\n[human] replaying from the checkpoint where next={past.next}")
    stream_events(agent, None, past.config)

    after = list(agent.graph.get_state_history(thread))
    parents = Counter(
        st.parent_config["configurable"]["checkpoint_id"]
        for st in after
        if st.parent_config
    )
    forks = sum(1 for count in parents.values() if count > 1)
    print(
        f"\nhistory grew: {len(history)} -> {len(after)} checkpoints, and "
        f"{forks} checkpoint now has TWO children — a fork, not a rerun."
    )


if __name__ == "__main__":
    agent = Agent(
        model=ChatAnthropic(model=MODEL, max_tokens=16000),
        tool_schemas=registry.tool_schemas(),
        system=SYSTEM_PROMPT,
        checkpointer=SqliteSaver(
            sqlite3.connect(":memory:", check_same_thread=False)
        ),
        interrupt_before=["action"],
    )
    demo_approval(agent)
    demo_modify(agent)
    demo_time_travel(agent)
