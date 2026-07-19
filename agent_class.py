"""The ReAct agent wrapped in a reusable class.

agent_langgraph.py builds one graph at module level and runs it as a script.
This version packages the same graph inside an `Agent` class — the shape
you'd actually embed in an application: construct once (graph compiled in
__init__), then call `.run()` as many times as you like, with the model,
tools, and system prompt all injected.

Structure follows the classic LangGraph course pattern:

    llm (call_llm) --exists_action?--> action (take_action) --> llm --> ... --> END

New behavior vs. the earlier files: self-correction on hallucinated tool
names. If the model asks for a tool that isn't in the registry, we don't
crash — we return "bad tool name, retry" as the tool result. The model reads
that observation on the next iteration and corrects itself.

Run:  python agent_class.py "What is 1234 * 5678?"
"""

import operator
import sys
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AnyMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph

import tools as registry

load_dotenv()  # reads ANTHROPIC_API_KEY from a local .env file, if present

MODEL = "claude-opus-4-8"

SYSTEM_PROMPT = (
    "You are a helpful assistant. Use the available tools whenever they can "
    "answer part of the question more reliably than you can from memory."
)


class AgentState(TypedDict):
    # `operator.add` makes this an append-only log: each node returns just its
    # new messages and LangGraph concatenates them onto the existing state.
    # (This is what MessagesState did for us in agent_langgraph.py.)
    messages: Annotated[list[AnyMessage], operator.add]


class Agent:
    def __init__(
        self, model, tool_schemas: list[dict], system: str = "", checkpointer=None
    ):
        self.system = system
        # Bind the same Anthropic-format schemas that tools.py already
        # defines — the registry stays the single source of truth.
        self.model = model.bind_tools(tool_schemas)
        self.known_tools = {schema["name"] for schema in tool_schemas}

        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_llm)
        graph.add_node("action", self.take_action)
        graph.add_conditional_edges(
            "llm", self.exists_action, {True: "action", False: END}
        )
        graph.add_edge("action", "llm")
        graph.set_entry_point("llm")
        # With a checkpointer, the graph saves its state (the message list)
        # after every node, keyed by thread_id — see agent_persistence.py.
        self.graph = graph.compile(checkpointer=checkpointer)

    def exists_action(self, state: AgentState) -> bool:
        """Did the model's last message request any tool calls?"""
        return len(state["messages"][-1].tool_calls) > 0

    def call_llm(self, state: AgentState) -> dict:
        """REASON: one model call over the conversation so far."""
        messages = state["messages"]
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        return {"messages": [self.model.invoke(messages)]}

    def take_action(self, state: AgentState) -> dict:
        """ACT + OBSERVE: run each requested tool, feeding failures back as
        observations instead of raising."""
        results = []
        for call in state["messages"][-1].tool_calls:
            name, args = call["name"], call["args"]
            print(f"[act]     {name}({args})")
            if name not in self.known_tools:
                # Self-correction: a hallucinated tool name becomes an
                # observation the model can react to, not a KeyError.
                result = "bad tool name, retry"
            else:
                try:
                    result = registry.run_tool(name, args)
                except Exception as exc:
                    result = f"Error: {exc}"
            print(f"[observe] {result}")
            results.append(
                ToolMessage(tool_call_id=call["id"], name=name, content=str(result))
            )
        return {"messages": results}

    def run(self, user_input: str, thread_id: str | None = None) -> str:
        # thread_id only has an effect when the Agent was built with a
        # checkpointer: same thread_id -> the saved conversation continues.
        config = {"configurable": {"thread_id": thread_id}} if thread_id else None
        state = self.graph.invoke({"messages": [("user", user_input)]}, config)
        return state["messages"][-1].text


if __name__ == "__main__":
    agent = Agent(
        model=ChatAnthropic(model=MODEL, max_tokens=16000),
        tool_schemas=registry.tool_schemas(),
        system=SYSTEM_PROMPT,
    )
    question = (
        " ".join(sys.argv[1:])
        or "What is 1234 * 5678? Then search the web for 'ReAct agent pattern'."
    )
    print(f"[user] {question}")
    answer = agent.run(question)
    print(f"\n=== final answer " + "=" * 40)
    print(answer)
