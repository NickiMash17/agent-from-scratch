"""The same ReAct agent, rebuilt with LangGraph — for comparison.

Everything agent_raw.py does with a while loop, LangGraph models as a graph:

    START -> agent -> (tool calls pending?) -> tools -> agent -> ... -> END

  - the `agent` node is one model call (the REASON step)
  - the `tools` node executes pending tool calls (the ACT + OBSERVE steps)
  - the conditional edge is the `if stop_reason != "tool_use": break`
  - MessagesState is the `messages` list we appended to by hand

Same loop, different clothes. LangGraph's value shows up when the graph stops
being a single loop: checkpointing/resume, human-in-the-loop interrupts,
branching between multiple agents, streaming intermediate state.

Run:  pip install langgraph langchain-anthropic
      python agent_langgraph.py "What is 1234 * 5678?"
"""

import sys

from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

import tools as impl

MODEL = "claude-opus-4-8"

SYSTEM_PROMPT = (
    "You are a helpful assistant. Use the available tools whenever they can "
    "answer part of the question more reliably than you can from memory."
)


# LangChain generates the JSON schema the model sees from the function
# signature and docstring — the same schema we wrote by hand in tools.py.
@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression, e.g. '(3 + 4) * 2'."""
    return impl.calculator(expression)


@tool
def web_search(query: str) -> str:
    """Search the web for current information."""
    return impl.web_search(query)


@tool
def get_current_time() -> str:
    """Get the current date and time in UTC."""
    return impl.get_current_time()


TOOLS = [calculator, web_search, get_current_time]

llm = ChatAnthropic(model=MODEL, max_tokens=16000)
llm_with_tools = llm.bind_tools(TOOLS)


def agent_node(state: MessagesState) -> dict:
    """One REASON step: call the model on the conversation so far."""
    response = llm_with_tools.invoke(state["messages"])
    # Returning a dict of state updates; MessagesState appends to the list.
    return {"messages": [response]}


graph = StateGraph(MessagesState)
graph.add_node("agent", agent_node)
graph.add_node("tools", ToolNode(TOOLS))  # ACT + OBSERVE, handled for us

graph.add_edge(START, "agent")
# tools_condition inspects the last message: tool calls pending -> "tools",
# otherwise -> END. This is our `if stop_reason != "tool_use": break`.
graph.add_conditional_edges("agent", tools_condition)
graph.add_edge("tools", "agent")  # after tools run, go reason again

app = graph.compile()

# LangGraph also ships the whole thing as a one-liner, which is exactly why
# it's worth knowing what it does inside:
#   from langgraph.prebuilt import create_react_agent
#   app = create_react_agent(llm, TOOLS)


def run_agent(user_input: str) -> str:
    state = app.invoke(
        {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_input},
            ]
        }
    )
    # Print the full trace so the runs are comparable with agent_raw.py.
    for message in state["messages"]:
        message.pretty_print()
    return state["messages"][-1].text


if __name__ == "__main__":
    question = (
        " ".join(sys.argv[1:])
        or "What is 1234 * 5678? Then search the web for 'ReAct agent pattern'."
    )
    answer = run_agent(question)
    print(f"\n=== final answer " + "=" * 40)
    print(answer)
