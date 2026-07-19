"""A different agent architecture: the essay writer.

Every agent so far has been ONE generalist LLM node in a loop, deciding
which tool to call next (ReAct). This graph is the opposite: FIVE
specialized nodes, each with its own prompt and one job, wired into a fixed
revision cycle. The model never chooses the control flow — the graph does.

    planner -> research_plan -> generate --(revisions left?)--> reflect
                                    |                              |
                                   END <---- generate <- research_critique

  planner            write a high-level outline
  research_plan      turn the task into search queries, gather sources
  generate           write (or revise) the essay from plan + sources
  reflect            critique the draft like a strict teacher
  research_critique  research answers to the critique, then generate again

The cycle is bounded by max_revisions in state, not by trusting the model
to know when to stop.

Run:  python agent_essay_writer.py
      (research nodes need TAVILY_API_KEY in .env; without it they skip
       gracefully and the essay is written from model knowledge alone)
"""

import os
from typing import TypedDict

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from pydantic import BaseModel
from tavily import TavilyClient

load_dotenv()

MODEL = "claude-opus-4-8"

model = ChatAnthropic(model=MODEL, max_tokens=16000)

_tavily_key = os.environ.get("TAVILY_API_KEY")
tavily = TavilyClient(api_key=_tavily_key) if _tavily_key else None


class EssayState(TypedDict):
    task: str
    plan: str
    draft: str
    critique: str
    content: list[str]      # accumulated research snippets
    revision_number: int
    max_revisions: int


class Queries(BaseModel):
    """Structured output for the research nodes (native Pydantic v2)."""

    queries: list[str]


# One focused prompt per node — this is the heart of the pattern.
PLAN_PROMPT = (
    "You are an expert writer tasked with writing a high level outline of an "
    "essay. Write such an outline for the user provided topic. Give an "
    "outline of the essay along with any relevant notes or instructions for "
    "the sections."
)

WRITER_PROMPT = (
    "You are an essay assistant tasked with writing excellent 5-paragraph "
    "essays. Generate the best essay possible for the user's request and the "
    "initial outline. If the user provides critique, respond with a revised "
    "version of your previous attempts. Utilize all the information below as "
    "needed:\n\n------\n\n{content}"
)

REFLECTION_PROMPT = (
    "You are a teacher grading an essay submission. Generate critique and "
    "recommendations for the user's submission. Provide detailed "
    "recommendations, including requests for length, depth, style, etc."
)

RESEARCH_PLAN_PROMPT = (
    "You are a researcher charged with providing information that can be "
    "used when writing the following essay. Generate a list of search "
    "queries that will gather any relevant information. Only generate 3 "
    "queries max."
)

RESEARCH_CRITIQUE_PROMPT = (
    "You are a researcher charged with providing information that can be "
    "used when making any requested revisions (as outlined below). Generate "
    "a list of search queries that will gather any relevant information. "
    "Only generate 3 queries max."
)


def _do_research(prompt: str, focus: str, state: EssayState) -> dict:
    """Shared body of both research nodes: queries -> Tavily -> snippets."""
    content = list(state.get("content") or [])
    if tavily is None:
        print("[skipped] TAVILY_API_KEY not set — continuing without research.")
        print("          (free key at https://tavily.com; add it to .env)")
        return {"content": content}

    queries = model.with_structured_output(Queries).invoke(
        [SystemMessage(content=prompt), HumanMessage(content=focus)]
    )
    for query in queries.queries:
        response = tavily.search(query=query, max_results=2)
        print(f"  query: {query!r} -> {len(response['results'])} results")
        for hit in response["results"]:
            content.append(hit["content"])
    return {"content": content}


def plan_node(state: EssayState) -> dict:
    response = model.invoke(
        [SystemMessage(content=PLAN_PROMPT), HumanMessage(content=state["task"])]
    )
    return {"plan": response.text}


def research_plan_node(state: EssayState) -> dict:
    return _do_research(RESEARCH_PLAN_PROMPT, state["task"], state)


def generation_node(state: EssayState) -> dict:
    content = "\n\n".join(state.get("content") or [])
    user = f"{state['task']}\n\nHere is my plan:\n\n{state['plan']}"
    # On revision passes, show the model its previous draft and the critique.
    # (The original lesson relies on new research alone to steer revisions;
    # passing the critique explicitly makes the revision actually address it.)
    if state.get("critique"):
        user += (
            f"\n\nHere is my previous draft:\n\n{state['draft']}"
            f"\n\nHere is critique of that draft — revise to address it:"
            f"\n\n{state['critique']}"
        )
    response = model.invoke(
        [SystemMessage(content=WRITER_PROMPT.format(content=content)),
         HumanMessage(content=user)]
    )
    return {
        "draft": response.text,
        "revision_number": state.get("revision_number", 1) + 1,
    }


def reflection_node(state: EssayState) -> dict:
    response = model.invoke(
        [SystemMessage(content=REFLECTION_PROMPT),
         HumanMessage(content=state["draft"])]
    )
    return {"critique": response.text}


def research_critique_node(state: EssayState) -> dict:
    return _do_research(RESEARCH_CRITIQUE_PROMPT, state["critique"], state)


def should_continue(state: EssayState) -> str:
    if state["revision_number"] > state["max_revisions"]:
        return END
    return "reflect"


builder = StateGraph(EssayState)
builder.add_node("planner", plan_node)
builder.add_node("research_plan", research_plan_node)
builder.add_node("generate", generation_node)
builder.add_node("reflect", reflection_node)
builder.add_node("research_critique", research_critique_node)

builder.set_entry_point("planner")
builder.add_edge("planner", "research_plan")
builder.add_edge("research_plan", "generate")
builder.add_conditional_edges("generate", should_continue, {END: END, "reflect": "reflect"})
builder.add_edge("reflect", "research_critique")
builder.add_edge("research_critique", "generate")

graph = builder.compile()


if __name__ == "__main__":
    task = (
        "Write an essay on whether AI agents should be built from scratch "
        "or with frameworks."
    )
    print(f"[task] {task}")

    initial: EssayState = {
        "task": task,
        "plan": "",
        "draft": "",
        "critique": "",
        "content": [],
        "revision_number": 1,
        "max_revisions": 2,  # first draft + one revision cycle
    }

    for event in graph.stream(initial):
        for node, update in event.items():
            print(f"\n{'=' * 70}\nNODE: {node}\n{'=' * 70}")
            if "plan" in update:
                print(update["plan"])
            if "content" in update:
                print(f"(research snippets gathered so far: {len(update['content'])})")
            if "draft" in update:
                words = len(update["draft"].split())
                print(f"--- draft (revision {update['revision_number'] - 1}, "
                      f"{words} words) ---")
                print(update["draft"])
            if "critique" in update:
                print(update["critique"])

    print(f"\n{'=' * 70}\ndone — final draft above.\n{'=' * 70}")
