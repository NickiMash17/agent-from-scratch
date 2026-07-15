# agent-from-scratch

A ReAct-style agent built twice: once in raw Python against the Anthropic API
(no framework), and once with LangGraph. Same tools, same behavior — the point
is to see exactly what a framework does for you by first doing it yourself.

This is a learning/portfolio piece, not production code.

```
tools.py            # 3 example tools + a ~20-line tool registry (shared by both agents)
agent_raw.py        # the ReAct loop, hand-rolled: one while loop, no dependencies beyond the SDK
agent_langgraph.py  # the identical agent expressed as a LangGraph graph
```

## What a ReAct loop actually is

[ReAct](https://arxiv.org/abs/2210.03629) ("Reasoning + Acting") is the
pattern behind almost every LLM agent: instead of answering in one shot, the
model alternates between **reasoning** about what to do, **acting** by calling
a tool, and **observing** the result — until it has enough to answer.

There is no magic. The "agent" is a plain loop around a chat API:

```
messages = [user question]
loop:
    1. REASON   send messages to the model
    2.          did the model request a tool call?
                  no  -> return its text: done
    3. ACT      yes -> run the requested tool(s) locally
    4. OBSERVE  append the tool results to messages
    5.          go to 1
```

A concrete trace for *"What is 1234 × 5678, and what time is it?"*:

| Step | Who | What happens |
|------|-----|--------------|
| 1 | model | "I need to compute this" → emits `tool_use: calculator({"expression": "1234 * 5678"})` |
| 2 | your code | runs `calculator(...)` → `"7006652"` → appends it as a `tool_result` message |
| 3 | model | sees the result, still needs the time → emits `tool_use: get_current_time({})` |
| 4 | your code | runs the tool → appends the result |
| 5 | model | has everything → responds with plain text. Loop exits. |

Three details make the loop work, and they're the details a framework hides:

1. **The API is stateless.** The model remembers nothing between calls. The
   `messages` list *is* the agent's memory, and you resend all of it every
   iteration — including the model's own previous tool calls, verbatim.
2. **The model never executes anything.** It only emits a structured request
   ("call `calculator` with this input"). Your code runs the tool, which is
   why *you* decide what tools exist, how they're sandboxed, and what the
   model gets to see.
3. **Errors go back into the conversation.** If a tool throws, you return the
   error message as the tool result (flagged with `is_error`) instead of
   crashing. The model reads it and usually recovers — retries, fixes its
   input, or tries a different tool.

The original ReAct paper predates native tool-calling, so it had the model
*write out* `Thought: ... / Action: ... / Observation: ...` as text and parsed
it with regex. Modern APIs build the Thought/Action structure into the
protocol (`tool_use` blocks, `stop_reason`), so `agent_raw.py` uses that —
same pattern, minus the brittle string parsing.

## The same thing in LangGraph

`agent_langgraph.py` re-expresses the loop as a graph:

```
START ──> agent ──(tool calls pending?)──> tools ──> agent ──> ... ──> END
```

Every piece maps one-to-one onto the raw version:

| Raw loop (`agent_raw.py`) | LangGraph (`agent_langgraph.py`) |
|---|---|
| the `messages` list you append to | `MessagesState` |
| `client.messages.create(...)` | the `agent` node |
| the `for block ... run_tool(...)` block | `ToolNode(TOOLS)` |
| `if stop_reason != "tool_use": break` | `tools_condition` conditional edge |
| the `while` loop itself | the `tools -> agent` edge |
| hand-written JSON schemas in `tools.py` | `@tool` decorator on typed functions |

LangGraph even ships the entire file as one call —
`create_react_agent(llm, tools)` — which is precisely why it's worth having
built it by hand once: that one-liner is this repo's `agent_raw.py`, boxed up.

## Why learn the raw loop before reaching for a framework?

- **Debugging.** When an agent loops forever, ignores a tool, or "forgets"
  something, the cause lives at this level: a malformed tool result, a missing
  message in the history, a bad tool description. If the loop is a black box,
  you're debugging blind.
- **Knowing what you're paying for.** Every iteration resends the whole
  conversation. Token cost and latency grow with history length — obvious
  when you wrote `messages.append(...)`, invisible behind `app.invoke(...)`.
- **Judging when you actually need the framework.** For a single
  reason-act-observe loop, the raw version is *shorter* than the LangGraph
  version. LangGraph earns its complexity when you need what a hand-rolled
  loop makes painful: checkpointing and resuming runs, human-in-the-loop
  interrupts, multi-agent graphs with branching, streaming intermediate state.
- **Frameworks churn; the pattern doesn't.** Agent framework APIs change
  every few months. The loop in `agent_raw.py` is the stable thing underneath
  all of them.

## Running it

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...   # Windows: $env:ANTHROPIC_API_KEY = "sk-ant-..."

python agent_raw.py "What is (17 + 3) ** 2 divided by 8?"
python agent_langgraph.py "What is (17 + 3) ** 2 divided by 8?"
```

Both print their full trace (reasoning, tool calls, observations) so you can
watch the loop run. Try a question that needs two different tools to see
multiple iterations.

The `web_search` tool is a stub that returns canned text — it exercises the
full loop without needing a search API key. Swapping in a real search API is
a good first extension.

## Ideas for extending it

- Give the agent a scratchpad file tool and watch it take notes.
- Add a max token budget and make the agent wrap up gracefully.
- In the LangGraph version, add a checkpointer (`MemorySaver`) and a
  human-approval interrupt before the `tools` node — the two features that
  are genuinely annoying to hand-roll.
