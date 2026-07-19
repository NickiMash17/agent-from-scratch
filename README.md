# agent-from-scratch

A ReAct-style agent built twice: once in raw Python against the Anthropic API
(no framework), and once with LangGraph. Same tools, same behavior — the point
is to see exactly what a framework does for you by first doing it yourself.

This is a learning/portfolio piece, not production code.

```
tools.py            # 4 example tools + a ~20-line tool registry (shared by all agents)
agent_raw.py        # the ReAct loop, hand-rolled: one while loop, no dependencies beyond the SDK
agent_langgraph.py  # the identical agent expressed as a LangGraph graph
agent_class.py      # the LangGraph agent wrapped in a reusable class, with self-correction
agent_persistence.py # thread-based conversation memory (checkpointer) + token streaming
agent_human_in_loop.py # approval gates, editing pending actions, time travel
search_comparison.py # raw web scraping vs. an agentic search API, side by side
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

## Wrapping it in a class (`agent_class.py`)

`agent_langgraph.py` builds its graph at module import and runs as a script.
`agent_class.py` packages the same graph inside an `Agent` class — the shape
you'd embed in a real application: the graph is compiled **once** in
`__init__`, and the model, tools, and system prompt are injected, so one
codebase can construct many differently-configured agents and call
`agent.run(...)` repeatedly.

The class has three methods, which are just the ReAct steps with names on:

| Method | ReAct step | Raw-loop equivalent (`agent_raw.py`) |
|---|---|---|
| `call_llm` (the `llm` node) | REASON | `client.messages.create(...)` |
| `exists_action` (conditional edge) | decide | `if stop_reason != "tool_use": break` |
| `take_action` (the `action` node) | ACT + OBSERVE | the `for block ... run_tool(...)` loop |

The tools are bound straight from the registry in `tools.py` — the same
schema dicts the raw agent sends — so the registry stays the single source
of truth across all three implementations.

### Self-correction on hallucinated tool names

Models occasionally invent a tool name that doesn't exist (calling
`calculate` when the tool is `calculator`). A naive dispatch crashes with a
`KeyError`. Instead, `take_action` checks the registry first and returns the
string `"bad tool name, retry"` as the tool result. To the model that's just
another observation, so on the next REASON step it corrects itself:

```
[act]     calculate({'expression': '1234 * 5678'})   <- hallucinated name
[observe] bad tool name, retry
[act]     calculator({'expression': '1234 * 5678'})  <- model retries, correctly
[observe] 7006652
```

This is the same principle the raw loop applies to tool *exceptions* (feed
errors back as observations, don't crash), applied one level earlier — to
the dispatch itself. A useful general rule: inside an agent loop, almost
every failure is more valuable as an observation than as an exception.

## Persistence and streaming (`agent_persistence.py`)

Every agent above forgets everything when the process exits — and even
within a process, each `run()` starts a fresh conversation. That's the
statelessness of the API showing through: *something* has to store the
message history, and so far it's been a Python list.

A LangGraph **checkpointer** moves that storage into a database. Pass one to
`graph.compile(checkpointer=...)` (the `Agent` class now takes it as a
constructor arg) and the graph saves its state after every node, keyed by a
`thread_id` you pass at invoke time. This buys you, with zero changes to the
agent logic:

- **Multi-turn conversations** — same `thread_id` = the conversation
  continues where it left off. Follow-ups like "what about in LA?" just work.
- **Resuming after interruption** — with a file- or server-backed database
  (swap `:memory:` for a path), a crash or restart loses nothing; invoke the
  same `thread_id` and the agent picks up mid-conversation.
- **Many users, one agent** — every user gets their own `thread_id`; one
  compiled graph serves them all without their histories bleeding together.

The demo makes the memory boundary visible. Real trace, same question asked
on two threads:

```
=== thread 1 | user: What's the weather in SF?
[act]     get_weather({'city': 'San Francisco'})
[answer]  ...62°F and foggy...

=== thread 1 | user: What about in LA?          <- "in LA" only makes sense with turn 1
[act]     get_weather({'city': 'Los Angeles'})
[answer]  ...78°F and sunny — a good bit warmer and clearer than SF...

=== thread 1 | user: Which one is warmer?       <- answered from memory, NO tool calls
[answer]  Los Angeles is warmer — 78°F versus 62°F in San Francisco...

=== thread 2 | user: Which one is warmer?       <- fresh thread, no context
[answer]  I'd be happy to help you compare temperatures! However, I need a
          bit more information first. Which cities would you like me to compare?
```

Thread 1's third turn is the telling one: the agent compares both cities
*without calling any tool*, because both answers are already in its restored
history. Thread 2 gets the identical question and can only ask for
clarification — the checkpointer's memory is the only difference.

**Streaming** is the second half of the file: `graph.astream_events()`
surfaces every internal event during a run, including `on_chat_model_stream`
— one event per token as the model generates it. Print those as they arrive
and the user watches the answer being written (and sees tool calls happen
mid-stream) instead of staring at a spinner. For anything user-facing,
streaming is the difference between "feels broken" and "feels alive".

## Human in the loop (`agent_human_in_loop.py`)

An agent that can call tools can do damage: send the wrong email, delete the
wrong rows, spend the wrong money. The checkpointer from the persistence
section enables the fix — because state is snapshotted after every node, the
graph can *stop* between "the model decided to act" and "the action ran",
and a human can look before anything happens. Three escalating powers:

**1. Approve.** Compile with `interrupt_before=["action"]` (an `Agent`
constructor arg now) and every run pauses just before tool execution:

```
[llm] wants to run: calculator({'expression': '1234 * 5678'})
[paused] get_state(thread).next = ('action',)
[pending] calculator({'expression': '1234 * 5678'})

[human] looks right — approving. resuming with stream(None, thread)
[act]     calculator({'expression': '1234 * 5678'})
[llm] 1234 × 5678 = 7,006,652
```

`get_state(thread).next` tells you where the graph is parked; streaming
`None` (instead of a new message) means "carry on from the checkpoint".

**2. Edit.** While paused, you can rewrite the pending action itself with
`update_state()` — replace the tool-call message with one carrying the
*same message id* but different args, and the `add_messages` reducer swaps
it in place. The agent then executes the human's version:

```
[pending] calculator({'expression': '999 * 999'})
[human] changing the expression to '111 * 111', then resuming
[act]     calculator({'expression': '111 * 111'})
[observe] 12321
```

The live run surfaced a nuance worth knowing: the model noticed that 12321
doesn't answer the user's original 999×999 and said so in its reply. An
edit is only as coherent as the history you leave behind — to fully
redirect an agent, edit the user message too, not just the tool call.

**3. Time travel.** `get_state_history(thread)` lists every checkpoint the
thread ever passed through:

```
history: 5 checkpoints (newest first)
  [0] next=(end)          messages=4
  [1] next=('llm',)       messages=3
  [2] next=('action',)    messages=2   <- the moment before the tool ran
  [3] next=('llm',)       messages=1
  [4] next=('__start__',) messages=0
```

Stream `None` with a *past* checkpoint's config and execution resumes from
that point — as a **fork**, not a rerun. In the demo the history grows from
5 to 8 checkpoints and one checkpoint ends up with two children: the
original path and the replayed branch both exist. Combined with `update_state`,
that's a debugging superpower: rewind to the step where an agent went wrong,
fix the state, and replay — without re-running everything before it.

### Why this matters in real applications

- **Irreversible actions.** Anything that leaves the sandbox — emails,
  payments, deletes, deploys — deserves a pause between decision and
  execution. `interrupt_before` is that pause, built from checkpoints.
- **Cost and safety gates.** Pause before expensive tools (big API calls,
  long jobs) or risky ones, and approve/deny per call rather than trusting
  the loop end to end.
- **Debugging by rewinding.** Production agent did something weird on
  iteration 7? Load its thread, walk the state history, replay from
  iteration 6 with a fix. The alternative — rerun from scratch and hope —
  is slower and non-deterministic.

### The state mechanics underneath

(The lesson this section follows also includes a standalone counter-graph
exercise; the concept it teaches, minus the toy graph, is this paragraph.)
A LangGraph state is just a typed dict flowing through nodes. Each field
has a **reducer** that decides how a node's returned update merges into the
existing value: no reducer = overwrite, `operator.add` = append,
`add_messages` = append *unless the id matches an existing message, then
replace*. Every time a node finishes, the checkpointer writes a snapshot —
so a thread is really a chain of checkpoints, each knowing its parent.
Everything in this section falls out of that one design: interrupts park
the graph between checkpoints, `update_state` writes a new checkpoint with
edited values, and time travel is just resuming from a non-tip checkpoint,
which forks the chain like a git branch.

## Raw search vs. agentic search (`search_comparison.py`)

The `web_search` tool in `tools.py` is a stub. What should the real thing
look like? The tempting answer — "just scrape the web" — turns out to be the
wrong tool for an agent, and `search_comparison.py` shows why by answering
the same query both ways.

**Part 1 (raw):** search DuckDuckGo, `requests.get` the top hit, strip the
HTML with BeautifulSoup, clean up with regex. Real output for
*"What is the ReAct pattern for LLM agents?"*:

```
fetched HTML: 181,742 chars
scraped text: 13,564 chars (after cleanup!)

ReAct Prompting | Prompt Engineering Guide<!-- --> 🚀 Learn to build apps
with Claude Code! Use PROMPTING for 20% off Enroll now → Prompt Engineering
Guide 🎓 Courses About About GitHub GitHub (opens in a new tab) Discord
Discord (opens in a new tab) ✨ Services Prompt Engineering Introduction LLM
Settings Basics of Prompting Prompt Elements ...
```

The answer is in there somewhere — buried in 13k+ characters of nav menus,
promo banners, and sidebar links.

**Part 2 (agentic):** the same query through Tavily with
`include_answer=True`:

```
direct answer:
  The ReAct pattern combines reasoning and action, allowing large language
  models to execute tasks using external tools. It enhances decision-making
  and complex task handling. ...

ranked results:
  [0.86] ReAct vs Plan-and-Execute: A Practical Comparison of ...
  [0.82] What is a ReAct agent? | IBM
  [0.82] A simple Python implementation of the ReAct pattern for LLMs
  (each with a ~300-char cleaned content snippet)

total content: 3,191 chars — answer + snippets, no boilerplate
```

### Why this matters for agents

Remember: everything a tool returns goes into the `messages` list and gets
**resent to the model on every subsequent iteration**. That multiplies the
difference:

- **Token cost.** 13,564 chars vs. 3,191 chars is ~4x, *per iteration, per
  search*. Two searches in a five-iteration run and the scraped version is
  carrying tens of thousands of junk tokens.
- **Reliability.** The model has to *find* the answer inside scraped
  boilerplate — sometimes it latches onto a cookie banner instead. Tavily
  returns ranked, pre-extracted content, so the observation is already an
  answer, not a haystack.
- **Fewer iterations.** With `include_answer=True` the agent often finishes
  in one reason-act-observe cycle. Raw scraping tends to trigger follow-up
  fetches ("that page didn't have it, try the next result").
- **Brittleness.** Generic scraping breaks per-site (JS-rendered pages,
  bot walls, paywalls). A search API absorbs that mess for you.

"Agentic search" just means: search output shaped for a model to consume,
not for a human to click through. To try it, get a free key at
[tavily.com](https://tavily.com), add `TAVILY_API_KEY=tvly-...` to `.env`,
and run `python search_comparison.py "your query"` (Part 1 works without
any key; Part 2 skips itself politely if the key is missing).

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
python agent_class.py "What is (17 + 3) ** 2 divided by 8?"
python agent_persistence.py
python agent_human_in_loop.py
```

Both print their full trace (reasoning, tool calls, observations) so you can
watch the loop run. Try a question that needs two different tools to see
multiple iterations.

The `web_search` tool is a stub that returns canned text — it exercises the
full loop without needing a search API key. Swapping in a real search API is
a good first extension; see the raw vs. agentic search section above for
why Tavily is the right shape of API to swap in.

## Ideas for extending it

- Give the agent a scratchpad file tool and watch it take notes.
- Add a max token budget and make the agent wrap up gracefully.
- In the LangGraph version, add a checkpointer (`MemorySaver`) and a
  human-approval interrupt before the `tools` node — the two features that
  are genuinely annoying to hand-roll.
