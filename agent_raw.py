"""A ReAct-style agent loop built directly on the Anthropic API — no framework.

The entire "agent" is ~40 lines: a while loop that
  1. sends the conversation to the model,
  2. checks whether the model wants to call a tool,
  3. runs the tool and appends the result to the conversation,
  4. repeats until the model answers in plain text.

Run:  export ANTHROPIC_API_KEY=...  (or `ant auth login`)
      python agent_raw.py "What is 1234 * 5678, and what time is it right now?"
"""

import json
import sys

import anthropic

from tools import run_tool, tool_schemas

MODEL = "claude-opus-4-8"

SYSTEM_PROMPT = (
    "You are a helpful assistant. Use the available tools whenever they can "
    "answer part of the question more reliably than you can from memory."
)


def run_agent(user_input: str, max_iterations: int = 10) -> str:
    client = anthropic.Anthropic()

    # The conversation is just a list of messages. The API is stateless, so
    # this list IS the agent's entire memory — we resend it on every step.
    messages = [{"role": "user", "content": user_input}]

    for iteration in range(1, max_iterations + 1):
        print(f"\n--- iteration {iteration} " + "-" * 40)

        # REASON: the model reads the whole conversation (including previous
        # tool results) and decides what to do next.
        response = client.messages.create(
            model=MODEL,
            max_tokens=16000,
            system=SYSTEM_PROMPT,
            tools=tool_schemas(),
            messages=messages,
        )

        # Show the model's visible output for this step. Any text before a
        # tool call is the model "thinking out loud" about its plan.
        for block in response.content:
            if block.type == "text":
                print(f"[model] {block.text}")

        # The assistant turn (text + tool_use blocks) must go back into the
        # history verbatim, so the model can see its own past actions.
        messages.append({"role": "assistant", "content": response.content})

        # If the model didn't ask for a tool, it has produced its final
        # answer and the loop is done.
        if response.stop_reason != "tool_use":
            return next(
                (b.text for b in response.content if b.type == "text"), ""
            )

        # ACT + OBSERVE: run every tool the model requested and collect the
        # results. All results go back in a single user message, each tagged
        # with the tool_use_id it answers.
        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            print(f"[act]     {block.name}({json.dumps(block.input)})")
            try:
                result = run_tool(block.name, block.input)
                is_error = False
            except Exception as exc:
                # Feed errors back to the model instead of crashing — it can
                # often recover (retry, fix its input, try another tool).
                result = f"Error: {exc}"
                is_error = True
            print(f"[observe] {result}")
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                    "is_error": is_error,
                }
            )
        messages.append({"role": "user", "content": tool_results})

    return "Stopped: hit the maximum number of iterations."


if __name__ == "__main__":
    question = (
        " ".join(sys.argv[1:])
        or "What is 1234 * 5678? Then search the web for 'ReAct agent pattern'."
    )
    print(f"[user] {question}")
    answer = run_agent(question)
    print(f"\n=== final answer " + "=" * 40)
    print(answer)
