"""Example tools + a minimal tool registry.

The registry is the whole "framework" here: a dict mapping a tool name to
its JSON schema (what the model sees) and its Python function (what we run).
Both agent implementations (agent_raw.py and agent_langgraph.py) reuse the
same underlying functions, so the only thing that differs between them is
the loop around the tools.
"""

import ast
import operator
from datetime import datetime, timezone

# name -> {"schema": <tool definition sent to the API>, "fn": <callable>}
TOOL_REGISTRY: dict[str, dict] = {}


def register_tool(name: str, description: str, input_schema: dict):
    """Decorator that adds a function to the registry with its schema."""

    def decorator(fn):
        TOOL_REGISTRY[name] = {
            "schema": {
                "name": name,
                "description": description,
                "input_schema": input_schema,
            },
            "fn": fn,
        }
        return fn

    return decorator


def tool_schemas() -> list[dict]:
    """The `tools` parameter for the Anthropic API."""
    return [entry["schema"] for entry in TOOL_REGISTRY.values()]


def run_tool(name: str, tool_input: dict) -> str:
    """Dispatch a tool call by name. Always returns a string."""
    if name not in TOOL_REGISTRY:
        raise KeyError(f"Unknown tool: {name}")
    return str(TOOL_REGISTRY[name]["fn"](**tool_input))


# ---------------------------------------------------------------------------
# Tool 1: calculator
# ---------------------------------------------------------------------------

# Safe arithmetic evaluation: walk the AST and only allow number literals
# and these operators. Never eval() model-generated strings directly.
_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval_node(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError(f"Unsupported expression element: {ast.dump(node)}")


@register_tool(
    name="calculator",
    description=(
        "Evaluate a basic arithmetic expression. Supports +, -, *, /, //, %, ** "
        "and parentheses. Call this for any math instead of computing it yourself."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "The expression to evaluate, e.g. '(3 + 4) * 2'",
            }
        },
        "required": ["expression"],
    },
)
def calculator(expression: str) -> str:
    tree = ast.parse(expression, mode="eval")
    return str(_eval_node(tree.body))


# ---------------------------------------------------------------------------
# Tool 2: web search (stub)
# ---------------------------------------------------------------------------

@register_tool(
    name="web_search",
    description=(
        "Search the web for current information. Call this when the answer "
        "depends on facts you don't know or that may have changed recently."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query"}
        },
        "required": ["query"],
    },
)
def web_search(query: str) -> str:
    # Stub: in a real agent you'd call a search API (Tavily, Brave, SerpAPI...).
    # Returning canned text keeps the example free of API keys while still
    # exercising the full reason -> act -> observe cycle.
    return (
        f"[stub] Top result for '{query}': "
        "This is a placeholder search result. Wire up a real search API here."
    )


# ---------------------------------------------------------------------------
# Tool 3: weather (stub)
# ---------------------------------------------------------------------------

@register_tool(
    name="get_weather",
    description="Get the current weather for a city.",
    input_schema={
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "City name, e.g. 'San Francisco'",
            }
        },
        "required": ["city"],
    },
)
def get_weather(city: str) -> str:
    # Stub with canned values so demos (agent_persistence.py) are
    # deterministic. Swap in a real weather API to make it live.
    canned = {
        "san francisco": "62F, foggy",
        "sf": "62F, foggy",
        "los angeles": "78F, sunny",
        "la": "78F, sunny",
    }
    report = canned.get(city.strip().lower(), "70F, partly cloudy")
    return f"Weather in {city}: {report}"


# ---------------------------------------------------------------------------
# Tool 4: current time
# ---------------------------------------------------------------------------

@register_tool(
    name="get_current_time",
    description="Get the current date and time in UTC.",
    input_schema={"type": "object", "properties": {}},
)
def get_current_time() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
