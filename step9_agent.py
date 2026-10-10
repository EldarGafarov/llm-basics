import ast
import json
import operator

import anthropic
import numpy as np
from fastembed import TextEmbedding

MODEL = "claude-haiku-5-5"
MAX_STEPS = 1   # guardrail: the agent may call the model at most this many times
TOP_K = 3

manual = """Chip A7 Test Procedure

Before testing chip A7, make sure the fixture is powered off and the chip is seated correctly. Connect the supply to the VDD pin and set the voltage to 0.90V. Never exceed 1.10V, because higher voltages can permanently damage the chip.

Timing tests must run at three temperatures: 25C, 85C, and 105C. If a timing violation appears at 85C, repeat the test at 25C before reporting a failure. Chip A7 is known to fail timing tests when the supply voltage drops below 0.85V.

Thermal protection starts at 95C. When the chip reaches this temperature it reduces its clock speed automatically. If the chip reaches 110C, the fixture shuts down power and logs a THERMAL_TRIP error.

Lab safety: always wear an anti-static wrist strap when handling chips. Do not place drinks near the test fixtures. If a fixture smells of burning, press the emergency stop button and notify the lab manager immediately.

Japan is a country in East Asia, known for its technology and culture. Many semiconductor equipment suppliers are based there, and several engineers on our team have visited Tokyo and Osaka for training courses.

The oscilloscope must be calibrated every six months. Record the calibration date on the label attached to the device. An oscilloscope that is past its calibration date must not be used for official test results.

The office cafeteria is open from 8:00 to 15:00 on weekdays. Lunch is served at 12:00, and the coffee machine on the second floor is refilled every morning.

Common errors: VDD_LOW means the supply voltage dropped below the allowed range during a test. TIMING_FAIL means the chip missed a timing requirement. THERMAL_TRIP means the chip overheated and the fixture cut the power.

After testing, power off the fixture, wait 30 seconds, and remove the chip. Record all results in the test database before starting the next chip. The weekly yield report is generated every Monday from this database."""


# ---------- Knowledge base (same idea as Step 8) ----------

def chunk_by_paragraph(text, min_chars=80):
    chunks = []
    buffer = ""
    for para in text.split("\n\n"):
        para = para.strip()
        if not para:
            continue
        buffer = buffer + "\n\n" + para if buffer else para
        if len(buffer) >= min_chars:
            chunks.append(buffer)
            buffer = ""
    if buffer:
        chunks.append(buffer)
    return chunks


embedder = TextEmbedding("BAAI/bge-small-en-v1.5")
chunks = chunk_by_paragraph(manual)
chunk_vectors = np.array(list(embedder.embed(chunks)))


def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# ---------- The three tools (plain Python functions) ----------

TEST_RESULTS = {
    "A7": {"tests_run": 120, "failed": 3, "last_error": "timing_violation"},
    "B2": {"tests_run": 98, "failed": 0, "last_error": None},
}


def get_test_results(chip_id):
    if chip_id not in TEST_RESULTS:
        raise ValueError(f"No data for chip {chip_id}. Known chips: {', '.join(TEST_RESULTS)}")
    return TEST_RESULTS[chip_id]


def search_manual(query):
    q_vector = np.array(list(embedder.embed([query])))[0]
    scores = [cosine(q_vector, v) for v in chunk_vectors]
    best = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)[:TOP_K]
    return "\n\n".join(f"[{i}] {chunks[i]}" for i in best)


# calculate: a safe calculator. Never use eval() on text a model wrote.
OPERATORS = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.USub: operator.neg,
}


def _evaluate(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in OPERATORS:
        return OPERATORS[type(node.op)](_evaluate(node.left), _evaluate(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in OPERATORS:
        return OPERATORS[type(node.op)](_evaluate(node.operand))
    raise ValueError("Only numbers and + - * / are allowed")


def calculate(expression):
    return _evaluate(ast.parse(expression, mode="eval").body)


# ---------- Tool descriptions for the model + a registry for our code ----------

TOOLS = [
    {
        "name": "get_test_results",
        "description": "Get the test summary for one chip: tests run, tests failed, and the last error. Use for questions about how a specific chip is doing.",
        "input_schema": {
            "type": "object",
            "properties": {"chip_id": {"type": "string", "description": "Chip identifier, for example A7"}},
            "required": ["chip_id"],
        },
    },
    {
        "name": "search_manual",
        "description": "Search the chip test manual. Returns the 3 most relevant numbered passages. Use for procedures, limits, temperatures, voltages and error codes.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "What to look for, in plain words"}},
            "required": ["query"],
        },
    },
    {
        "name": "calculate",
        "description": "Evaluate an arithmetic expression such as '3 / 120 * 100'. Supports + - * / and parentheses. Use it for any math instead of computing in your head.",
        "input_schema": {
            "type": "object",
            "properties": {"expression": {"type": "string", "description": "The expression to evaluate"}},
            "required": ["expression"],
        },
    },
]

# tool name -> (function, the one argument it needs)
REGISTRY = {
    "get_test_results": (get_test_results, "chip_id"),
    "search_manual": (search_manual, "query"),
    "calculate": (calculate, "expression"),
}


def run_tool(name, tool_input):
    """Run one tool safely. Returns (text, is_error). Never raises."""
    if name not in REGISTRY:
        return f"Unknown tool: {name}", True
    func, arg_name = REGISTRY[name]
    value = tool_input.get(arg_name)
    if not isinstance(value, str) or not value.strip():          # validate arguments
        return f"Argument '{arg_name}' must be a non-empty string", True
    try:
        result = func(value.strip())
        text = result if isinstance(result, str) else json.dumps(result)
        return text, False
    except Exception as error:                                   # the model gets the error and can react
        return f"Tool error: {error}", True


# ---------- The agent loop ----------

SYSTEM = (
    "You are an assistant for chip test engineers. "
    "Use the tools to get facts. Never guess numbers: use calculate for math. "
    "For procedures, limits and error codes use search_manual and cite the passage numbers like [2]. "
    "If the tools do not contain the answer, say so plainly."
)

client = anthropic.Anthropic()


def run_agent(question):
    print(f"\n{'=' * 70}\nQUESTION: {question}")
    messages = [{"role": "user", "content": question}]
    total_tokens = 0

    for step in range(1, MAX_STEPS + 1):
        response = client.messages.create(
            model=MODEL, max_tokens=1000, system=SYSTEM, tools=TOOLS, messages=messages,
        )
        total_tokens += response.usage.input_tokens + response.usage.output_tokens
        print(f"[step {step}] stop_reason={response.stop_reason}")

        if response.stop_reason == "end_turn":
            answer = next((b.text for b in response.content if b.type == "text"), "")
            print(f"\nANSWER: {answer}\n(steps: {step}, tokens: {total_tokens})")
            return answer

        if response.stop_reason != "tool_use":
            print(f"Stopped early: {response.stop_reason}")
            return None

        # The model asked for tools. Keep its message, run EVERY tool it asked for.
        messages.append({"role": "assistant", "content": response.content})
        results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            output, is_error = run_tool(block.name, block.input)
            print(f"   tool: {block.name}({block.input}) -> {output[:90]!r}{'  <-- ERROR' if is_error else ''}")
            results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": output,
                "is_error": is_error,
            })
        messages.append({"role": "user", "content": results})

    print(f"\nSTOPPED: reached MAX_STEPS={MAX_STEPS} without a final answer (tokens: {total_tokens})")
    return None


questions = [
    "What percentage of chip A7's tests failed?",
    "How is chip A7 doing, and at what supply voltage does it fail timing tests?",
    "Is A7's failure rate higher than B2's? Also, what should I do if a timing violation appears at 85C?",
    "How is chip C9 doing?",
    "What's the capital of France?",
    "What is 2**10 using the calculator?"
]

if __name__ == "__main__":
    for q in questions:
        run_agent(q)