# LLM API Basics: a hands-on guide (Python + Claude API)

A step-by-step practice project for learning the fundamentals of working with an LLM API. Each step is a small script that teaches one idea, followed by a list of experiments you can run by editing a line or two. Written while preparing for a junior/mid GenAI + cloud software engineer interview.

**Stack:** Python 3.12, `anthropic` SDK, `pydantic`, model `claude-haiku-5-5` (cheap and fast, good enough for practice).

**How to use this guide:** run each script as-is first, read the output, then work through its "Things to try" list. Changing one thing at a time and watching what happens is how the ideas stick.

## Contents

1. [Setup](#setup)
2. [Step 1: Your first API call](#step-1-your-first-api-call-step1_basicpy)
3. [Step 2: System prompts](#step-2-system-prompts-step2_systempy)
4. [Step 3: Structured output with a prompt only](#step-3-structured-output-with-a-prompt-only-step3_structuredpy)
5. [Step 4: Enforced schema](#step-4-enforced-schema-step4_schemapy)
6. [Step 5: Tool calling](#step-5-tool-calling-step5_toolspy)
7. [Making model output predictable](#making-model-output-predictable-interview-summary)
8. [Troubleshooting](#troubleshooting)
9. [Next steps](#next-steps)

## Setup

1. Create an API key in the Claude Console (Settings, then API keys) and add a few dollars of credit. All scripts here cost a tiny fraction of a cent per run.
2. Create and activate a virtual environment, then install the dependencies:

   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

   On Mac/Linux, activate with `source .venv/bin/activate`.

3. Set your key as an environment variable. Note the **quotes**; without them PowerShell treats the key as a command name and fails. The variable lasts for the current terminal window only, so set it again in every new terminal:

   ```powershell
   $env:ANTHROPIC_API_KEY="your-key-here"
   ```

   On Mac/Linux: `export ANTHROPIC_API_KEY="your-key-here"`.

4. Check that it works:

   ```powershell
   python -c "import anthropic; print(anthropic.__version__)"
   ```

**Never put your key in the code and never commit it.** The SDK reads `ANTHROPIC_API_KEY` automatically. The `.gitignore` in this repo excludes `.venv/` and `.env`.

**Why a virtual environment?** A venv gives the project its own private set of packages, so different projects don't clash and a broken install can't affect the rest of your machine. It's the Python equivalent of `node_modules`.

**Cost tip:** `response.usage` in every script shows the tokens used. A typical run here is a few hundred tokens, which is a tiny fraction of a cent on Haiku.

---

## Step 1: Your first API call (`step1_basic.py`)

The smallest possible program: send one message, print the answer and how it ended.

```python
import anthropic

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY automatically

response = client.messages.create(
    model="claude-haiku-5-5",
    max_tokens=300,
    messages=[
        {"role": "user", "content": "What's my name?"},
    ],
)

print(response.content[0].text)
print("--- usage ---")
print(response.usage)
print("stop reason:", response.stop_reason)
```

### What is going on

- `anthropic.Anthropic()` creates the client and reads your key from the environment variable.
- `messages` is a list of `{role, content}` items. It is the **only** memory the model has. Roles are `user` (you) and `assistant` (the model).
- `max_tokens` caps the length of the answer.
- `response.content` is a **list of blocks**. For now there is one text block, so `[0].text` works. In later steps you'll see why that isn't always safe.
- `usage` shows input and output tokens, which is what you pay for.
- `stop_reason` says why the model stopped: `end_turn` (finished naturally) or `max_tokens` (cut off).

This version asks "What's my name?" with no history, so the model should say it doesn't know. That's the point: **the API is stateless.**

### Things to try

- **Give it a memory.** Replace `messages` with the full conversation and watch the answer change:

  ```python
  messages=[
      {"role": "user", "content": "My name is Eldar."},
      {"role": "assistant", "content": "Nice to meet you, Eldar!"},
      {"role": "user", "content": "What's my name?"},
  ],
  ```

  Compare `input_tokens` with and without the history. You pay again for the history on every call, which is why long chats get expensive and eventually hit the context window.
- **Truncate it.** Set `max_tokens=20`. The answer stops mid-sentence and `stop reason` becomes `max_tokens`. Real applications must check this.
- **Change the question.** Ask for a poem, a summary, or an explanation "like I'm five" and compare the output tokens.
- **Compare models.** Swap the model string for `claude-sonnet-5-5` and compare the answer, speed, and token counts. Pick the cheapest model that does the job.
- **Look at the raw object.** Run `print(response)` to see everything the API returns.
- **Challenge:** build a chat loop. Keep a `messages` list, read input with `input()`, append each user message and each model reply, and send the whole list every turn. This is how every chatbot works.

---

## Step 2: System prompts (`step2_system.py`)

The `system` parameter holds standing instructions (role, language, format, limits). It is separate from `messages`.

```python
import anthropic

client = anthropic.Anthropic()

question = "Give me a name for a tool that analyzes chip test logs."

systems = {
    "one name only": "You are a naming assistant for hardware engineers. Reply with exactly one name and nothing else.",
    "in Hebrew": "You are a naming assistant for hardware engineers. Reply with exactly one name, in Hebrew, and nothing else.",
    "detailed": "You are a naming assistant for hardware engineers. Give three name ideas, each with a one-sentence explanation.",
}

for label, system_prompt in systems.items():
    print(f"\n=== system prompt: {label} ===")
    response = client.messages.create(
        model="claude-haiku-5-5",
        max_tokens=300,
        system=system_prompt,
        messages=[{"role": "user", "content": question}],
    )
    print(response.content[0].text)
```

### What is going on

- The user question is identical in all three calls. Only the system prompt changes, and the behavior changes completely.
- `systems` is a dictionary used to keep three prompts in one place. The key is only a label for the printout; the model never sees it. The value is the text sent as `system=`.
- `for label, system_prompt in systems.items()` runs the call three times, once per entry.
- A system prompt typically says who the model is, then how to answer (format, language, length).

**Note on temperature:** temperature controls randomness (low = consistent, high = varied). On newer Claude models (released after Opus 4.6) it can no longer be set, and the default 1.0 is used, so passing `temperature=0` raises an error. Many other LLM APIs still expose it. On models where you can't set it, consistency comes from precise prompts, enforced schemas, and validation in your code.

### Things to try

- **See the randomness.** Run just the "one name only" call five times in a loop. The names will vary, because the default temperature is fixed at 1.0.
- **Write a new system prompt.** Add a fourth entry, for example a pirate, a very formal engineer, or "reply in exactly five words", and see how well the model follows it.
- **System prompt vs user message.** Put the rules in the user message instead of `system=` and compare how reliably they're followed.
- **Break the rules on purpose.** Add a conflicting instruction ("reply with one name" and "explain your choice") and see which one wins.
- **Watch the token cost.** Print `response.usage.input_tokens` for each call. The system prompt counts as input on every call.
- **Challenge:** make a system prompt that makes the model answer only about chip testing and politely refuse everything else.

---

## Step 3: Structured output with a prompt only (`step3_structured.py`)

Ask the model to extract fields from a chip test log line and reply only with JSON, then parse it with `json.loads`.

```python
import json
import anthropic

client = anthropic.Anthropic()

#log_line = "TEST_ID=1042 CHIP=A7 RESULT=FAIL VOLTAGE=0.82V TEMP=85C ERROR=timing_violation"
# 1. Missing field
#log_line = "TEST_ID=1043 CHIP=B2 RESULT=PASS VOLTAGE=0.90V"

# 2. Different format
#log_line = "chip A7 failed test 1044 at 0.82 volts, 85 degrees, timing problem"

# 3. Something that isn't a log line
log_line = "What's the weather like today?"

response = client.messages.create(
    model="claude-haiku-5-5",
    max_tokens=300,
    system=(
        "You extract data from chip test log lines. "
        "Reply with only a JSON object with these fields: "
        "test_id (integer), chip (string), passed (boolean), "
        "voltage (number, in volts), temperature_c (number), error (string or null). "
        "No other text."
    ),
    messages=[{"role": "user", "content": log_line}],
)

print("block types:", [block.type for block in response.content])

text = next((block.text for block in response.content if block.type == "text"), None)
if text is None:
    raise SystemExit("No text block in the response")

print("raw:", text)

try:
    data = json.loads(text)
    print("parsed:", data)
    print("passed?", data["passed"])
except json.JSONDecodeError:
    print("Model did not return valid JSON")

print("input tokens:", response.usage.input_tokens)
print("output tokens:", response.usage.output_tokens)
print("total tokens:", response.usage.input_tokens + response.usage.output_tokens)
```

### What is going on

- The system prompt describes the exact fields and types, and says "no other text".
- `block types` shows what came back. `response.content[0]` is **not always text**: for unusual input the model may return a `thinking` block first, and `.text` would crash. So the script looks for the block whose type is `"text"` instead of relying on position.
- `json.loads` turns the text into a Python dictionary. The `try/except` keeps the script from crashing if the text isn't valid JSON.

### What this step teaches

On clean input it works well, and the output is very stable because the task has one right answer and the prompt pins down the format. But the weather question showed the limit: the model **improvised**. It returned `null` for fields that should be integers or booleans and used the `error` field to carry a message. Downstream code would have read `passed = None` as "failed" without any warning. **A strict prompt is a request, not a guarantee.**

### Things to try

Uncomment the `log_line` options one at a time (only one should be active) and for each one check: did it return valid JSON, did it invent values, and did the script crash?

- **Clean line:** works, same result every time.
- **Missing field:** the model returns `null` for the missing temperature instead of inventing a value. Note that my prompt only allowed `null` for `error`, so it went beyond the spec.
- **Different format:** it extracts from free text, but copies wording (`"timing problem"` vs `"timing_violation"`), which would break a program expecting fixed error codes.
- **Not a log line:** valid JSON, but broken meaning. This is the case Step 4 fixes.
- **Make it break.** Change "No other text." to "Add a friendly explanation after the JSON." and watch `json.loads` fail. Then you'll see why the instruction matters.
- **Print `response.usage`** on the weather question and compare output tokens with a clean line. The thinking step costs tokens.

---

## Step 4: Enforced schema (`step4_schema.py`)

The fix: define the schema as a Pydantic class and let the API enforce it (structured outputs).

```python
import anthropic
from pydantic import BaseModel

class LogEntry(BaseModel):
    is_log_line: bool
    test_id: int | None
    chip: str | None
    passed: bool | None
    voltage: float | None
    temperature_c: float | None
    error: str | None

client = anthropic.Anthropic()

# log_line = "What's the weather like today?"

log_line = "TEST_ID=1042 CHIP=A7 RESULT=FAIL VOLTAGE=0.82V TEMP=85C ERROR=timing_violation"

response = client.messages.parse(
    model="claude-haiku-5-5",
    max_tokens=1000,
    system=(
        "You extract data from chip test log lines. "
        "If the input is not a chip test log line, set is_log_line to false "
        "and every other field to null."
    ),
    messages=[{"role": "user", "content": log_line}],
    output_format=LogEntry,
)

print("stop reason:", response.stop_reason)

if response.stop_reason != "end_turn":
    raise SystemExit("Response incomplete or refused, don't use the data")

entry = response.parsed_output
print(entry)

if entry.is_log_line:
    print("passed?", entry.passed)
else:
    print("Not a log line, skipping")

print("input tokens:", response.usage.input_tokens)
print("output tokens:", response.usage.output_tokens)
print("total tokens:", response.usage.input_tokens + response.usage.output_tokens)
```

### What is going on

- `class LogEntry(BaseModel)` is the schema written as a Python class. `int | None` means "an integer or null".
- `client.messages.parse(..., output_format=LogEntry)` converts the class to a JSON schema, sends it, and validates the reply. The API constrains the output to match the schema.
- `response.parsed_output` is a real `LogEntry` object with correct types: `entry.passed` is a real boolean, `entry.voltage` a float.
- **`is_log_line` gives the model an honest way to say "no".** Without it, the model had to abuse other fields (Step 3). Your code checks it first and skips the rest.
- **The `stop_reason` check:** output can still be incomplete (`max_tokens`) or refused, so check that the model finished normally before trusting the data.
- Input tokens are higher than you'd expect for a short question, because the schema and an extra system prompt added by structured outputs count as input.

### Results

| Input | Prompt only (Step 3) | Schema + `is_log_line` (Step 4) |
|---|---|---|
| Real log line | Worked | Worked |
| Not a log line | Improvised, hid the problem in `error` | `is_log_line=False`, code skips it |

### Things to try

- **Swap the input.** Run the weather question, then the real log line, then the free-text one from Step 3 (`"chip A7 failed test 1044 at 0.82 volts, 85 degrees, timing problem"`). Compare with Step 3.
- **Add a field.** For example `severity: str | None`, and update the system prompt to explain it. See how the model fills it.
- **Restrict the values.** Use `from typing import Literal` and change `error` to `Literal["timing_violation", "power_failure", "other"] | None`. This becomes an enum in the schema. Per the API docs, capitalization of enum values isn't guaranteed, so compare them case-insensitively in real code.
- **Make a field required for real.** Remove `| None` from `chip` and see what happens with the weather question.
- **Break the budget.** Lower `max_tokens` to a very small number and see what the script does. Always handle incomplete output.
- **Validate yourself.** Try creating `LogEntry(is_log_line=True, test_id="abc", ...)` by hand and read the Pydantic error. This is the "validate in your own code" layer.

---

## Step 5: Tool calling (`step5_tools.py`)

The model never runs your function. It can only **ask** for it. Your code runs it and sends the result back, and then the model writes the final answer. This loop is the foundation of every agent.

```
You → model:  "How is chip A7 doing?" + a description of your tool
Model → you:  "Please call get_test_results with chip_id='A7'"   (stop_reason = tool_use)
You:          run the real Python function yourself
You → model:  the function's result
Model → you:  the final answer in plain language
```

```python
import json
import anthropic

client = anthropic.Anthropic()
MODEL = "claude-haiku-5-5"

# 1. The real function. Fake data, just for practice.
TEST_RESULTS = {
    "A7": {"tests_run": 120, "failed": 3, "last_error": "timing_violation"},
    "B2": {"tests_run": 98, "failed": 0, "last_error": None},
}

def get_test_results(chip_id: str) -> dict:
    return TEST_RESULTS.get(chip_id, {"error": f"No data for chip {chip_id}"})

# 2. Describe the function to the model.
tools = [
    {
        "name": "get_test_results",
        "description": "Get the test summary for a chip: tests run, tests failed, and the last error.",
        "input_schema": {
            "type": "object",
            "properties": {
                "chip_id": {"type": "string", "description": "Chip identifier, for example A7"}
            },
            "required": ["chip_id"],
        },
    }
]

#messages = [{"role": "user", "content": "How is chip A7 doing in testing?"}]
#messages = [{"role": "user", "content": "How is chip B2 doing?"}]
messages = [{"role": "user", "content": "What's the capital of France?"}]

# 3. First call: the model decides whether it needs the tool.
response = client.messages.create(model=MODEL, max_tokens=1000, tools=tools, messages=messages)
total_tokens = response.usage.input_tokens + response.usage.output_tokens

print("stop reason:", response.stop_reason)
print("block types:", [block.type for block in response.content])

if response.stop_reason == "tool_use":
    tool_call = next(block for block in response.content if block.type == "tool_use")
    print("model asked for:", tool_call.name, tool_call.input)

    # 4. YOUR code runs the function.
    result = get_test_results(**tool_call.input)
    print("function returned:", result)

    # 5. Send the model's request and your result back, then ask again.
    messages.append({"role": "assistant", "content": response.content})
    messages.append({
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": tool_call.id, "content": json.dumps(result)}
        ],
    })
    response = client.messages.create(model=MODEL, max_tokens=1000, tools=tools, messages=messages)
    total_tokens += response.usage.input_tokens + response.usage.output_tokens

final_text = next((block.text for block in response.content if block.type == "text"), "")
print("\nFINAL ANSWER:", final_text)
print("total tokens (both calls):", total_tokens)
```

### What is going on

- `tools=[...]` describes your function: a name, a **description** (the model reads it to decide when to use the tool), and a JSON schema for the inputs. Good descriptions matter.
- `stop_reason == "tool_use"` means "I'm not done, I need you to run something".
- The `tool_use` block contains the tool name, its input (`{"chip_id": "A7"}`), and an `id`.
- Your code runs `get_test_results`, not the model. The `function returned:` line is printed by your Python code.
- The `tool_result` block must carry the same `tool_use_id`, so the model knows which request you're answering.
- The model's whole reply (`response.content`) is appended to the history unchanged before the result. If it contains a thinking block, it has to be passed back exactly as received.
- Two API calls, so tokens are summed. The second call includes the whole history again (statelessness, from Step 1).

### Things to try

- **Swap the question.** Uncomment each `messages` line in turn. For A7 and B2 the model asks for the tool (`stop_reason: tool_use`). For "capital of France" it answers directly (`end_turn`, one text block), because it decided it didn't need the tool. Compare the total tokens: a tool round trip costs about double.
- **Unknown chip.** Ask about chip C9. The function returns an error dictionary; see how the model explains it to the user.
- **Restrict the scope.** Add this to both `create(...)` calls and ask the France question again:

  ```python
  system="You are an assistant for chip test engineers. Only answer questions about chip test data. For anything else, politely say you can only help with chip testing.",
  ```
- **Improve the tool description.** Make it vague ("gets data") and see whether the model still picks the tool correctly. Descriptions steer the model.
- **Add a second tool**, for example `list_failing_chips()` that returns the chips with failures, and ask "Which chips have failures?". The model chooses between tools.
- **Challenge (a real gotcha):** ask "Compare chips A7 and B2". The model may request **several tools in one reply**, and this script only handles the first `tool_use` block. The API expects a `tool_result` for every request. Fix it by looping over all `tool_use` blocks and returning one `tool_result` for each.
- **Challenge:** wrap steps 3 to 5 in a `while` loop that continues until `stop_reason` is `end_turn`. That is an agent.

---

## Key ideas to remember

- **The API is stateless.** The model only knows what's in `messages`. A chat is a loop that resends the history.
- **Tokens are the unit of cost and of the context window.** Input includes the system prompt, the schema, and the history.
- **Always check `stop_reason`:** `end_turn`, `max_tokens`, `tool_use`.
- **`response.content` is a list of blocks** (`text`, `thinking`, `tool_use`). Find the block you need by `type`.
- **A prompt is a request; a schema is a guarantee.**
- **The model asks, your code acts.** Tool calling is how a model gets data and takes actions.
- **Pick the cheapest model that passes your tests.**

## Troubleshooting

| Problem | Cause and fix |
|---|---|
| `The term 'sk-ant-...' is not recognized` | Missing quotes. Use `$env:ANTHROPIC_API_KEY="your-key"`. |
| Key works in one terminal but not a new one | The variable only lasts for that terminal window. Set it again. |
| `TypeError: ... unexpected keyword argument 'temperature'` | Newer Claude models don't support setting temperature. Remove it. |
| `AttributeError: 'ThinkingBlock' object has no attribute 'text'` | The first content block isn't always text. Find the block with `type == "text"`. |
| `can't open file 'step4_schema.py'` | The file name is missing `.py` or the file is in another folder. Rename it and run from the project folder. |
| No credit or billing error | Add credit on the Console's Billing page. |

## Next steps

- A tiny RAG app (chunking, embeddings, vector search, answer with sources), and experiments with chunk size.
- A hand-written agent loop with 2 or 3 tools and guardrails (iteration limit, argument validation, error handling).
- An MCP server that exposes a couple of tools, connected to an agent.