# LLM API Basics: a hands-on guide (Python + Claude API)

A step-by-step practice project for learning the fundamentals of working with an LLM API. Each step is a small script that teaches one idea, followed by a list of experiments you can run by editing a line or two. Written while preparing for a junior/mid GenAI + cloud software engineer interview.

**Stack:** Python 3.12, `anthropic` SDK, `pydantic`, `fastembed` + `numpy` (local embeddings for the RAG steps), model `claude-haiku-5-5` (cheap and fast, good enough for practice).

**How to use this guide:** run each script as-is first, read the output, then work through its "Things to try" list. Changing one thing at a time and watching what happens is how the ideas stick.

## Contents

1. [Setup](#setup)
2. [Step 1: Your first API call](#step-1-your-first-api-call-step1_basicpy)
3. [Step 2: System prompts](#step-2-system-prompts-step2_systempy)
4. [Step 3: Structured output with a prompt only](#step-3-structured-output-with-a-prompt-only-step3_structuredpy)
5. [Step 4: Enforced schema](#step-4-enforced-schema-step4_schemapy)
6. [Step 5: Tool calling](#step-5-tool-calling-step5_toolspy)
7. [Step 6: Embeddings](#step-6-embeddings-meaning-as-numbers-step6_embeddingspy)
8. [Step 7: Chunking](#step-7-chunking-step7_chunkingpy)
9. [Step 8: The full RAG pipeline](#step-8-the-full-rag-pipeline-step8_ragpy)
10. [Making model output predictable](#making-model-output-predictable-interview-summary)
11. [Key ideas to remember](#key-ideas-to-remember)
12. [Troubleshooting](#troubleshooting)
13. [Next steps](#next-steps)

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

## Step 6: Embeddings, meaning as numbers (`step6_embeddings.py`)

No Claude involved here. An **embedding** turns a piece of text into a list of numbers so that **texts with similar meaning get similar numbers**. Then "search by meaning" becomes "find the closest numbers".

Anthropic doesn't offer its own embedding model (its docs point to Voyage AI, which needs a separate account and key). To keep this free and simple, the guide runs a small embedding model **locally** with `fastembed`. The concepts are identical with a hosted model. The first run downloads the model from Hugging Face, so it needs internet and takes a minute.

```powershell
pip install fastembed numpy
```

```python
import numpy as np
from fastembed import TextEmbedding

model = TextEmbedding("BAAI/bge-small-en-v1.5")

documents = [
    "Chip A7 fails timing tests when the supply voltage drops below 0.85V.",
    "To reset the test fixture, power cycle the board and wait 30 seconds.",
    "The weekly yield report is generated every Monday from the test database.",
    "Thermal throttling starts at 95 degrees Celsius to protect the chip.",
    "Calibration of the oscilloscope must be done every six months.",
]

doc_vectors = np.array(list(model.embed(documents)))
print("shape:", doc_vectors.shape)  # (number of documents, size of each vector)

query = "My processor stops working correctly when the power is too low"
query_vector = np.array(list(model.embed([query])))[0]

def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

scores = [cosine(query_vector, v) for v in doc_vectors]

for score, doc in sorted(zip(scores, documents), reverse=True):
    print(f"{score:.3f}  {doc}")
```

### What is going on

- `model.embed(...)` returns one vector per text. `shape: (5, 384)` means 5 documents, each turned into **384 numbers**.
- The 384 is fixed by the model. Other models give 768, 1024 or more. Bigger vectors can hold more detail but cost more to store and search.
- No single number means anything you could name ("number 17 is voltage"). The meaning is spread across all 384 together, like a point on a map that needs both latitude and longitude.
- **Cosine similarity** measures how closely two vectors point in the same direction. Closer to 1 means more similar meaning. The `cosine` function is `dot(a, b) / (norm(a) * norm(b))`.
- The query ("processor stops working when the power is too low") shares almost no words with the best document ("Chip A7 fails timing tests when the supply voltage drops below 0.85V"), yet it ranked first (0.762). That is **semantic search**.
- **This script never generates text.** It only ranks documents you gave it, so it can't make anything up. The "G" in RAG comes later, in Step 8.

### Reading the scores

- They are **not percentages or probabilities.**
- **Only the ranking is reliable.** The same score can mean different things with a different model or different texts.
- **There is no universal "good enough" cutoff.** Even the unrelated yield report scored 0.40. A score alone doesn't tell you whether the answer is actually in your documents.
- Always embed the question and the documents with **the same model**. Vectors from different models can't be compared.

### Things to try

- Change the query to something about cleaning the lab or temperature limits and see which document wins.
- Write a query that uses the exact words of a different document and compare the scores.
- Add documents of your own. For example, a document about Japan and the query "What can I do in Asia?" ranks Japan first (0.646) even though the word "Asia" is not in it.
- Print `doc_vectors[0][:8]` to see the first 8 of the 384 numbers.

---

## Step 7: Chunking (`step7_chunking.py`)

A real document is too long to embed as one vector or paste whole into every prompt. **Chunking** cuts it into pieces, and each piece gets its own vector. Retrieval then returns only the pieces that matter.

Why not embed the whole manual as one vector? One vector for nine different topics is a blur, and you would have to send the whole manual to the model on every question: more tokens (money), slower, and more noise. Small focused chunks mean precise retrieval and a short prompt.

The test document is a 1,943-character chip A7 manual with nine paragraphs, including two "noise" paragraphs (Japan and the cafeteria) that have nothing to do with testing:

```python
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

def chunk_text(text, chunk_size=300, overlap=50):
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return chunks

chunks = chunk_text(manual, 300, 50)
print(len(chunks), "chunks")
for i, c in enumerate(chunks):
    print(f"\n--- chunk {i} ({len(c)} chars) ---\n{c}")
```

### What is going on

- The chunker is **fixed-size with overlap**: take 300 characters, step forward by `300 - 50` so each chunk repeats the last 50 characters of the previous one.
- The overlap exists because a cut can land in the middle of a sentence. If the overlap is at least as long as the sentence that straddles the cut, the sentence survives whole in one chunk.
- With 300/50 the manual becomes 8 chunks and the key sentence ("Chip A7 is known to fail timing tests when the supply voltage drops below 0.85V.", 80 characters) stays intact. The overlap is shorter than the sentence, so this is partly luck.

### Experiment: chunk size and overlap

The key sentence sits at characters 415 to 495. Does it survive each setting?

| chunk_size | overlap | chunks | key sentence intact |
|---|---|---|---|
| 300 | 50 | 8 | yes (luck: overlap 50 is shorter than the 80-character sentence) |
| 150 | 0 | 13 | **no** |
| 600 | 100 | 4 | yes (guaranteed: overlap 100 is at least as long as the sentence) |
| 300 | 0 | 7 | yes (luck: the cut happened to miss it) |
| 100 | 20 | 25 | yes (luck) |
| 200 | 50 | 13 | yes (luck) |
| 200 | 0 | 10 | yes (luck) |

Lessons:

- **Only an overlap at least as long as the sentence guarantees it survives.** Most rows above "worked" by luck: the cut fell somewhere harmless. Change the manual and they could break.
- **Small chunks with no overlap can cut the sentence you need** (150/0 did).
- **Large chunks keep context but blur topics** (one vector for several ideas) and put more irrelevant text into the prompt.
- **Small chunks are precise but lose context** ("it fails below 0.85V": which chip?).
- Fixed-size chunking is blind to meaning. Step 8 uses paragraphs instead, which respect the document's own structure.

### Things to try

- Add your own paragraphs to the manual (more noise, or a second chip) and see how the chunks change.
- Print only the chunk that contains the key sentence for each setting.
- Find a size/overlap that cuts the key sentence in half, and write down why.

---

## Step 8: The full RAG pipeline (`step8_rag.py`)

RAG (Retrieval-Augmented Generation) is an open-book exam. Claude doesn't know your documents, so for each question you **find the relevant pieces, paste them into the prompt, and let Claude answer from them.**

```
Phase 1, once:       document -> chunks -> embeddings -> stored
Phase 2, per question:
  R  Retrieval   embed the question, find the top K similar chunks
  A  Augmentation paste those chunks into the prompt
  G  Generation  Claude reads the prompt and answers (with citations)
```

```python
import anthropic
import numpy as np
from fastembed import TextEmbedding

MODEL = "claude-haiku-5-5"
TOP_K = 3

manual = """..."""   # the same chip A7 manual string as in Step 7


# ---------- Phase 1: prepare (done once) ----------

def chunk_by_paragraph(text, min_chars=80):
    """One chunk per paragraph. A paragraph that is too short (like the title)
    is merged with the one after it."""
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
print(f"Indexed {len(chunks)} chunks, vectors shape {chunk_vectors.shape}")


# ---------- Phase 2: answer (done for every question) ----------

def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def retrieve(question, k=TOP_K):
    """Return the k most similar chunks as (chunk_index, score)."""
    q_vector = np.array(list(embedder.embed([question])))[0]
    scores = [cosine(q_vector, v) for v in chunk_vectors]
    best = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)[:k]
    return [(i, scores[i]) for i in best]


SYSTEM = (
    "You answer questions about chip testing using ONLY the numbered context chunks provided. "
    "Cite the chunk numbers you used, like [2]. "
    "If the context does not contain the answer, reply exactly: I can't find that in the documents."
)

client = anthropic.Anthropic()


def ask(question):
    hits = retrieve(question)

    print(f"\nQUESTION: {question}")
    print("retrieved:")
    for i, score in hits:
        preview = chunks[i][:60].replace("\n", " ")
        print(f"  chunk {i}  score {score:.3f}  {preview}...")

    # Augmentation: build the prompt from the retrieved chunks
    context = "\n\n".join(f"[{i}] {chunks[i]}" for i, _ in hits)
    prompt = f"Context:\n{context}\n\nQuestion: {question}"

    response = client.messages.create(
        model=MODEL,
        max_tokens=1000,
        system=SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )

    if response.stop_reason != "end_turn":
        print("ANSWER: (incomplete or refused, stop reason:", response.stop_reason + ")")
        return

    answer = next((b.text for b in response.content if b.type == "text"), "")
    print("ANSWER:", answer)
    print("tokens:", response.usage.input_tokens + response.usage.output_tokens)


questions = [
    "At what supply voltage does chip A7 fail timing tests?",
    "What happens if the chip reaches 110C?",
    "When is the cafeteria open?",
    "What can I do in Asia?",
    "Who is the CEO of the company?",
    "What should I do if a timing violation appears at 85C and the chip is also hot?",
]

for q in questions:
    ask(q)
```

### What is going on

**Phase 1 (runs once).** `chunk_by_paragraph` cuts at the blank lines, so no sentence is ever split (the fix for the problem seen in Step 7). Each chunk is embedded, and the vectors are kept in `chunk_vectors`. Together, chunks plus vectors are the "database".

**Phase 2 (runs for every question).**

- **R, Retrieval:** `retrieve` embeds the question, scores it against every chunk with cosine similarity, and returns the top `K`.
- **A, Augmentation:** `ask` builds the prompt by pasting the top chunks, numbered `[0]`, `[1]`..., above the question. That is all augmentation is: **building the prompt**. Nothing is "understood" here. The chunks are just text placed in front of the model.
- **G, Generation:** Claude reads the prompt and writes the answer. This is the only step that uses the LLM.

Chunking and embedding are **setup**, not letters of RAG. They make retrieval possible but happen before any question arrives.

The **system prompt** does three jobs: it restricts Claude to the context ("ONLY"), asks for citations by chunk number, and gives it a fixed phrase to use when the answer isn't there. An exact refusal phrase is something your code can detect.

### Results with `TOP_K = 3`

| Question | What retrieval did | What Claude did |
|---|---|---|
| Supply voltage where A7 fails timing | Right chunk ranked first (top score about 0.88) | Correct, with citation |
| What happens at 110C | Right chunk ranked first (about 0.79) | Correct, with citation |
| Cafeteria hours | Cafeteria chunk first (0.82) | Answered from it. The question has nothing to do with chips, but it **is** in the documents. |
| What can I do in Asia? | Japan chunk retrieved (0.60) | "I can't find that in the documents": the chunk talks about Japan but gives no answer to the question |
| Who is the CEO? | Best score only 0.46, but **three chunks were returned anyway** | "I can't find that in the documents." |

### What this step teaches

- **Retrieval always returns something.** Even for a question nothing can answer, you still get the top K chunks. The system prompt is the safety net.
- **Separate retrieval problems from generation problems.** If an answer is wrong, first print the retrieved chunks. If the right chunk is missing, the fix is in chunking, the embedding model, or `K`. If it is present and the answer is still wrong, the fix is in the prompt.
- **`K` is a trade-off.** With `K = 1`, the two-part question ("timing violation at 85C and the chip is also hot") missed the thermal chunk. With `K = 3` it appeared. A larger `K` adds noise and tokens.
- **The guardrail matters.** With the "ONLY the context" wording removed, Claude did not invent facts in these tests, but it added suggestions that were not in the documents and lost the exact refusal phrase your code could detect. A too-narrow scope in the prompt ("chip testing") can also cause **false refusals**: the cafeteria question was refused when "ONLY..." was removed but the chip-testing scope stayed.
- **Telling Claude "do whatever reasoning you need, but don't make up facts"** still lets it add advice beyond the context. If you want strict grounding, say "Do not add advice, suggestions, or inferences that are not stated in the context."

### Things to try

- **Change `TOP_K`** to 1, 2, 5 and rerun all questions. Watch which questions break and how the token count changes.
- **Hallucination traps.** Ask "What is the clock speed of chip A7 after thermal throttling?" (the manual says it reduces the clock but gives no number) and "At what supply voltage does chip B2 fail timing tests?" (the manual is about A7 only). The correct behavior is the exact refusal phrase.
- **Swap the chunker.** Replace `chunk_by_paragraph` with `chunk_text` from Step 7 (`chunk_size=300`, `overlap=50`) and compare retrieved chunks and answers. If you import from `step7_chunking.py`, wrap its demo code in `if __name__ == "__main__":` first.
- **Remove or loosen the guardrail** in `SYSTEM` one part at a time and rerun the last three questions.
- **Add your own documents** (a second chip, a new error code) and write questions that need two chunks at once.
- **Challenge:** measure retrieval on its own. For each question, write down which chunk *should* come back and check whether it is in the top K, before involving Claude at all.

---

## Making model output predictable (interview summary)

1. Lower the temperature, where the model allows it.
2. Use a strict system prompt that states the exact format.
3. Use structured output with a schema, so the format is enforced and not just requested.
4. Validate in your own code (Pydantic, `stop_reason` checks) and handle failures. Never trust model output blindly.

## Key ideas to remember

- **The API is stateless.** The model only knows what's in `messages`. A chat is a loop that resends the history.
- **Tokens are the unit of cost and of the context window.** Input includes the system prompt, the schema, and the history.
- **Always check `stop_reason`:** `end_turn`, `max_tokens`, `tool_use`.
- **`response.content` is a list of blocks** (`text`, `thinking`, `tool_use`). Find the block you need by `type`.
- **A prompt is a request; a schema is a guarantee.**
- **The model asks, your code acts.** Tool calling is how a model gets data and takes actions.
- **Pick the cheapest model that passes your tests.**
- **An embedding is meaning as numbers.** Similar meaning gives nearby vectors. Use the same model for documents and questions.
- **Scores are only good for ranking.** There is no universal cutoff, and retrieval always returns something.
- **Chunking decides what retrieval can find.** Cut at natural boundaries, keep each chunk about one idea, and use overlap as insurance.
- **RAG = Retrieval, Augmentation, Generation.** Augmentation is just building the prompt from the retrieved chunks.
- **When a RAG answer is wrong, check retrieval first.** Print the retrieved chunks before touching the prompt.
- **Guardrails need an exact, detectable refusal phrase** so your code can tell "not in the documents" from an answer.

## Troubleshooting

| Problem | Cause and fix |
|---|---|
| `The term 'sk-ant-...' is not recognized` | Missing quotes. Use `$env:ANTHROPIC_API_KEY="your-key"`. |
| Key works in one terminal but not a new one | The variable only lasts for that terminal window. Set it again. |
| `TypeError: ... unexpected keyword argument 'temperature'` | Newer Claude models don't support setting temperature. Remove it. |
| `AttributeError: 'ThinkingBlock' object has no attribute 'text'` | The first content block isn't always text. Find the block with `type == "text"`. |
| `can't open file 'step4_schema.py'` | The file name is missing `.py` or the file is in another folder. Rename it and run from the project folder. |
| No credit or billing error | Add credit on the Console's Billing page. |
| `pip install fastembed` fails, or the first run of Step 6 hangs | The first run downloads the embedding model from Hugging Face. Check your internet connection and wait a minute. |
| Step 7 prints chunks when imported from another file | Put the demo code under `if __name__ == "__main__":` so importing only loads the functions. |
| RAG answers the wrong thing | Print the retrieved chunks and scores first. If the right chunk is missing, change chunking, `TOP_K`, or the embedding model. If it is there, change the prompt. |

## Next steps

- A hand-written agent loop with 2 or 3 tools (including a `search_manual` tool that wraps the Step 8 pipeline) and guardrails: iteration limit, argument validation, errors returned to the model, step logging.
- An MCP server that exposes a couple of tools, connected to the agent.
- RAG upgrades to explore: hybrid search (keyword + vector), reranking, metadata filters, query rewriting.