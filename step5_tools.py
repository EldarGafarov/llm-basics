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