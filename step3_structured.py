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