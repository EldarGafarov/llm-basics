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