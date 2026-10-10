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