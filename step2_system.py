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