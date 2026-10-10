# llm-basics

Small, step-by-step Python examples of calling the Claude API with the [Anthropic SDK](https://github.com/anthropics/anthropic-sdk-python). Each script builds on the previous one, using a running example: parsing chip test logs.

## Steps

| Script | What it shows |
| --- | --- |
| [step1_basic.py](step1_basic.py) | A basic `messages.create` call, plus reading token usage and the stop reason |
| [step2_system.py](step2_system.py) | How different system prompts change the model's output for the same question |
| [step3_structured.py](step3_structured.py) | Asking for JSON in the prompt and parsing it by hand (and handling bad output) |
| [step4_schema.py](step4_schema.py) | Structured output with a Pydantic schema via `messages.parse` |
| [step5_tools.py](step5_tools.py) | Tool use: the model requests a function call, your code runs it, and you send back the result |

## Setup

Requires Python 3.10+ and an [Anthropic API key](https://console.anthropic.com/).

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your-key   # Windows PowerShell: $env:ANTHROPIC_API_KEY = "your-key"
```

## Run

```bash
python step1_basic.py
```

Steps 3 to 5 have several sample inputs commented out in the source. Swap them in to see how the model handles missing fields, different formats, and off-topic input.
