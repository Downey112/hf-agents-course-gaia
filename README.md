# GAIA agent: Hugging Face Agents Course, Unit 4

My final-assignment agent for the [Hugging Face Agents Course](https://huggingface.co/learn/agents-course). It answers 20 level-1 questions from the [GAIA](https://huggingface.co/gaia-benchmark) validation set and submits them to the course's scoring API.

## How it works

A [smolagents](https://github.com/huggingface/smolagents) `CodeAgent` (it writes Python to call tools) running on Google's free Gemini API, with these tools:

| Tool | Used for |
|---|---|
| `web_search` (DuckDuckGo) | finding pages and facts |
| `visit_webpage` | reading a page found by search |
| `wikipedia_page` | full Wikipedia articles as markdown, tables included |
| `analyze_media` | images, audio and YouTube videos, read natively by Gemini |
| `read_text_file` | attached code and text files (Excel goes through pandas) |

Free-tier Gemini models are often busy, so the model wrapper retries and falls back across several Gemini Flash versions. The scoring is exact match, so the prompt tells the agent to return only the bare answer.

## Run

```bash
uv venv --python 3.11 .venv && uv pip install "smolagents[toolkit]" openai pandas openpyxl python-dotenv requests markdownify
echo GEMINI_API_KEY=your_key > .env    # free key: https://aistudio.google.com/apikey
python run.py            # answers every question, saving to answers.json
python run.py --submit   # sends answers.json for scoring
```

Hugging Face free accounts can't run Gradio Spaces, so this runs locally and submits to the scoring API directly; this repo is the `agent_code` link.
