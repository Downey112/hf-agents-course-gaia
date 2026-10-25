"""GAIA agent for the Hugging Face Agents Course (Unit 4).

A smolagents CodeAgent running on Google's free Gemini API, with tools for web
search, webpages, Wikipedia, attached files and media (images, audio, YouTube).
"""

import base64
import mimetypes
import os
import re
import time
from pathlib import Path

import requests
from dotenv import load_dotenv
from markdownify import markdownify
from smolagents import CodeAgent, DuckDuckGoSearchTool, OpenAIServerModel, VisitWebpageTool, tool

load_dotenv()
# Reuse the key file from the Unit 1 project if this folder has no .env of its own
load_dotenv(Path(__file__).resolve().parents[2] / "hf-first-agent" / ".env")

GEMINI_KEY = os.environ["GEMINI_API_KEY"]
GEMINI_NATIVE = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
GEMINI_OPENAI = "https://generativelanguage.googleapis.com/v1beta/openai/"
# Tried in order; free-tier models are often busy (503) or out of daily quota (429)
MODEL_IDS = ["gemini-3-flash-preview", "gemini-3.6-flash", "gemini-3.5-flash",
             "gemini-flash-lite-latest", "gemini-3.1-flash-lite"]
MEDIA_MODEL_IDS = ["gemini-3-flash-preview", "gemini-3.6-flash", "gemini-3.5-flash", "gemini-flash-lite-latest"]
QUOTA_COOLDOWN = 600  # seconds to skip a model after a 429, instead of retrying it
_blocked_until = {}


def _available(model_ids):
    now = time.time()
    return [m for m in model_ids if _blocked_until.get(m, 0) <= now] or model_ids[-1:]


def _note_failure(model_id, error):
    if "429" in str(error) or "quota" in str(error).lower():
        _blocked_until[model_id] = time.time() + QUOTA_COOLDOWN


class FallbackGeminiModel(OpenAIServerModel):
    """OpenAIServerModel that skips models out of quota and falls back to the next one."""

    def __init__(self, model_ids, **kwargs):
        self.model_ids = model_ids
        super().__init__(
            model_id=model_ids[0], api_base=GEMINI_OPENAI, api_key=GEMINI_KEY,
            client_kwargs={"timeout": 90, "max_retries": 0},
            retry=False,  # smolagents' own backoff waits 60s+ per 429; we fall back to another model instead
            **kwargs,
        )

    def generate(self, *args, **kwargs):
        last_error = None
        for attempt in range(3):
            for model_id in _available(self.model_ids):
                self.model_id = model_id
                try:
                    return super().generate(*args, **kwargs)
                except Exception as e:  # 429/503/timeouts: try the next model
                    last_error = e
                    _note_failure(model_id, e)
            time.sleep(10 * (attempt + 1))
        raise last_error


def _gemini_native(parts):
    """Call Gemini's native API with text/media parts and return the reply text."""
    last_error = None
    for attempt in range(3):
        for model in _available(MEDIA_MODEL_IDS):
            try:
                r = requests.post(
                    GEMINI_NATIVE.format(model=model),
                    headers={"x-goog-api-key": GEMINI_KEY},
                    json={"contents": [{"parts": parts}]},
                    timeout=180,
                )
                if r.ok:
                    return "".join(p.get("text", "") for p in r.json()["candidates"][0]["content"]["parts"])
                last_error = f"HTTP {r.status_code}: {r.text[:200]}"
                _note_failure(model, last_error)
            except Exception as e:
                last_error = str(e)
        time.sleep(10 * (attempt + 1))
    return f"Error from Gemini: {last_error}"


@tool
def analyze_media(source: str, question: str) -> str:
    """Answers a question about an image, audio file, video file or YouTube video by looking at/listening to it.
    Use this for any .png/.jpg/.mp3/.wav/.mp4 file or a YouTube link; ask a specific, detailed question.
    Args:
        source: A local file path (e.g. 'files/abc.mp3') or a YouTube URL.
        question: What you want to know about the media, e.g. 'List every ingredient mentioned'.
    """
    if source.startswith("http"):
        media = {"file_data": {"file_uri": source}}
    else:
        path = Path(source)
        if not path.exists():
            return f"Error: file '{source}' not found."
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        media = {"inline_data": {"mime_type": mime, "data": base64.b64encode(path.read_bytes()).decode()}}
    return _gemini_native([media, {"text": question}])


@tool
def read_text_file(path: str) -> str:
    """Reads a text-based file (e.g. .py, .txt, .csv, .json, .md) and returns its contents.
    For Excel files use pandas.read_excel in code instead.
    Args:
        path: Local path of the file, e.g. 'files/abc.py'.
    """
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")[:40000]
    except Exception as e:
        return f"Error reading '{path}': {e}"


@tool
def wikipedia_page(title: str, language: str = "en") -> str:
    """Gets the full content of a Wikipedia article as markdown, including its tables (e.g. discographies, lists, results).
    Prefer this over web search for anything Wikipedia covers.
    Args:
        title: The article title, e.g. 'Mercedes Sosa' or 'Mercedes Sosa discography'.
        language: Wikipedia language code, e.g. 'en' or 'pl'.
    """
    try:
        r = requests.get(
            f"https://{language}.wikipedia.org/w/api.php",
            params={"action": "parse", "page": title, "prop": "text", "format": "json", "redirects": 1},
            headers={"User-Agent": "hf-agents-course-gaia/1.0"},
            timeout=30,
        ).json()
        if "error" in r:
            return f"Error: {r['error'].get('info')}. Try web_search to find the exact title."
        text = markdownify(r["parse"]["text"]["*"])
        return f"# {r['parse']['title']}\n\n{text}"[:60000]
    except Exception as e:
        return f"Error fetching Wikipedia page '{title}': {e}"


ANSWER_RULES = """
Answer the question below. Use your tools to look things up instead of guessing.
When you are done, call final_answer() with ONLY the answer, following these rules exactly:
- A number: digits only, no commas, no units (no '$' or '%') unless the question asks for them.
- A string: as few words as possible, no articles, no abbreviations, digits as digits.
- A list: comma-separated, following the above rules for each item, in the order/format the question asks.
- Follow any formatting the question requests (alphabetical order, IOC code, first name only, etc.).
- Never include explanations or the words 'FINAL ANSWER' in final_answer().
"""


def build_agent():
    return CodeAgent(
        tools=[DuckDuckGoSearchTool(), VisitWebpageTool(), wikipedia_page, analyze_media, read_text_file],
        model=FallbackGeminiModel(MODEL_IDS, temperature=0.2),
        additional_authorized_imports=["pandas", "numpy", "openpyxl", "math", "datetime", "re", "json", "collections", "itertools"],
        max_steps=12,
        verbosity_level=1,
    )


def answer(question: str, file_path: str | None = None) -> str:
    task = ANSWER_RULES + "\nQuestion: " + question
    if file_path:
        task += f"\n\nThe question's attached file is at: {file_path}"
    for _ in range(2):  # one retry when the agent hands back code or nothing
        result = clean_answer(build_agent().run(task))
        if result:
            return result
    return ""


def clean_answer(result) -> str:
    """Turn the agent's raw output into a bare answer, or '' if it is not one."""
    text = str(result).replace("FINAL ANSWER:", "").strip()
    # The model sometimes returns its final code block instead of running it
    called = re.search(r"final_answer\(\s*(['\"])(.*?)\1\s*\)", text, re.S)
    if called:
        return called.group(2).strip()
    if text.startswith("```") or text in ("", "None"):
        return ""
    return text
