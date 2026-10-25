"""Fetch the Unit 4 questions, answer them with the agent, and optionally submit.

    python run.py              # answer all questions not answered yet (saved in answers.json)
    python run.py --redo ID    # re-answer one task_id
    python run.py --submit     # submit answers.json to the scoring API
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import requests

API = "https://agents-course-unit4-scoring.hf.space"
USERNAME = "downey112"  # exact Hugging Face username (lowercase); the certificate checker matches it exactly
AGENT_CODE = "https://github.com/Downey112/hf-agents-course-gaia/tree/main"
ANSWERS = Path("answers.json")
FILES = Path("files")
QUESTION_TIMEOUT = 600  # seconds; a stuck question is skipped, not allowed to block the rest


def load_answers():
    return json.loads(ANSWERS.read_text(encoding="utf-8")) if ANSWERS.exists() else {}


def download_file(task):
    """Get the task's attachment from the course API, else from the gated GAIA dataset."""
    if not task.get("file_name"):
        return None
    FILES.mkdir(exist_ok=True)
    path = FILES / task["file_name"]
    if path.exists():
        return path.as_posix()
    r = requests.get(f"{API}/files/{task['task_id']}", timeout=60)
    if r.ok:
        path.write_bytes(r.content)
        return path.as_posix()
    # The course API stopped serving files; fetch only the attachment (never the answer metadata)
    from dotenv import load_dotenv
    from huggingface_hub import hf_hub_download

    load_dotenv(Path(__file__).resolve().parents[2] / "hf-first-agent" / ".env")
    cached = hf_hub_download(
        "gaia-benchmark/GAIA", f"2023/validation/{task['file_name']}",
        repo_type="dataset", token=os.getenv("HF_TOKEN"),
    )
    path.write_bytes(Path(cached).read_bytes())
    return path.as_posix()


def answer_one(task_id):
    """Child-process entry point: answer one task and print the answer as JSON on the last line."""
    from agent import answer

    task = next(t for t in requests.get(f"{API}/questions", timeout=60).json() if t["task_id"] == task_id)
    print(json.dumps({"answer": answer(task["question"], download_file(task))}, ensure_ascii=False))


def run(redo=None):
    questions = requests.get(f"{API}/questions", timeout=60).json()
    answers = load_answers()
    for i, task in enumerate(questions, 1):
        tid = task["task_id"]
        if (redo and tid != redo) or (not redo and tid in answers):
            continue
        print(f"\n===== {i}/{len(questions)} {tid}\n{task['question'][:200]}", flush=True)
        try:
            if task.get("file_name"):
                download_file(task)  # fail fast here if the attachment is unavailable
            proc = subprocess.run(
                [sys.executable, __file__, "--one", tid],
                capture_output=True, text=True, encoding="utf-8", timeout=QUESTION_TIMEOUT,
            )
            print(proc.stdout[-3000:], proc.stderr[-1500:], sep="\n", flush=True)
            result = json.loads(proc.stdout.strip().splitlines()[-1])["answer"]
            if not result:
                print("NO ANSWER: agent returned nothing usable", flush=True)
                continue
            answers[tid] = result
        except subprocess.TimeoutExpired:
            print(f"SKIPPED: no answer within {QUESTION_TIMEOUT}s", flush=True)
            continue
        except Exception as e:
            print(f"FAILED: {type(e).__name__}: {str(e)[:300]}", flush=True)
            continue
        print(f">>> ANSWER: {answers[tid]}", flush=True)
        ANSWERS.write_text(json.dumps(answers, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n{len(answers)}/{len(questions)} answered", flush=True)


def submit():
    answers = load_answers()
    payload = {
        "username": USERNAME,
        "agent_code": AGENT_CODE,
        "answers": [{"task_id": t, "submitted_answer": a} for t, a in answers.items()],
    }
    r = requests.post(f"{API}/submit", json=payload, timeout=120)
    print(r.status_code, json.dumps(r.json(), indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--submit", action="store_true")
    p.add_argument("--redo")
    p.add_argument("--one")
    a = p.parse_args()
    if a.one:
        answer_one(a.one)
    elif a.submit:
        submit()
    else:
        run(a.redo)
