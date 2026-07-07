"""Track 1 entrypoint.

Reads tasks from /input/tasks.json, dispatches each to a Fireworks model via
the harness proxy, writes /output/results.json, exits 0. Skeleton router:
always picks the first model in ALLOWED_MODELS. Replace `choose_model` with
capability-aware routing to compete on token efficiency.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

from openai import AsyncOpenAI

INPUT_PATH = Path("/input/tasks.json")
OUTPUT_PATH = Path("/output/results.json")

# Spec caps: 10 min total, 30 s per request, 60 s cold-start.
# Leave headroom so we always emit a valid JSON file even under pressure.
TOTAL_BUDGET_S = 9 * 60
PER_REQUEST_TIMEOUT_S = 25
MAX_CONCURRENCY = 16


def load_env() -> tuple[str, str, list[str]]:
    api_key = os.environ["FIREWORKS_API_KEY"]
    base_url = os.environ["FIREWORKS_BASE_URL"]
    models = [m.strip() for m in os.environ["ALLOWED_MODELS"].split(",") if m.strip()]
    if not models:
        sys.exit("ALLOWED_MODELS is empty")
    return api_key, base_url, models


def choose_model(task: dict, allowed: list[str]) -> str:
    # Extension point: classify task -> pick cheapest model that will pass the
    # accuracy gate. Skeleton just uses the first allowed model.
    return allowed[0]


async def run_task(
    client: AsyncOpenAI, task: dict, allowed: list[str]
) -> dict:
    model = choose_model(task, allowed)
    try:
        resp = await asyncio.wait_for(
            client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": task["prompt"]}],
            ),
            timeout=PER_REQUEST_TIMEOUT_S,
        )
        answer = resp.choices[0].message.content or ""
    except Exception as e:
        # Preserve valid JSON contract; a missing answer is better than a crash.
        # Log to stderr so failures are visible without corrupting results.json.
        print(
            f"[{task['task_id']}] {type(e).__name__}: {e}",
            file=sys.stderr,
            flush=True,
        )
        answer = ""
    return {"task_id": task["task_id"], "answer": answer}


async def run(tasks: list[dict], client: AsyncOpenAI, allowed: list[str]) -> list[dict]:
    # Pre-seed results so a timeout still produces one entry per task.
    results = {t["task_id"]: {"task_id": t["task_id"], "answer": ""} for t in tasks}
    sem = asyncio.Semaphore(MAX_CONCURRENCY)

    async def bounded(task: dict) -> None:
        async with sem:
            results[task["task_id"]] = await run_task(client, task, allowed)

    try:
        await asyncio.wait_for(
            asyncio.gather(*(bounded(t) for t in tasks)),
            timeout=TOTAL_BUDGET_S,
        )
    except asyncio.TimeoutError:
        pass  # Fall through and emit whatever completed.

    # Preserve input order.
    return [results[t["task_id"]] for t in tasks]


async def main() -> int:
    api_key, base_url, allowed = load_env()
    tasks = json.loads(INPUT_PATH.read_text())

    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    results = await run(tasks, client, allowed)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(results))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
