# Hybrid-Token-Efficient-Routing-Agent

ScoobaCats submission for the **AMD Developer Hackathon ACT II, Track 1
(General-Purpose AI Agent)**.

A batch container that reads NL tasks, routes each to the most token-efficient
model in the harness-provided allowlist, and writes back results. Ranked by
accuracy-gate pass + ascending token count.

## Container contract

| | |
|---|---|
| Base image | `python:3.12-slim` (`linux/amd64`) |
| Entrypoint | `python -m app.main` (batch — no server) |
| Reads | `/input/tasks.json` |
| Writes | `/output/results.json` |
| Env in | `FIREWORKS_API_KEY`, `FIREWORKS_BASE_URL`, `ALLOWED_MODELS` |
| Runtime cap | 10 min total, 30 s per request, 60 s cold-start |
| Image cap | 10 GB compressed |

All Fireworks calls go through `FIREWORKS_BASE_URL` (harness proxy — records
tokens for scoring). Direct-to-Fireworks calls score zero. Model IDs are read
from `ALLOWED_MODELS` at runtime — never hardcoded.

## Build

Judging VM is `linux/amd64`. On Apple Silicon you **must** cross-build:

```sh
docker buildx build --platform linux/amd64 \
  --tag ghcr.io/<owner>/scoobacats-track1:dev \
  --push .
```

On an Intel/AMD host or GitHub Actions:

```sh
docker build -t scoobacats-track1:dev .
```

## Local test

Use `local_test/tasks.json` as sample input. Point `FIREWORKS_BASE_URL` at your
own Fireworks endpoint for dev; the harness will inject its proxy URL at
evaluation time.

```sh
mkdir -p out
docker run --rm \
  --env FIREWORKS_API_KEY=$FIREWORKS_API_KEY \
  --env FIREWORKS_BASE_URL=https://api.fireworks.ai/inference/v1 \
  --env ALLOWED_MODELS=accounts/fireworks/models/qwen3-30b-a3b \
  -v "$PWD/local_test":/input:ro \
  -v "$PWD/out":/output \
  scoobacats-track1:dev

cat out/results.json
```

## Extension points

The skeleton always picks the first model in `ALLOWED_MODELS`. The token-
efficiency ranking is won by upgrading these two functions in `app/main.py`:

- `choose_model(task, allowed)` — classify the prompt into one of the 8
  capability buckets (factual / math / sentiment / summarisation / NER /
  code-debug / logic / code-gen) and pick the cheapest model that will still
  clear the accuracy gate for that bucket.
- `run_task(...)` — add capability-specific system prompts, response-length
  hints, and verify-then-shrink loops. Every saved token moves you up the
  leaderboard.

## Non-goals for Track 1

- No local model inference. Local tokens count as **zero** — all inference
  must be a Fireworks call through `FIREWORKS_BASE_URL`.
- No AMD/ROCm/MI300X in the runtime path. The judging VM is a plain
  `linux/amd64` host.
- No persistent server. Batch in, batch out, exit 0.
