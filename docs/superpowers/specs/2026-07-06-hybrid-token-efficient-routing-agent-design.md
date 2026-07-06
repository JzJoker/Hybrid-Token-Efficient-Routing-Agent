# Hybrid Token-Efficient Routing Agent — Design

**Status:** Draft, pending user review
**Date:** 2026-07-06
**Team:** ScoobaCats
**Track:** AMD Developer Hackathon ACT II, Track 1 ("General-Purpose AI Agent" per
the official submission guide; team has been calling it "Hybrid Token-Efficient
Routing Agent")

## 1. Goal

Build a Docker container that answers natural-language tasks across 8 capability
categories (factual knowledge, mathematical reasoning, sentiment classification,
text summarization, named entity recognition, code debugging, logical/deductive
reasoning, code generation), scored on:

1. **Accuracy gate** — LLM-judge must pass a threshold or the submission is
   excluded from the leaderboard entirely.
2. **Token efficiency** — among submissions that pass the gate, ranked ascending
   by total tokens recorded by the judging proxy. Fewer tokens = higher rank.

A secondary goal is qualifying for the $1,000 Google DeepMind "Best Use of
Gemma" prize by making Gemma 4 variants the default models for scored answers.

## 2. Hard constraints (from the official Track 1 rules)

- Container reads `/input/tasks.json` (`[{"task_id", "prompt"}]`), writes
  `/output/results.json` (`[{"task_id", "answer"}]`), exits 0 on success.
- Max runtime: 10 minutes total, for however many tasks are in the batch.
- Env vars injected at eval time, must not be hardcoded or bundled in a `.env`:
  `FIREWORKS_API_KEY`, `FIREWORKS_BASE_URL`, `ALLOWED_MODELS`.
- **All inference that produces a scored answer must go through Fireworks via
  `FIREWORKS_BASE_URL`.** Local models/tokens count as zero for the score —
  meaning local compute is free, but it also cannot be the source of a scored
  answer. It can only steer/support the required Fireworks call.
- Only models listed in `ALLOWED_MODELS` may be called; anything else
  invalidates the submission.
- No hardcoded or cached answers — eval uses unseen prompt variants.
- Malformed `/output/results.json` scores zero.
- Image compressed size ≤ 10GB. Submissions rate-limited to 10/hour/team.

`ALLOWED_MODELS` found for this track (subject to reconfirmation on launch day):
`minimax-m3`, `kimi-k2p7-code`, `gemma-4-31b-it`, `gemma-4-26b-a4b-it`,
`gemma-4-31b-it-nvfp4`.

## 3. Core strategy: "local steers, Fireworks answers"

Every task's scored answer comes from exactly one Fireworks call (plus, rarely,
one escalation call). The local Gemma model, served on the AMD Instinct MI300X
via vLLM, does zero-cost work around that call:

- Classifies the task into 1 of the 8 categories
- Selects the model, prompt template, and `max_tokens` budget for that category
- Runs a category-specific critic check on the Fireworks response
- Decides whether to escalate to a stronger model
- Repairs response formatting locally where possible, to avoid a retry call

This was chosen over two alternatives considered and rejected:
- **Try-local-first-fallback** (answer fully locally, escalate to Fireworks only
  on low confidence) — not viable, since local answers aren't recorded/scored
  and a task answered only by local inference would score zero for that task.
- **Local drafts, Fireworks rubber-stamps** (local generates the full answer,
  Fireworks is pinged with a minimal confirm-only prompt) — rejected as a
  plausible rules violation; risks being read as circumventing "all inference
  must go through Fireworks," and disqualification is worse than a lower
  token-efficiency rank.

## 4. Container flow

1. Entrypoint script (`app/main.py`) launches `vllm serve` for the local model
   as a background subprocess; polls `/v1/models` until healthy.
2. Reads `/input/tasks.json`.
3. Tasks are processed concurrently (bounded worker pool — see §7) rather than
   serially, since the 10-minute budget must cover network latency to Fireworks
   across however many tasks are in the batch.
4. Per task:
   a. Local Gemma classifies the task into one of the 8 categories.
   b. Category → tier lookup (§5) selects: Fireworks model, prompt template,
      `max_tokens` budget.
   c. Call Fireworks via `FIREWORKS_BASE_URL` with the selected
      model/prompt/budget. This produces the candidate scored answer.
   d. Category-specific local critic (§6) checks the response.
   e. If the critic flags failure/low confidence: one escalation call to
      `minimax-m3` (§6), with the failed attempt and critic's reason included
      in the escalation prompt for context. At most one escalation per task —
      if the escalation also fails the critic check, its answer is used
      anyway (best effort; no further escalation, to keep runtime bounded).
   f. Local formatting repair (e.g. malformed JSON/schema) is attempted before
      falling back to a retry call. At most one retry call per task.
5. Collect all `{task_id, answer}` pairs, write `/output/results.json`.
6. Shut down the local vLLM subprocess, exit 0.

## 5. Model routing table

| Tier | Categories | Default model | Rationale |
|---|---|---|---|
| Cheap | Sentiment classification, NER, factual knowledge | `gemma-4-31b-it-nvfp4` | Quantized variant of the dense Gemma 4, which is independently benchmarked as unusually token-efficient (fewer output tokens than rivals at comparable capability). Avoids the MoE variant's chain-of-thought verbosity. |
| Code | Code debugging, code generation | `kimi-k2p7-code` | Purpose-built for agentic coding; benchmarked at ~30% fewer reasoning tokens than its predecessor while scoring higher on coding benchmarks. |
| Heavy | Mathematical reasoning, logical/deductive reasoning, text summarization | `gemma-4-31b-it` (dense) | Benchmarked as both capable (beats larger models on math/coding) and token-efficient — a rare combination, and the reason it isn't reserved only for escalation. |
| Escalation (any tier, on critic failure) | — | `minimax-m3` | Highest benchmarked intelligence index of the five available models, including the best coding and GPQA (graduate-level knowledge) scores. Used only when the tier's default model is flagged uncertain, so it stays rare and doesn't dominate the token count. |

`gemma-4-26b-a4b-it` (MoE variant) is deliberately not used as a default
anywhere: its explicit chain-of-thought reasoning trades lower compute cost for
higher output-token count, which works against the token-efficiency score even
though it's cheap to serve.

Benchmark/pricing data behind this table was gathered via web search on
2026-07-06 (Artificial Analysis Intelligence Index, Fireworks blog posts and
docs) and is not yet confirmed against Fireworks' exact listing for these
specific model IDs — re-verify once `ALLOWED_MODELS` and API access are
confirmed on launch day, and adjust the table if actual behavior differs.

## 6. Local critic design (per category)

The critic's job is to decide, per task, whether the tier's default answer is
good enough to keep or should be escalated. The mechanism is category-shaped:
deterministic checks where the answer is objectively verifiable, softer
confidence/consistency checks where it isn't.

| Category | Critic mechanism |
|---|---|
| Code debugging / code generation | **Deterministic.** Locally parse/lint the returned code (and run it against any test cases present in the prompt, if any). Pass/fail. On failure, escalate with the parse/runtime error included in the prompt so the escalation model fixes a known bug rather than re-guessing. |
| Mathematical reasoning | **Deterministic.** Locally recompute the arithmetic/derivation independently and compare. Escalate on disagreement. |
| Logical/deductive reasoning | **Deterministic.** Plug the returned answer back into the stated constraints and verify all hold. Escalate on any violation. |
| Sentiment classification / NER | **Self-consistency.** Local Gemma independently re-derives the label/entities from the same prompt; escalate if it disagrees with the Fireworks answer. |
| Factual knowledge | **Confidence signal.** Local Gemma checks the Fireworks answer for hedging/uncertainty language, and separately flags whether the question is graduate-level/obscure (matches why `minimax-m3`, with the highest GPQA score, is the escalation target). Escalate on either signal. |
| Text summarization | **Constraint + faithfulness check.** Locally verify any stated length/format constraint is met, and spot-check that key facts extracted from the source text aren't missing from the summary. Escalate on either failure. |

## 7. Concurrency and runtime budget

The 10-minute limit applies to the whole batch, not per task, so tasks are
processed with a bounded concurrent worker pool (async HTTP calls to Fireworks)
rather than serially — local classification/critic steps are fast (local
GPU, no network), but Fireworks round-trips are the dominant latency and must
be parallelized to fit an unknown-sized task batch in the time limit. Worker
count is configurable and should be tuned against observed per-call latency
once real Fireworks access is available; start conservative and raise it once
timing data exists.

## 8. Error handling

- **Fireworks call failure (network/rate limit/5xx):** retry once with
  backoff; if it still fails, fall back to the local critic's best available
  answer for that task rather than leaving it out of `results.json` entirely
  (a missing task is likely scored as a hard failure; a wrong-but-present
  answer at least has a chance at the accuracy gate).
- **Malformed Fireworks response (not valid JSON / doesn't match expected
  answer shape):** attempt local repair (e.g., extract the answer from
  surrounding prose, reformat) before considering a retry call. At most one
  retry call per task, to bound token usage and runtime.
- **Local vLLM server fails to start:** container should exit non-zero
  immediately rather than attempting to run without the local layer, since the
  whole routing/critic design depends on it being available.
- **Overall runtime approaching the 10-minute limit:** any tasks not yet
  processed when a time budget checkpoint is hit are answered with a direct,
  un-escalated cheap-tier Fireworks call (skip local critic/escalation) rather
  than risking the whole container missing the deadline and scoring zero.

## 9. Testing strategy

- **Unit tests per critic type** using fixture tasks per category (known-buggy
  code snippets, constraint puzzles with a known valid answer, ambiguous
  sentiment examples) — verify each critic mechanism correctly flags
  pass/fail independent of any real Fireworks call (mock the Fireworks
  response).
- **Routing table tests** — verify each of the 8 categories maps to the
  expected tier/model, and that escalation always targets `minimax-m3`.
- **Container contract test** — a small `/input/tasks.json` fixture run
  through the full container, verifying `/output/results.json` is valid JSON
  in the expected shape and the process exits 0.
- **Integration test against real Fireworks** (using a personal/dev API key,
  not the harness-injected one) — run once real API access exists, to sanity
  check the routing table's token/accuracy assumptions from §5 before
  committing to it for submission.

## 10. Repo changes required

The current `Dockerfile`/README describe a long-running `vllm serve` HTTP
server as the container's whole job — that matched the "infra proof" milestone
but not the actual submission contract, which is a one-shot batch job. Needed
changes (implementation-plan level, not detailed here):

- `app/main.py` (or similar) becomes the container `CMD`, replacing the direct
  `vllm serve` entrypoint. It launches `vllm serve` itself as a subprocess.
- `requirements.txt` will need an async HTTP-capable client for the worker
  pool in §7 (the existing `httpx`/`openai` packages already support async
  usage — confirm before adding anything new).
- Local model choice: repo currently pins `google/gemma-2-2b-it` as the Step
  B placeholder (explicit prior decision to leave as-is). This design doesn't
  require changing that yet — the local model's job here (classification +
  critic checks) doesn't need frontier-scale capability, but should be
  revisited once the orchestrator is running and latency/quality can be
  measured.

## 11. Open risks / follow-ups

- Gemma 4 pricing/benchmark data (§5) was gathered from general web sources,
  not confirmed against Fireworks' exact listing for these specific IDs —
  re-verify on launch day.
- `gemma-4-31b-it-nvfp4`'s output-token behavior is assumed identical to the
  dense model (quantization typically affects speed/cost, not output length)
  — not independently confirmed.
- Worker pool concurrency (§7) needs real latency data to tune; starting value
  is a guess.
- GPU Request Form and GitHub MCP token setup are tracked separately (not part
  of this design) — see session notes in `claudelocal.md`.
