# Hybrid-Token-Efficient-Routing-Agent

ScoobaCats submission for the AMD Developer Hackathon ACT II, Track 1
(Hybrid Token-Efficient Routing Agent).

This repo currently contains the **local-inference container skeleton**: a
Docker image that boots a vLLM ROCm server on an AMD Instinct MI300X and
exposes an OpenAI-compatible endpoint on port 8000. Routing, confidence, and
escalation logic land in later tasks — this milestone is *green boot + one
successful request*.

## Pinned base image

`rocm/vllm:rocm7.13.0_gfx94X-dcgpu_ubuntu24.04_py3.13_pytorch_2.10.0_vllm_0.19.1`

- MI300X's arch is `gfx942`. AMD groups the datacenter gfx94x GPUs under the
  `gfx94X-dcgpu` tag family on Docker Hub, so this tag targets MI300X.
- Do **not** change to `:latest` — reproducibility on the scoring box is the
  whole point of pinning.

## Build

```sh
docker build -t scoobacats-agent:dev .
```

Building locally on a laptop (no GPU) will succeed — vLLM won't start without
ROCm devices, but the image itself builds fine.

## Run on an MI300X instance

Spin up a single MI300X on AMD Developer Cloud, then:

```sh
docker run -it --rm \
  --device /dev/kfd \
  --device /dev/dri \
  --network=host \
  --ipc=host \
  --group-add=video \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --env HF_TOKEN=$HF_TOKEN \
  -p 8000:8000 \
  scoobacats-agent:dev
```

The default `MODEL` is `Qwen/Qwen3-0.6B` (ungated, tiny — used to prove infra).
Override for Gemma (Step B) once you've accepted the license on the HF model
page and exported `HF_TOKEN`:

```sh
docker run ... --env MODEL=google/gemma-2-2b-it ... scoobacats-agent:dev
```

## Verify

From the MI300X instance, after the log shows the server is up:

```sh
# List models
curl http://localhost:8000/v1/models

# One chat completion
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "Say hello in 3 words."}],
    "max_tokens": 16
  }'
```

A valid JSON response = infra proven. Stop the instance immediately after
verification passes so idle GPU time isn't billed.

## Gotchas

- **Gated models 401**: unaccepted license or missing/wrong `HF_TOKEN`, not a
  container bug.
- **HIP graph hangs**: add `--enforce-eager` to the `vllm serve` command as a
  diagnostic.
- **dtype**: ROCm prefers `float16`. This is set explicitly in the Dockerfile
  CMD to avoid the bfloat16 auto-cast warning.
- **Do not reinstall torch/vllm/rocm in `requirements.txt`** — the base image
  already ships tuned builds; a pip reinstall pulls the CUDA wheel and breaks
  the container.
- **GPU sanity check**: inside the running container, `rocm-smi` should show
  the MI300X with ~192 GB HBM. If not, device passthrough flags are wrong.
