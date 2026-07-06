# AMD prebuilt ROCm + vLLM base for MI300X (gfx942).
# The rocm/vllm image family names datacenter gfx94x GPUs as "gfx94X-dcgpu".
# Pin the exact tag — never :latest — so the scoring environment reproduces.
FROM rocm/vllm:rocm7.13.0_gfx94X-dcgpu_ubuntu24.04_py3.13_pytorch_2.10.0_vllm_0.19.1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

# Placeholder ungated model for Step A (infra proof).
# Override at `docker run` with --env MODEL=... to swap in Gemma (Step B).
ENV MODEL=Qwen/Qwen3-0.6B

EXPOSE 8000

# Use shell form so $MODEL is expanded at container start.
CMD vllm serve $MODEL --dtype float16 --host 0.0.0.0 --port 8000
