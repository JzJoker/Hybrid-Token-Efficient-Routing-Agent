# Track 1: judging VM is linux/amd64. Build with:
#   docker buildx build --platform linux/amd64 --tag <img>:<tag> --push .
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

# Batch-mode entrypoint: read /input/tasks.json, write /output/results.json, exit 0.
# No server, no ports. The harness runs the container to completion.
ENTRYPOINT ["python", "-m", "app.main"]
