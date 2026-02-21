# syntax=docker/dockerfile:1

FROM python:3.10.11-slim AS builder

ENV POETRY_VIRTUALENVS_CREATE=false \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TORCH_CUDA_ARCH_LIST="" \
    CUDA_VISIBLE_DEVICES="" \
    FORCE_CPU="1" \
    TRANSFORMERS_CACHE=/tmp/transformers_cache \
    HF_DATASETS_CACHE=/tmp/datasets_cache

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libsqlite3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip uv

COPY pyproject.toml README.md ./
COPY utils ./utils
COPY controllers ./controllers
COPY routers ./routers
COPY main.py ./main.py
COPY tests ./tests

# Install project dependencies from default Python index
RUN pip3 install --no-cache-dir -e .

# Force CPU-only torch wheel
RUN pip3 install --no-cache-dir --upgrade --force-reinstall torch --index-url https://download.pytorch.org/whl/cpu

FROM python:3.10.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TORCH_CUDA_ARCH_LIST="" \
    CUDA_VISIBLE_DEVICES="" \
    FORCE_CPU="1"

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates zstd \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://ollama.com/install.sh | sh

COPY --from=builder /usr/local/lib/python3.10 /usr/local/lib/python3.10
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app /app
COPY scripts/start-with-ollama.sh /app/scripts/start-with-ollama.sh

RUN chmod +x /app/scripts/start-with-ollama.sh

EXPOSE 8000 11434

CMD ["/app/scripts/start-with-ollama.sh"]
