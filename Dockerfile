# syntax=docker/dockerfile:1

FROM python:3.10.11-slim AS builder

ENV POETRY_VIRTUALENVS_CREATE=false \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TORCH_CUDA_ARCH_LIST="" \
    CUDA_VISIBLE_DEVICES="" \
    FORCE_CPU="1"

WORKDIR /app

RUN pip install --upgrade pip uv

# Install CPU-only torch first to prevent CUDA variants
RUN pip install torch==2.2.2+cpu --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md ./
COPY utils ./utils
COPY controllers ./controllers
COPY routers ./routers
COPY main.py ./main.py
COPY tests ./tests

RUN uv pip install --system .

FROM python:3.10.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TORCH_CUDA_ARCH_LIST="" \
    CUDA_VISIBLE_DEVICES="" \
    FORCE_CPU="1"

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.10 /usr/local/lib/python3.10
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app /app

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
