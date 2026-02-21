#!/usr/bin/env sh
set -eu

OLLAMA_PID=""

cleanup() {
  if [ -n "$OLLAMA_PID" ]; then
    kill "$OLLAMA_PID" >/dev/null 2>&1 || true
  fi
}

trap cleanup EXIT INT TERM

LLM_PROVIDER="${CHATBOT_LLM_PROVIDER:-placeholder}"
LLM_MODEL="${CHATBOT_LLM_MODEL:-llama3.2}"
OLLAMA_BASE_URL="${CHATBOT_OLLAMA_BASE_URL:-http://127.0.0.1:11434}"

if [ "$LLM_PROVIDER" = "ollama" ]; then
  echo "Starting Ollama server..."
  ollama serve >/tmp/ollama.log 2>&1 &
  OLLAMA_PID=$!

  echo "Waiting for Ollama to become ready..."
  READY=0
  for i in $(seq 1 30); do
    if curl -fsS "${OLLAMA_BASE_URL%/}/api/tags" >/dev/null 2>&1; then
      READY=1
      break
    fi
    sleep 1
  done

  if [ "$READY" -eq 1 ]; then
    echo "Pulling model: $LLM_MODEL"
    ollama pull "$LLM_MODEL"
  else
    echo "Warning: Ollama did not become ready in time; continuing without model pull."
  fi
fi

exec uvicorn main:app --host 0.0.0.0 --port 8000
