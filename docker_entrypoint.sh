#!/bin/bash
set -e

# Working dir is /app in the Dockerfile
cd /app

# If chroma_db is not mounted or empty, run ingest_data.py to populate it
if [ ! -d "./chroma_db" ] || [ -z "$(ls -A ./chroma_db 2>/dev/null || true)" ]; then
  echo "Chroma DB missing or empty — running ingest_data.py"
  # Run ingestion; if it fails, print but continue to attempt to start the app
  if python ingest_data.py; then
    echo "Ingest completed."
  else
    echo "ingest_data.py failed; continuing to start the service (may fail)."
  fi
else
  echo "Chroma DB present — skipping ingest"
fi

# Optionally start Ollama server in background if available and requested.
# Set OLLAMA_AUTO_SERVE=true in your .env to enable this.
# Optional pre-pull of Ollama models before serving.
# Set OLLAMA_PRELOAD_MODELS to a comma-separated list, e.g.:
#   OLLAMA_PRELOAD_MODELS=ollama/llama3.1:8b,mistral
# Optionally set OLLAMA_PRELOAD_TIMEOUT (seconds) per model (default: 600).
if [ -n "${OLLAMA_PRELOAD_MODELS:-}" ]; then
  if command -v ollama >/dev/null 2>&1; then
    echo "OLLAMA_PRELOAD_MODELS is set. Will attempt to pull: ${OLLAMA_PRELOAD_MODELS}"
    IFS=',' read -ra _MODELS <<< "${OLLAMA_PRELOAD_MODELS}"
    for m in "${_MODELS[@]}"; do
      # trim spaces
      model=$(echo "$m" | sed 's/^ *//;s/ *$//')
      if [ -z "$model" ]; then
        continue
      fi
      echo "Attempting to pull Ollama model: $model"
      # Use timeout if available, otherwise run normally. Failures are logged but do not abort container start.
      if command -v timeout >/dev/null 2>&1; then
        if timeout ${OLLAMA_PRELOAD_TIMEOUT:-600} ollama pull "$model" > /var/log/ollama-pull.log 2>&1; then
          echo "Pulled $model successfully."
        else
          echo "Warning: failed to pull $model (see /var/log/ollama-pull.log). Continuing." >&2
        fi
      else
        if ollama pull "$model" > /var/log/ollama-pull.log 2>&1; then
          echo "Pulled $model successfully."
        else
          echo "Warning: failed to pull $model (see /var/log/ollama-pull.log). Continuing." >&2
        fi
      fi
    done
  else
    echo "OLLAMA_PRELOAD_MODELS set but 'ollama' CLI not found; skipping pre-pull." >&2
  fi
fi

# Optionally start Ollama server in background if available and requested.
# Set OLLAMA_AUTO_SERVE=true in your .env to enable this.
if [ "${OLLAMA_AUTO_SERVE:-""}" = "true" ]; then
  if command -v ollama >/dev/null 2>&1; then
    echo "Starting ollama serve in background..."
    # Start ollama in background; write logs to /var/log/ollama.log
    ollama serve > /var/log/ollama.log 2>&1 &
    sleep 2
    echo "Ollama serve started (logs: /var/log/ollama.log)"
  else
    echo "OLLAMA_AUTO_SERVE requested but 'ollama' CLI not found; skipping." >&2
  fi
fi

# Finally exec uvicorn so signals propagate correctly
exec uvicorn main:app --host 0.0.0.0 --port 8000
