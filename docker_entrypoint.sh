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
