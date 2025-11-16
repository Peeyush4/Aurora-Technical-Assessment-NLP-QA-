FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install minimal system deps required by some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
	build-essential \
	git \
	wget \
	curl \
	&& rm -rf /var/lib/apt/lists/*

# Make 'python' point to python3
RUN ln -s /usr/bin/python3 /usr/bin/python

# Copy and install Python requirements (use cache when possible)
COPY requirements.txt .
RUN python -m pip install --upgrade pip --break-system-packages || true

# Try to install pinned requirements; if a CUDA-specific torch wheel fails,
# fall back to installing a generic CPU torch and re-install remaining deps.
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt || python -m pip install --no-cache-dir --break-system-packages torch

# Optionally install the Ollama CLI so the container can serve local Ollama models.
# NOTE: this runs the official Ollama installer script at build time. If you prefer
# to run Ollama on the host instead (recommended for production), skip this step.
RUN set -eux; \
    if curl -fsSL https://ollama.com/install.sh -o /tmp/ollama-install.sh; then \
        /bin/sh /tmp/ollama-install.sh || true; \
    else \
        echo "Could not download Ollama installer; skipping ollama install"; \
    fi


# Install spaCy English model package so it's always available at runtime
# Installing as a package ensures `spacy.load("en_core_web_sm")` succeeds.
RUN python -m spacy download en_core_web_sm --break-system-packages

# Copy application code (do NOT bake large data or DB into the image; mount at runtime)
COPY . .

# Recommended: mount ./chroma_db and ./data as volumes when running the container

EXPOSE 8000

COPY docker_entrypoint.sh /usr/local/bin/docker_entrypoint.sh
RUN chmod +x /usr/local/bin/docker_entrypoint.sh

ENTRYPOINT ["/usr/local/bin/docker_entrypoint.sh"]