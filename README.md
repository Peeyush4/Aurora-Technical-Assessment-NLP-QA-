# Aurora Technical Assessment — NLP Q&A

This repository implements a Standard RAG (Retrieve→Generate) Q&A pipeline over message logs. It includes data ingestion, a local vector store (ChromaDB), retrieval tools, lightweight local LLM generator wrappers, and a FastAPI app exposing an `/ask` endpoint.

## Table of Contents
- [Key Components](#key-components)
- [Problem Statement](#problem-statement)
- [Quickstart](#quickstart)
	- [Prerequisites](#prerequisites)
	- [Install](#install)
	- [Build the vector store (one-time)](#build-the-vector-store-one-time)
	- [Run the API](#run-the-api)
- [Example Request](#example-request)
- [Code Structure and Details](#code-structure-and-details)
	- [Data ingestion (`ingest_data.py`)](#data-ingestion-ingest_datapy)
	- [Vector DB / Retriever (`core/db.py`)](#vector-db--retriever-coredbpy)
	- [Tools (`tools.py`)](#tools-toolspy)
	- [QA Orchestration (`qa_system.py`)](#qa-orchestration-qasystempy)
	- [Generator wrappers (`generators/`)](#generator-wrappers-generators)
- [Design Decisions and Rationale](#design-decisions-and-rationale)
- [Troubleshooting & Common Issues](#troubleshooting--common-issues)
- [Security & Privacy Notes](#security--privacy-notes)
- [Development & Testing](#development--testing)
- [Future Work](#future-work)

## Key Components
- **`chroma_db/`**: Persistent ChromaDB directory created by `ingest_data.py` that stores embeddings and metadata for messages.
- **`generators/`**: Lightweight LLM generator wrappers (local/in-process) used to call different local LLM backends (Ollama, HuggingFace, etc.).
- **`ingest_data.py`**: Script to build the vector store from source data (`data/response_1762800357568.json`), compute embeddings, and persist them to `chroma_db/`.
- **`main.py`**: FastAPI application that exposes the `/ask` endpoint and delegates question answering to the RAG QA system.
- **`qa_system.py`**: High-level orchestrator implementing the RAG flow and prompt composition used to call the generator with context assembled from retrieved documents.
- **`tools.py`**: Tool definitions used by the agent: name extraction (`find_user_names`), retrieval (`get_user_messages` / `search_messages`), and simple system statistics (`get_system_stats`).

## Problem Statement
This project was developed as part of the Aurora Technical Assessment for an NLP Q&A system. The formal problem statement is available at:

https://gist.github.com/ogurtsov/f65b5c1975d901979a30f00b2bf4c5df

## Data Source
The messages used by this project were collected into a single JSON file: `data/response_1762800357568.json`.

- Location: `data/response_1762800357568.json` (project root `data/` folder).
- Origin: these are aggregated message logs exported/collected by the author and stored as one combined file for ingestion and reproducible testing.
- Format: the file contains an array of message records (each record typically includes fields such as `timestamp`, `user_id`, `user_name`, and `message` text). The ingestion script `ingest_data.py` expects this structure and converts each record into a document with metadata before inserting it into ChromaDB.

If you need to rebuild the vector store from a different source, replace `data/response_1762800357568.json` with your own export (or split/transform it) and run:

```powershell
python ingest_data.py
```

API docs and testing:
- Once the FastAPI app is running, the interactive docs are available at `http://127.0.0.1:8000/docs`.
- The `/ask` endpoint in the docs is reachable at: `http://127.0.0.1:8000/docs#/default/ask_ask_post` — use this page to craft a request and test questions against the ingested data.

## Quickstart

### Prerequisites
- Python 3.10+ (project uses modern LangChain/LangGraph and local generator wrappers)
- `pip` and a virtual environment
- Optional GPU for local model acceleration (if using heavy local models)

### Install
Run inside a virtual environment. Example (PowerShell):

```powershell
& "C:/MY FILES/Peeyush-Personal/Coding/.venv/Scripts/Activate.ps1"
pip install -r requirements.txt
```

> Note: On other systems use your platform's venv activation command.

### Environment variables (`.env`)
Place a `.env` file in the project root (the same directory as this `README.md`) to store API keys and model configuration for generators. The project uses `python-dotenv` in `generators/litellm.py`, and individual generator wrappers read environment variables directly.

Common environment variables used by the included generators:

- `GOOGLE_API_KEY` — optional; required by `generators/gemini.py` to call Google Generative AI.
- `GOOGLE_MODEL` — optional; defaults to `gemini-2.5-flash` in `generators/gemini.py`.
- `HUGGINGFACE_API_KEY` or `HUGGINGFACE_HUB_TOKEN` — required when using private Hugging Face models or when litellm needs a key. `generators/litellm.py` prints a warning if this is missing for HF models.
- `OLLAMA_HOST` — optional, e.g. `http://localhost:11434` if you self-host Ollama; litellm or other wrappers may read this from the environment depending on your setup.
- `OLLAMA_API_KEY` — optional for Ollama cloud-hosted deployments if auth is required by your setup.
- `OPENAI_API_KEY` — optional if you configure litellm or other code paths to call OpenAI.

- `GENERATOR_MODEL` — optional. If present, this selects which generator backend the project should prefer (examples: `litellm`, `huggingface`, `gemini`, `ollama`). If omitted the code typically defaults to `litellm` or the generator configured in your runtime/startup logic.
- `LITELLM_MODEL_NAME` — required when `GENERATOR_MODEL` is set to `litellm` (or when you call `generators/litellm.py` directly). This is the model identifier passed to `litellm.completion()` and can point to local prefixes (for Ollama) or to a Hugging Face path. Example values: `ollama/mistral`, `huggingface/google/flan-t5-base`, `gpt-4o-mini`.

Note on quoting: `.env` parsers accept both `KEY=value` and `KEY="value"` forms; prefer `KEY=value` without quotes for portability.

Example `.env` (store in project root; do NOT commit this file):

```
GENERATOR_MODEL=litellm
LITELLM_MODEL_NAME=ollama/mistral

GOOGLE_API_KEY=sk-...your_google_api_key_here...
GOOGLE_MODEL=gemini-2.5-flash
HUGGINGFACE_API_KEY=hf_xxx_your_hf_key
HUGGINGFACE_HUB_TOKEN=hf_xxx_your_hf_token
OLLAMA_HOST=http://localhost:11434
OLLAMA_API_KEY=ollama_api_key_if_needed
OPENAI_API_KEY=sk-...your_openai_key...
```

Security and usage notes:

- Add `.env` to your `.gitignore` to avoid accidentally committing secrets:

```
# local environment variables
.env
```
- If you use PowerShell instead of a `.env` file, you can set variables for the current session with:

```powershell
$env:GOOGLE_API_KEY = 'sk-...'
```

- The `generators/litellm.py` module calls `load_dotenv()` so it will automatically load variables from `.env` when the Python process starts. Other generator wrappers read `os.getenv(...)` directly.



### Build the vector store (one-time)
For the vector store, this project uses ChromaDB to create a simple, file-based database. This keeps the take-home project self-contained. For production, run ChromaDB as a dedicated service and connect using the HTTP client.

This reads `data/response_1762800357568.json`, computes embeddings, and writes a persistent ChromaDB into `chroma_db/`.

Run:

```powershell
python ingest_data.py
```

### Run the API
Start the FastAPI app (example):

```powershell
uvicorn main:app --reload --port 8000
```

Open `http://localhost:8000/docs` to view and exercise the API.

Note: after you run the `uvicorn` server, open the interactive docs at `http://127.0.0.1:8000/docs` (Swagger UI) to try queries and test the `/ask` endpoint. Visiting `http://127.0.0.1:8000` (the root) returns a small status JSON (useful for health checks) — the interactive API and the question form live under `/docs`.

### If you use Ollama (local) models
If your `LITELLM_MODEL_NAME` points to an Ollama-style model (for example `ollama/mistral`) or you plan to call Ollama directly, make sure the Ollama service is running before starting the FastAPI app. In a development workflow you can run Ollama in a separate terminal and then start the API:

PowerShell (separate terminals):

```powershell
ollama serve
# in another terminal
uvicorn main:app --reload --port 8000
```

If you prefer to start Ollama from PowerShell and leave it running in the background you can use `Start-Process`:

```powershell
ollama serve
uvicorn main:app --reload --port 8000
```

Notes:
- Ensure the Ollama model referenced by `LITELLM_MODEL_NAME` is available locally (pull or install it with Ollama prior to serving) or that your `OLLAMA_HOST` points to a reachable Ollama instance.
- `generators/litellm.py` will attempt to call the model name you set in `LITELLM_MODEL_NAME`; if Ollama is not running you will see connection/auth errors from the litellm client.

Pulling Ollama models
---------------------
If the model you plan to serve is not installed locally, pull it with the Ollama CLI before starting `ollama serve`. Replace `<model>` with the model name (match the name used in `LITELLM_MODEL_NAME` where applicable):

```powershell
# Pull a model (example)
ollama pull <model>

# Examples:
ollama pull mistral
ollama pull ollama/mistral

# List installed models to verify
ollama list
```

- Pulling requires internet access and may take time depending on model size.
- If you use a namespaced identifier in `LITELLM_MODEL_NAME` (e.g. `ollama/mistral`), ensure the pull command uses the same identifier or the canonical name shown by `ollama list`.
- After pulling, start the server with `ollama serve` and then start the API.

## Example Request
POST to `/ask` with JSON payload. Example (PowerShell / curl):

```powershell
curl -X POST "http://localhost:8000/ask" -H "Content-Type: application/json" -d '{"question": "What is Thiago Monteiro's phone number?"}'
```

Example response (simplified):

```json
{
	"answer": "The phone number of Thiago Monteiro is 415-123-4567.",
	"evidence": [
		{ "source": "doc-1", "text": "Call me at 415-123-4567" }
	]
}
```

## Code Structure and Details

### Data ingestion (`ingest_data.py`)
- Loads `data/response_1762800357568.json` and converts items into documents with metadata (`user_name`, `user_id`, `timestamp`).
- Batches inserts into ChromaDB and computes embeddings using a SentenceTransformer model.

### Vector DB / Retriever (`core/db.py`)
- Uses `langchain_chroma.Chroma` with `HuggingFaceEmbeddings` (model: `all-MiniLM-L6-v2`).
- Builds a retriever with default `k=10` (returns top-k candidate documents for the LLM).

### Tools (`tools.py`)
- `find_user_names` — spaCy NER + fuzzy matching to find canonical user names.
- `search_messages` / `get_user_messages` — metadata-filtered retrieval by `user_name`.
- `get_system_stats` — returns simple diagnostics (number of users, messages, etc.).

### QA Orchestration (`qa_system.py`)
- `get_rag_information` performs retrieval and returns a context string used to prompt the generator.
- `answer_question` composes the final prompt and calls the generator to produce an answer; the code supports both RAG-based answering and a profile-file-based fallback.

### Generator wrappers (`generators/`)
- Abstraction layer exposing `generate()`/`invoke()` methods for different local LLM backends (Ollama, HuggingFace). This makes it easy to swap model backends.

## Design Decisions and Rationale
- ChromaDB (`PersistentClient`): chosen for zero-dependency local persistence and reproducibility.
- Retrieval + deterministic extraction: retrieval provides provenance; deterministic post-processing (regex + timestamp heuristics) is used for structured facts (phone numbers, emails). The LLM synthesizes natural-language answers from retrieved evidence.

## Troubleshooting & Common Issues
- If you see `FATAL: spaCy model 'en_core_web_sm' not found.` run:

```powershell
python -m pip install spacy
python -m spacy download en_core_web_sm
```

- If ChromaDB connection fails, ensure `ingest_data.py` was run and `chroma_db/` exists and is writable.

- If the `/ask` endpoint raises an internal error related to `profiles`, check `qa_system.py` — the `answer_question` flow conditionally uses `profiles` only when `using_rag=False`. Ensure `profiles` is defined before use or use the RAG flow.

## Security & Privacy Notes
- Do not expose raw credit-card-like tokens. Mask sensitive fields and require explicit authorization for high-sensitivity data.

## Development & Testing
- `data_analysis.ipynb` contains exploratory analysis, classification templates, and generator call examples.
- `test_agent.py` and variants exist to exercise the agent graph during development.

## Future Work
- Add deterministic extraction endpoints for structured data (phones, emails, preferences).
- Add authentication, rate limiting, and observability (LangSmith/OpenTelemetry) for production readiness.
- Integrate streaming / production ChromaDB and a monitoring platform for model-drift detection.

---

If you'd like, I can also:
- add a minimal example `curl`/Postman collection for `/ask`;
- add a short `CONTRIBUTING.md` with local dev steps;
- or apply this README text as the repository README file (already applied).

## Bonus: Design Notes & Architecture

After extensive data analysis, we identified that this project's core challenge is the "dirty" and stateful nature of the 3,349-message dataset. A successful system must handle complex reasoning, contradictions, and ambiguous language. Below are the architectures we considered, why they failed in testing, and the final approach we selected.


**IMPLEMENTED: Standard RAG (Selected for this repository)**

For this codebase and the deliverable submitted, I implemented the Standard RAG pipeline (Retrieve → Generate). This was chosen for the assessment submission to provide a clear, reproducible baseline implementation within the time constraints.

What the repository implements:
- A persistent ChromaDB vector store built from `data/response_1762800357568.json` using `ingest_data.py`.
- A retriever + prompt composition flow in `qa_system.py` that fetches top-K documents and asks a generator to synthesize an answer.
- A FastAPI wrapper (`main.py`) exposing `/ask` that delegates to the RAG flow.

Known limitations of this implemented approach:
- Aggregate questions that require scanning dozens of messages may miss items when `k` is small.
- Reasoning across semantically different messages is brittle; the generator often needs richer, multi-part context to connect events.
- For structured facts (phone numbers, counts), the model may hedge if retrieved context is noisy; deterministic extraction or a deterministic fallback would improve reliability.

Why this was chosen for the submission:
- Simplicity: provides a working, reproducible baseline within the time and compute constraints.
- Transparency: easy to inspect retrieved evidence and reproduction of answers.

The following sections document other architectures considered for completeness.

### Path 2 — Offline Profile Agent (The "Brittle" Plan) — REJECTED

This plan attempted to run an offline, stateful agent (`build_profiles.py`) to summarize all 3,349 messages into canonical JSON profiles.

Why it was REJECTED:
- Hallucinations: LLM-run batch processing invented values (fake phone numbers) and mis-classified fields.
- Mis-reasoning: failed to connect related events and polluted global preferences with one-off requests.
- Too brittle and costly: required long, expensive LLM runs and produced unreliable outputs.

Note: the repository contains a small remnant of this offline/profile approach. The file `profile_builder.py` (an offline script used during experiments) and the `profiles/` folder (sample generated profile text files) remain in the project for reference only. They are artifacts of the brittle plan and are not used by the main RAG pipeline; the profile outputs in `profiles/` may contain noisy or partially-generated data.

### Path 3 — Online Agentic RAG (The "Slow & Stupid" Plan) — REJECTED

This approach implemented a real-time agent that loops between tools (LangGraph + toolbox) to plan and call retrievals.

Why it was REJECTED:
- Poor real-time planning with local 7B/8B models (Mistral, Llama3.1) — the agent failed to reason consistently.
- "Thinking out loud": the agent often described its plan rather than invoking tools, led to loops and hallucinated non-existent tools.
- Too slow: per-query latency of 20–30 seconds was unacceptable for a small API.

### The 10x Solution — Hybrid Multi-RAG Pipeline - REJECTED

This architecture is an alternative approach (not implemented here) that would shift the hard work to engineered Python code and lightweight offline processing so online queries are fast (2–5s) and reliable.

Offline (The "Smart Sorter"):
- (optional, not included) Run a one-time classifier to assign categories (profile_update, preference, event_request, feedback, noise). Note: the example classifier script `classify_messages.py` is a proposed component in the design notes but is NOT included in this repository; implementing it is listed under Future Work.
- Ingest messages into targeted ChromaDB collections (examples): `db_facts` (preferences/profile updates), `db_events` (requests/complaints), and `db_feedback` (thank-you/feedback).

Online (The "Smart Pipeline"):
- `main.py` + `qa_system.py` implement deterministic, engineer-written logic (not a planning agent).
- Use spaCy + fuzzy name matching to identify the target user, then run a small intent classifier or keyword detector on the question to pick a reasoning plan.
- Execute targeted RAG queries (e.g., `k=20` for facts queries) against the appropriate DB(s) and assemble a multi-part context.
- Finally, call a compact, low-latency generator (Mistral) with the curated context so the model only needs to surface and synthesize—no heavy planning.

Why this works:
- Fast: offline sorting reduces online search space and keeps API latency low.
- Accurate: targeted retrievals and engineered plans capture widely dispersed clues and connect them deterministically.
- Robust: Python code controls logic; LLMs are used only for controlled synthesis, reducing hallucinations.

Why this was not implemented in this submission:
- The offline classification stage proved time-consuming in practice — each classification run took on the order of ~10s, so processing the full dataset would take many hours. Given the time and compute constraints for this assessment, I chose to implement the Standard RAG (Retrieve→Generate) baseline instead. The Hybrid Multi-RAG design remains documented here as a recommended future direction if you have the resources to run the offline preprocessing.

---

Reference: the original assessment and API spec are available at the challenge link: https://gist.github.com/ogurtsov/f65b5c1975d901979a30f00b2bf4c5df