# AGENTS.md

## What this is

FastAPI server wrapping NVIDIA's LocateAnything-3B vision-language model behind an OpenAI-compatible API. Python 3.11+, managed by `uv`.

## Commands

```bash
make install     # uv sync (dependencies)
make download    # Pre-download model (~6GB) via huggingface-cli
make serve       # LOCATE_PORT=8282 uv run locateanything-api
```

## Key gotchas

- **Two entrypoints, one is dead.** `server.py` (root) is a legacy inline server — ignore it. `main.py` (root) is a placeholder. The real entrypoint is `src/locateanything_api/server.py` → `main()`, registered in pyproject.toml as `locateanything-api`.
- **Model load is lazy** unless `LOCATE_PRELOAD_MODEL=true`. First request triggers a slow load + download (runs off the event loop via `asyncio.to_thread` so the server stays responsive). Concurrent generation is bounded by `LOCATE_MAX_CONCURRENCY` (default 2) using an asyncio semaphore.
- **Port mismatch.** Makefile hardcodes `8282`; Dockerfile and config default to `8000`. Config wins when using the installed CLI directly.
- **No lockfile.** `uv.lock` is not committed. `make install` runs `uv sync` which may pull different compatible versions on different machines.
- **Config is all env vars** with `LOCATE_` prefix (e.g. `LOCATE_PORT`, `LOCATE_MODEL_ID`). See `src/locateanything_api/config.py` for the full list. Supports `.env` file.
- **`src/decord/`** is a stub only — macOS ARM64 video loading placeholder.

## Architecture (files)

```
src/locateanything_api/
  config.py      # Pydantic-settings (LOCATE_* env vars, lru_cached singleton)
  schemas.py     # Request/response models (ChatCompletion, Responses, Models)
  backend.py     # Model loading, prompt prep, generation, bbox parsing (async semaphore concurrency)
  app.py         # FastAPI app factory — routes, streaming, streaming helpers
  server.py      # Uvicorn runner entry point (calls create_app, main())

server.py (root) # DEPRECATED — inline legacy server, not used
main.py (root)   # DEPRECATED — placeholder
```

## Testing

No test suite exists. No CI. No lint/typecheck config.

## PLAN.md

Contains a future self-evolving agent system design. Not implemented — treat as aspirational spec, not current architecture.
