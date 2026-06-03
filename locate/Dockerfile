# syntax=docker/dockerfile:1.7
FROM ghcr.io/astral-sh/uv:debian

ENV DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/data/huggingface \
    LOCATE_HOST=0.0.0.0 \
    LOCATE_PORT=8000 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_DOWNLOADS=automatic

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        libglib2.0-0 \
        libgomp1 \
        libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY src ./src

# Install from the standard PyPI index. On Linux, current PyTorch wheels pull
# their default CUDA user-space dependency set from PyPI; the host/container
# runtime still provides the NVIDIA kernel driver.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health >/dev/null || exit 1

CMD ["/opt/venv/bin/locateanything-api"]
