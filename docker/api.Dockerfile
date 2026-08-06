# API image. CPU-only torch by default — a deployment server rarely has a GPU, and
# a single verification is fast enough on CPU. For a central deployment serving
# many sites, swap the index URL for the CUDA build.
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# libgl/libglib are needed by OpenCV even in the headless build.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision \
    && pip install -r requirements.txt

COPY ml ./ml
COPY api ./api
COPY pyproject.toml ./

# Run as a non-root user: this process handles biometric data.
RUN useradd --create-home --uid 10001 sigver \
    && mkdir -p /app/data /app/artifacts \
    && chown -R sigver:sigver /app
USER sigver

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS http://localhost:8000/api/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
