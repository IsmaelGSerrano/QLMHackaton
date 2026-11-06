# Small inference image: XGBoost (local) vs the trained circuit (local simulators, garnet:mock, IQM Garnet).
# Build from the repo root (the image needs a few modules of src/qlmaml):
#   uv run python inference/export_bundle.py
#   docker build -f inference/Dockerfile -t qlmaml-inference .
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.11.10 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 \
    PATH="/app/.venv/bin:$PATH" PYTHONPATH=/app

WORKDIR /app
COPY inference/pyproject.toml inference/uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-install-project

# Only the modules the app imports (circuit, features, Garnet helpers, backend picker).
COPY src/qlmaml/__init__.py src/qlmaml/circuit.py src/qlmaml/features.py src/qlmaml/garnet.py \
     src/qlmaml/hardware.py src/qlmaml/qpu_pick.py qlmaml/
COPY inference/app.py ./
COPY inference/static/ static/
COPY inference/bundle/ bundle/

RUN useradd --create-home app
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=60s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz')"
ENTRYPOINT ["python", "app.py"]
CMD ["serve", "--port", "8000"]
