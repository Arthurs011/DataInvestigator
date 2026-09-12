# DataInvestigator — reproducible container
#
#   docker build -t data-investigator .
#   docker run --rm -v "$PWD/reports:/reports" \
#     -v "$HOME/.env:/run/secrets/env" \
#     data-investigator --data /data

# Pin the toolchain for byte-for-byte reproducibility.
FROM ghcr.io/astral-sh/uv:0.8.15 AS base
WORKDIR /app

# Install non-interactive matplotlib/system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 fonts-dejavu-core ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./
COPY src ./src
COPY scripts ./scripts
RUN uv sync --no-dev --frozen

ENV INVESTIGATOR_MODEL=openai/gpt-4o-mini \
    PYTHONUNBUFFERED=1

# Mount the dataset and API key at runtime.
VOLUME ["/data", "/reports"]
ENTRYPOINT ["uv", "run", "investigator"]
CMD ["--data", "/data", "--out", "/reports"]