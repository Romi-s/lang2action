# lang2action agent image (Linux: pybullet builds from source, so we need a compiler)
FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir -e .

COPY tests ./tests

ENV LANG2ACTION_PERCEPTION=sim
ENTRYPOINT ["lang2action"]
CMD ["--help"]
