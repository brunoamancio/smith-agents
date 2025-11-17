FROM python:3.12-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH="/root/.local/bin:${PATH}"

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && uv pip install --system acp-sdk "fastapi==0.115.0" "uvicorn==0.30.6"

COPY acp-agents/echo/agent.py /app/agent.py
COPY acp_agents_common /app/acp_agents_common

EXPOSE 8080

CMD ["python", "agent.py"]
