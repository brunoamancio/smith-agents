# This Dockerfile will be replaced by "Dockerfile-bkp" once zed-industries/codex-acp supports
# session persistence (PR: https://github.com/zed-industries/codex-acp/pull/66)

FROM rust:1.81-alpine AS codex_builder

ARG CODEX_ACP_REPO=https://github.com/brunoamancio/codex-acp.git
ARG CODEX_ACP_REF=main

RUN apk add --no-cache git pkgconf openssl-dev openssl-libs-static build-base perl

ENV OPENSSL_NO_VENDOR=1

WORKDIR /build
RUN git clone --depth 1 --branch "${CODEX_ACP_REF}" "${CODEX_ACP_REPO}" . \
    && cargo build --release --bin codex-acp

FROM python:3.14-alpine

ARG CODEX_ACP_VERSION=0.4.0
ARG CODEX_ACP_TARGET=x86_64-unknown-linux-gnu
ARG CODEX_ACP_SHA256=ca531fce73e9c715599b3aea707713bc9883ea78d098384b6fdd63c46c5ea5d5

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CODEX_ACP_VERSION=${CODEX_ACP_VERSION}

WORKDIR /app

RUN apk add --no-cache curl ca-certificates jq xxd bash git

RUN python -m pip install --upgrade pip==25.3

COPY --from=codex_builder /build/target/release/codex-acp /usr/local/bin/codex-acp
RUN chmod +x /usr/local/bin/codex-acp && mkdir -p /workspace

COPY agents/acp-agents/codex/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY agents/acp-agents/codex/app /app/app
COPY agents/acp_agents_common /app/acp_agents_common

ENV CODEX_ACP_BIN=/usr/local/bin/codex-acp \
    CODEX_WORKDIR=/workspace \
    CODEX_SESSION_PERSIST=1

EXPOSE 8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
