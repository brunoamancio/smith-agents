# Smith Agents

This repository hosts the dockerized "echo" and "codex" agent services used by
the Smith IDE plugin. Clone it on its own (or consume it via the Smith repo) to
run the agents locally with Docker Compose.

## Prerequisites

- Docker Engine and Docker Compose plugin
- Python 3.12+ (only if you plan to modify the agent code)

## Quick Start

```powershell
cd agents
cp .env.example .env
# edit .env to point SMITH_WORKDIR to a folder you want agents to access
docker compose up codex-agent
```

The `.env` file defines `SMITH_WORKDIR`, which maps into the container at
`/workspace`. Anything inside that path is what the agent can read and modify.
By default it falls back to `./workspace` inside the repo if you omit the env.

To run the echo agent instead:

```powershell
docker compose up echo-agent
```

Both services expose HTTP ports matching the compose file (`8001` for echo and
`8002` for codex).

## Volume Layout

- `${SMITH_WORKDIR:-./workspace}:/workspace`: host project files visible to
  agents.
- `./codex/config:/app/codex-config`: codex-specific configuration files (API
  keys, session cache, etc.).

Adjust the `.env` file or compose volumes if you want to mount a different
directory or provide alternate config.
