# Smith Codex ACP

FastAPI service wrapping the Codex Agent Control Plane (ACP) with shared Smith agent utilities.

## Published Tags

<!-- TAGS-START -->
- `0.3.15`
- `0.3`
- `latest`
<!-- TAGS-END -->

## Usage

```bash
docker pull th3b0yr0x/smith-codex-acp:latest
docker run --rm -p 8080:8080 \
  -v /path/to/workdir:/workspace \
  -v /path/to/codex/config:/app/codex-config \
  -e LOG_LEVEL=INFO -e CODEX_HOME=/app/codex-config \
  th3b0yr0x/smith-codex-acp:latest
```

`/path/to/workdir` should point to the project files you want Codex to read and modify; they will be mounted inside the container at `/workspace`.

`/path/to/codex/config` should reference your local Codex configuration directory (the folder that contains `config.toml`) so the container sees it at `/app/codex-config`.

## Configuration

- `CODEX_ACP_BIN` default `/usr/local/bin/codex-acp`
- `CODEX_WORKDIR` default `/workspace`
- `CODEX_SESSION_PERSIST` default `1`
- `CODEX_HOME` default `/app/codex-config`
- `LOG_LEVEL` default `INFO`

## Source

This image is built from `acp-agents/codex/Dockerfile`. See the repository for build scripts and release automation.

