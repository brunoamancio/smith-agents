from __future__ import annotations

import json
import logging
import os
import sys
import uuid
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from acp_agents_common import discover_services_root


SERVICES_ROOT = discover_services_root(Path(__file__).resolve().parent)
if str(SERVICES_ROOT) not in sys.path:
    sys.path.append(str(SERVICES_ROOT))

from acp_agents_common.attachments import (
    ATTACHMENT_METADATA_INLINE_LIMIT_KEY,
    ATTACHMENT_METADATA_KEY,
    Attachment,
    AttachmentError,
    DEFAULT_MAX_INLINE_BYTES,
    PROMPT_CAPABILITIES_KEY,
    PROMPT_CAPABILITY_EMBEDDED_CONTEXT,
    PROMPT_CAPABILITY_IMAGE,
    normalize_attachments,
)
from acp_agents_common.mode_utils import (
    AGENT_NAME_KEY,
    CLIENT_SESSION_ID_KEY,
    MODELS_BLOCK_KEY,
    MODE_ID_KEY,
    MODEL_ID_KEY,
    RUN_EVENTS_KEY,
    RUN_ID_KEY,
    RUN_OUTPUT_KEY,
    RUN_STATUS_KEY,
    RUN_STOP_REASON_KEY,
    SESSION_ID_KEY,
    MODES_BLOCK_KEY,
)

from .codex_bridge import CODEX_AGENT_NAME, CodexError, CodexManager, CodexProcessClosed


def _configure_logging() -> logging.Logger:
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logger = logging.getLogger("codex.api")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


logger = _configure_logging()


def _format_detail(value: Any) -> str:
    text = str(value)
    if len(text) > 160:
        return f"{text[:157]}..."
    return text


def _log(level: str, message: str, **details: Any) -> None:
    entry = message
    if details:
        formatted = ", ".join(f"{key}={_format_detail(value)}" for key, value in details.items())
        entry = f"{message} ({formatted})"
    getattr(logger, level)(entry)


def _default_binary() -> Path:
    return Path(os.getenv("CODEX_ACP_BIN", "/usr/local/bin/codex-acp"))


DEFAULT_WORKDIR_ENV = "CODEX_WORKDIR"
DEFAULT_CONTAINER_WORKSPACE = "/workspace"
_ROOT_PATH = Path("/")


def _default_workspace() -> Path:
    env_override = os.getenv(DEFAULT_WORKDIR_ENV)
    if env_override:
        return Path(env_override)

    repo_root = SERVICES_ROOT.parent
    if repo_root.exists() and repo_root != _ROOT_PATH:
        return repo_root

    return Path(DEFAULT_CONTAINER_WORKSPACE)


ATTACHMENT_MAX_INLINE_BYTES = int(os.getenv("CODEX_ATTACHMENT_MAX_INLINE_BYTES", DEFAULT_MAX_INLINE_BYTES))


def _build_attachment_metadata(prompt_caps: Dict[str, bool]) -> Dict[str, Any]:
    return {
        PROMPT_CAPABILITY_IMAGE: bool(prompt_caps.get(PROMPT_CAPABILITY_IMAGE)),
        PROMPT_CAPABILITY_EMBEDDED_CONTEXT: bool(
            prompt_caps.get(PROMPT_CAPABILITY_EMBEDDED_CONTEXT)
        ),
        "resourceLink": True,
        ATTACHMENT_METADATA_INLINE_LIMIT_KEY: ATTACHMENT_MAX_INLINE_BYTES,
    }


manager = CodexManager(binary_path=_default_binary(), workspace=_default_workspace())
app = FastAPI(title="Smith Codex Bridge", version="0.1.0")


def _build_session_response(result: Dict[str, Any]) -> Dict[str, Any]:
    prompt_caps = result.get(PROMPT_CAPABILITIES_KEY) or manager.prompt_capabilities
    session_payload = {
        SESSION_ID_KEY: result.get(SESSION_ID_KEY),
        MODELS_BLOCK_KEY: result.get(MODELS_BLOCK_KEY),
        MODES_BLOCK_KEY: result.get(MODES_BLOCK_KEY),
        PROMPT_CAPABILITIES_KEY: dict(prompt_caps),
        "metadata": {ATTACHMENT_METADATA_KEY: _build_attachment_metadata(prompt_caps)},
    }
    return {"session": session_payload}


class RunRequest(BaseModel):
    agent_name: str
    mode: str
    session_id: Optional[str] = None
    model_id: Optional[str] = None
    input: list[Dict[str, Any]]
    attachments: List[Attachment] = Field(default_factory=list)

    class Config:
        populate_by_name = True


class SessionNewRequest(BaseModel):
    agent_name: str
    client_session_id: str
    mcpServers: Optional[List[Dict[str, Any]]] = None

    class Config:
        populate_by_name = True


class SessionLoadRequest(BaseModel):
    agent_name: str
    session_id: str
    client_session_id: Optional[str] = None
    mcpServers: Optional[List[Dict[str, Any]]] = None

    class Config:
        populate_by_name = True


class SessionModelRequest(BaseModel):
    agent_name: str
    session_id: str
    model_id: str
    client_session_id: Optional[str] = None

    class Config:
        populate_by_name = True


class SessionModeRequest(BaseModel):
    agent_name: str
    session_id: str
    mode_id: str
    client_session_id: Optional[str] = None

    class Config:
        populate_by_name = True


class SessionCancelRequest(BaseModel):
    agent_name: str
    session_id: str
    client_session_id: Optional[str] = None

    class Config:
        populate_by_name = True


class SessionPruneRequest(BaseModel):
    keep_codex_session_ids: List[str] = Field(default_factory=list)


class SessionDeleteRequest(BaseModel):
    codex_session_ids: List[str] = Field(default_factory=list)


def _bearer_token(header: Optional[str]) -> Optional[str]:
    if not header:
        return None
    if header.lower().startswith("bearer "):
        token = header[7:].strip()
        return token or None
    return header.strip() or None


@app.on_event("shutdown")
async def _shutdown_event() -> None:
    await manager.stop()


@app.on_event("startup")
async def _startup_event() -> None:
    _log("info", "Agent started")


@app.get("/ping")
async def ping() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/agents")
async def list_agents() -> Dict[str, Any]:
    agent_capabilities = manager.prompt_capabilities
    return {
        "agents": [
            {
                "name": CODEX_AGENT_NAME,
                "description": "Codex ACP agent",
                "metadata": {
                    "provider": "openai",
                    "sdk": "codex-acp",
                    ATTACHMENT_METADATA_KEY: _build_attachment_metadata(agent_capabilities),
                },
            }
        ]
    }


@app.post("/session/new")
async def create_session(
    payload: SessionNewRequest,
    authorization: Optional[str] = Header(default=None, alias="Authorization"),
) -> Dict[str, Any]:
    _log(
        "debug",
        "Incoming /session/new request",
        agent=payload.agent_name,
        client_session_id=payload.client_session_id,
        has_token=bool(authorization),
    )
    if payload.agent_name.lower() != CODEX_AGENT_NAME:
        raise HTTPException(
            status_code=404,
            detail={"message": f"Unknown agent '{payload.agent_name}'"},
        )
    token = _bearer_token(authorization)
    try:
        result = await manager.create_session_with_models(
            token,
            payload.agent_name,
            payload.client_session_id,
            payload.mcpServers or [],
        )
    except CodexError as exc:
        _log("error", "Codex session/new failed", agent=payload.agent_name, error=exc.to_dict())
        status = 401 if exc.code == -32000 else 502
        raise HTTPException(status_code=status, detail=exc.to_dict()) from exc
    except CodexProcessClosed as exc:
        raise HTTPException(status_code=502, detail={"message": str(exc)}) from exc
    return _build_session_response(result)


@app.post("/sessions/prune")
async def prune_sessions(
    payload: SessionPruneRequest,
    authorization: Optional[str] = Header(default=None, alias="Authorization"),
) -> JSONResponse:
    _log(
        "info",
        "Pruning Codex sessions",
        keep_ids=len(payload.keep_codex_session_ids),
        keep_session_ids=payload.keep_codex_session_ids,
    )

    try:
        result = await manager.prune_sessions(set(payload.keep_codex_session_ids))
    except CodexError as exc:
        raise HTTPException(status_code=502, detail=exc.to_dict()) from exc
    except CodexProcessClosed as exc:
        raise HTTPException(status_code=502, detail={"message": str(exc)}) from exc
    except Exception as exc:  # pragma: no cover - defensive
        _log(
            "error",
            "Unexpected error while pruning sessions",
            error=str(exc),
        )
        raise HTTPException(status_code=500, detail={"message": str(exc)}) from exc

    return JSONResponse({"result": result})


@app.post("/sessions/delete")
async def delete_sessions(
    payload: SessionDeleteRequest,
    authorization: Optional[str] = Header(default=None, alias="Authorization"),
) -> JSONResponse:
    _log(
        "info",
        "Deleting Codex sessions",
        target_ids=len(payload.codex_session_ids),
        codex_session_ids=payload.codex_session_ids,
    )

    _bearer_token(authorization)  # token not required, but process must be ready for future calls
    try:
        result = await manager.delete_sessions(set(payload.codex_session_ids))
    except CodexError as exc:
        raise HTTPException(status_code=502, detail=exc.to_dict()) from exc
    except CodexProcessClosed as exc:
        raise HTTPException(status_code=502, detail={"message": str(exc)}) from exc
    except Exception as exc:  # pragma: no cover - defensive
        _log(
            "error",
            "Unexpected error while deleting sessions",
            error=str(exc),
        )
        raise HTTPException(status_code=500, detail={"message": str(exc)}) from exc

    return JSONResponse({"result": result})


@app.post("/session/load")
async def load_session(
    payload: SessionLoadRequest,
    authorization: Optional[str] = Header(default=None, alias="Authorization"),
) -> Dict[str, Any]:
    _log(
        "debug",
        "Incoming /session/load request",
        agent=payload.agent_name,
        session_id=payload.session_id,
        client_session_id=payload.client_session_id,
        has_token=bool(authorization),
    )
    if payload.agent_name.lower() != CODEX_AGENT_NAME:
        raise HTTPException(
            status_code=404,
            detail={"message": f"Unknown agent '{payload.agent_name}'"},
        )
    token = _bearer_token(authorization)
    try:
        result = await manager.load_session_with_models(
            token,
            payload.agent_name,
            payload.session_id,
            payload.client_session_id,
            payload.mcpServers,
        )
    except CodexError as exc:
        _log("error", "Codex session/load failed", session_id=payload.session_id, error=exc.to_dict())
        status = 401 if exc.code == -32000 else 502
        raise HTTPException(status_code=status, detail=exc.to_dict()) from exc
    except CodexProcessClosed as exc:
        raise HTTPException(status_code=502, detail={"message": str(exc)}) from exc
    return _build_session_response(result)


@app.post("/session/model")
async def update_session_model(
    payload: SessionModelRequest,
    authorization: Optional[str] = Header(default=None, alias="Authorization"),
) -> Dict[str, Any]:
    _log(
        "debug",
        "Incoming /session/model request",
        agent=payload.agent_name,
        session_id=payload.session_id,
        model_id=payload.model_id,
        client_session_id=payload.client_session_id,
        has_token=bool(authorization),
    )
    if payload.agent_name.lower() != CODEX_AGENT_NAME:
        raise HTTPException(
            status_code=404,
            detail={"message": f"Unknown agent '{payload.agent_name}'"},
        )
    token = _bearer_token(authorization)
    try:
        result = await manager.set_session_model_for_agent(
            token,
            payload.agent_name,
            payload.session_id,
            payload.model_id,
            payload.client_session_id,
        )
    except CodexError as exc:
        _log(
            "error",
            "Codex session/model failed",
            session_id=payload.session_id,
            model_id=payload.model_id,
            error=exc.to_dict(),
        )
        status = 401 if exc.code == -32000 else 502
        raise HTTPException(status_code=status, detail=exc.to_dict()) from exc
    except CodexProcessClosed as exc:
        raise HTTPException(status_code=502, detail={"message": str(exc)}) from exc
    return _build_session_response(result)


@app.post("/session/mode")
async def update_session_mode(
    payload: SessionModeRequest,
    authorization: Optional[str] = Header(default=None, alias="Authorization"),
) -> Dict[str, Any]:
    _log(
        "debug",
        "Incoming /session/mode request",
        agent=payload.agent_name,
        session_id=payload.session_id,
        mode_id=payload.mode_id,
        client_session_id=payload.client_session_id,
        has_token=bool(authorization),
    )
    if payload.agent_name.lower() != CODEX_AGENT_NAME:
        raise HTTPException(
            status_code=404,
            detail={"message": f"Unknown agent '{payload.agent_name}'"},
        )
    token = _bearer_token(authorization)
    try:
        result = await manager.set_session_mode_for_agent(
            token,
            payload.agent_name,
            payload.session_id,
            payload.mode_id,
            payload.client_session_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail={"message": str(exc)}) from exc
    except CodexError as exc:
        _log(
            "error",
            "Codex session/mode failed",
            session_id=payload.session_id,
            mode_id=payload.mode_id,
            error=exc.to_dict(),
        )
        status = 401 if exc.code == -32000 else 502
        raise HTTPException(status_code=status, detail=exc.to_dict()) from exc
    except CodexProcessClosed as exc:
        raise HTTPException(status_code=502, detail={"message": str(exc)}) from exc
    return _build_session_response(result)


@app.post("/session/cancel")
async def cancel_session_prompt(
    payload: SessionCancelRequest,
    authorization: Optional[str] = Header(default=None, alias="Authorization"),
) -> JSONResponse:
    _log(
        "debug",
        "Incoming /session/cancel request",
        agent=payload.agent_name,
        session_id=payload.session_id,
        client_session_id=payload.client_session_id,
        has_token=bool(authorization),
    )
    if payload.agent_name.lower() != CODEX_AGENT_NAME:
        raise HTTPException(
            status_code=404,
            detail={"message": f"Unknown agent '{payload.agent_name}'"},
        )
    token = _bearer_token(authorization)
    try:
        await manager.cancel_session_prompt_for_agent(
            token,
            payload.agent_name,
            payload.session_id,
            payload.client_session_id,
        )
    except CodexError as exc:
        _log(
            "error",
            "Codex session/cancel failed",
            session_id=payload.session_id,
            error=exc.to_dict(),
        )
        status = 401 if exc.code == -32000 else 502
        raise HTTPException(status_code=status, detail=exc.to_dict()) from exc
    except CodexProcessClosed as exc:
        raise HTTPException(status_code=502, detail={"message": str(exc)}) from exc
    return JSONResponse({"result": "cancelled"})


@app.post("/runs")
async def create_run(
    payload: RunRequest,
    authorization: Optional[str] = Header(default=None, alias="Authorization"),
) -> JSONResponse:
    _log(
        "debug",
        "Incoming /runs request",
        agent=payload.agent_name,
        mode=payload.mode,
        session_id=payload.session_id,
        model_id=payload.model_id,
        input_messages=len(payload.input),
        attachments=len(payload.attachments),
        has_token=bool(authorization),
    )

    token = _bearer_token(authorization)

    if payload.agent_name.lower() != CODEX_AGENT_NAME:
        raise HTTPException(
            status_code=404,
            detail={"message": f"Unknown agent '{payload.agent_name}'"},
        )
    if payload.mode.lower() != "sync":
        raise HTTPException(
            status_code=400,
            detail={"message": f"Unsupported run mode '{payload.mode}'"},
        )

    try:
        attachment_blocks = normalize_attachments(
            payload.attachments,
            inline_limit=ATTACHMENT_MAX_INLINE_BYTES,
        )
    except AttachmentError as exc:
        raise HTTPException(status_code=exc.status_code, detail={"message": str(exc)}) from exc

    try:
        result = await manager.run_prompt(
            token,
            payload.session_id,
            payload.input,
            attachments=attachment_blocks,
            model_id=payload.model_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail={"message": str(exc)}) from exc
    except CodexError as exc:
        _log(
            "error",
            "Codex run failed",
            session_id=payload.session_id,
            error=exc.to_dict(),
        )
        status = 401 if exc.code == -32000 else 502
        raise HTTPException(status_code=status, detail=exc.to_dict()) from exc
    except CodexProcessClosed as exc:
        # The process died unexpectedly; surface as 502 for the client.
        raise HTTPException(status_code=502, detail={"message": str(exc)}) from exc

    run_id = str(uuid.uuid4())
    response_payload = {
        "run": {
            RUN_ID_KEY: run_id,
            AGENT_NAME_KEY: CODEX_AGENT_NAME,
            SESSION_ID_KEY: result.session_id,
            RUN_STATUS_KEY: "completed",
            RUN_STOP_REASON_KEY: result.stop_reason,
            RUN_OUTPUT_KEY: result.output_messages,
            RUN_EVENTS_KEY: result.events,
        }
    }

    _log(
        "info",
        "Run completed",
        run_id=run_id,
        session_id=result.session_id,
        stop_reason=result.stop_reason,
        model_id=payload.model_id,
        attachments=len(attachment_blocks),
    )
    _log(
        "debug",
        "Responding to /runs",
        run_id=run_id,
        session_id=result.session_id,
        stop_reason=result.stop_reason,
        output_messages=len(result.output_messages),
        event_count=len(result.events),
        model_id=payload.model_id,
        attachments=len(attachment_blocks),
    )

    return JSONResponse(response_payload)


@app.get("/session/{session_id}/updates")
async def session_updates(
    session_id: str,
    authorization: Optional[str] = Header(default=None, alias="Authorization"),
) -> StreamingResponse:
    _log(
        "debug",
        "Incoming session stream request",
        session_id=session_id,
        has_token=bool(authorization),
    )

    token = _bearer_token(authorization)

    async def event_stream() -> AsyncIterator[str]:
        try:
            async for update in manager.stream_session(token, session_id):
                payload = json.dumps(update)
                update_type = None
                if isinstance(update, dict):
                    update_type = update.get("sessionUpdate") or update.get("type")
                _log(
                    "debug",
                    "Streaming session update",
                    session_id=session_id,
                    update_type=update_type or "unknown",
                    details=update,
                )
                yield f"data: {payload}\n\n"
        except CodexError as exc:
            _log("error", "Streaming error", session_id=session_id, error=exc.to_dict())
        except CodexProcessClosed as exc:
            _log("warning", "Session stream closed", session_id=session_id, reason=str(exc))
        finally:
            _log("debug", "Streaming finished", session_id=session_id)

    return StreamingResponse(event_stream(), media_type="text/event-stream")
