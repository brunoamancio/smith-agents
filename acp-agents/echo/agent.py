import asyncio
import json
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from acp_agents_common import discover_services_root
from acp_agents_common.attachments import Attachment


SERVICES_ROOT = discover_services_root(Path(__file__).resolve().parent)
if str(SERVICES_ROOT) not in sys.path:
    sys.path.append(str(SERVICES_ROOT))

from acp_agents_common.mode_utils import (
    AGENT_NAME_KEY,
    CLIENT_SESSION_ID_KEY,
    MODELS_BLOCK_KEY,
    MODE_AUTO,
    MODE_FULL_ACCESS,
    MODE_ID_KEY,
    MODE_READ_ONLY,
    MODES_BLOCK_KEY,
    MODEL_ID_KEY,
    RUN_EVENTS_KEY,
    RUN_ID_KEY,
    RUN_OUTPUT_KEY,
    RUN_STATUS_KEY,
    RUN_STOP_REASON_KEY,
    SESSION_ID_KEY,
    ModeCapabilities,
    canonicalize_models,
    canonicalize_modes,
)

ECHO_AGENT_NAME = "echo"
DEFAULT_MODEL_ID = "echo-v1"
DEFAULT_ASSISTANT_FALLBACK = "I can only echo text input."

_MODEL_ENTRIES: List[Dict[str, Any]] = [
    {
        "model_id": DEFAULT_MODEL_ID,
        "name": "Echo v1",
        "description": "Deterministic echo responses for development and testing.",
    }
]

_MODE_PRESETS: Dict[str, ModeCapabilities] = {
    MODE_READ_ONLY: ModeCapabilities(filesystem_read=False),
    MODE_AUTO: ModeCapabilities(filesystem_read=False, filesystem_write=False, auto_apply=False),
    MODE_FULL_ACCESS: ModeCapabilities(
        filesystem_read=False,
        filesystem_write=False,
        terminal_exec=False,
        network_access=False,
        auto_apply=False,
    ),
}

_MODE_ENTRIES: List[Dict[str, Any]] = [
    {
        "id": MODE_READ_ONLY,
        "label": "Read Only",
        "description": "Inspect context without mutating the filesystem or network.",
    },
]

_MODE_IDS = set(_MODE_PRESETS.keys())
_MODEL_IDS = {entry["model_id"] for entry in _MODEL_ENTRIES}


@dataclass
class EchoSession:
    agent_name: str
    session_id: str
    client_session_id: str
    current_mode_id: str = MODE_READ_ONLY
    current_model_id: str = DEFAULT_MODEL_ID
    _watchers: Set[asyncio.Queue] = field(default_factory=set)

    def register_watcher(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue()
        self._watchers.add(queue)
        return queue

    def unregister_watcher(self, queue: asyncio.Queue) -> None:
        self._watchers.discard(queue)

    async def broadcast(self, update: Dict[str, Any]) -> None:
        if not self._watchers:
            return
        stale: List[asyncio.Queue] = []
        for queue in self._watchers:
            try:
                queue.put_nowait(update)
            except asyncio.QueueFull:
                stale.append(queue)
        for queue in stale:
            self._watchers.discard(queue)

    def close(self, reason: str) -> None:
        if not self._watchers:
            return
        notice = {
            "sessionId": self.session_id,
            "sessionUpdate": "session_closed",
            "reason": reason,
        }
        for queue in list(self._watchers):
            queue.put_nowait(notice)
            queue.put_nowait(None)
        self._watchers.clear()

    def models_payload(self) -> Dict[str, Any]:
        return {
            "current_model_id": self.current_model_id,
            "available_models": [dict(entry) for entry in _MODEL_ENTRIES],
        }

    def modes_payload(self) -> Dict[str, Any]:
        return {
            "current_mode_id": self.current_mode_id,
            "available_modes": [dict(entry) for entry in _MODE_ENTRIES],
        }

app = FastAPI(title="Smith Echo Adapter", version="0.1.0")

_sessions_lock = asyncio.Lock()
_sessions_by_id: Dict[str, EchoSession] = {}
_client_to_session: Dict[Tuple[str, str], str] = {}


def _client_key(agent_name: str, client_session_id: str) -> Tuple[str, str]:
    return (agent_name.lower(), client_session_id)


def _session_payload(session: EchoSession) -> Dict[str, Any]:
    return {
        SESSION_ID_KEY: session.session_id,
        MODELS_BLOCK_KEY: canonicalize_models(session.models_payload()),
        MODES_BLOCK_KEY: canonicalize_modes(session.modes_payload(), _MODE_PRESETS, session.agent_name),
    }


def _ensure_agent(agent_name: str) -> str:
    normalized = (agent_name or "").strip().lower()
    if normalized != ECHO_AGENT_NAME:
        raise HTTPException(
            status_code=404,
            detail={"message": f"Unknown agent '{agent_name}'"},
        )
    return normalized


async def _store_session(session: EchoSession) -> None:
    async with _sessions_lock:
        key = _client_key(session.agent_name, session.client_session_id)
        previous = _client_to_session.get(key)
        if previous and previous != session.session_id:
            _sessions_by_id.pop(previous, None)
        _client_to_session[key] = session.session_id
        _sessions_by_id[session.session_id] = session


async def _resolve_session(agent_name: str, identifier: Optional[str]) -> EchoSession:
    if not identifier:
        raise HTTPException(status_code=404, detail={"message": "Session id is required"})
    async with _sessions_lock:
        session = _sessions_by_id.get(identifier)
        if session is None:
            key = _client_key(agent_name, identifier)
            mapped = _client_to_session.get(key)
            if mapped:
                session = _sessions_by_id.get(mapped)
    if session is None:
        raise HTTPException(
            status_code=404,
            detail={"message": f"Session '{identifier}' not found for agent '{agent_name}'"},
        )
    return session


async def _session_by_remote(session_id: str) -> EchoSession:
    async with _sessions_lock:
        session = _sessions_by_id.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail={"message": f"Session '{session_id}' not found"})
    return session


def _remove_client_mappings_locked(session_id: str) -> None:
    stale_keys = [key for key, value in _client_to_session.items() if value == session_id]
    for key in stale_keys:
        _client_to_session.pop(key, None)


async def _remove_sessions_matching(predicate: Callable[[str], bool], reason: str) -> List[str]:
    sessions_to_close: List[EchoSession] = []
    removed_ids: List[str] = []
    async with _sessions_lock:
        for session_id, session in list(_sessions_by_id.items()):
            if not predicate(session_id):
                continue
            removed_ids.append(session_id)
            _sessions_by_id.pop(session_id, None)
            _remove_client_mappings_locked(session_id)
            sessions_to_close.append(session)
    for session in sessions_to_close:
        session.close(reason)
    return removed_ids


def _extract_user_text(messages: List[Dict[str, Any]]) -> str:
    for message in reversed(messages or []):
        role = str(message.get("role", "")).lower()
        if role != "user":
            continue
        parts = message.get("parts") or []
        collected: List[str] = []
        for part in parts:
            if not isinstance(part, dict):
                continue
            content = part.get("content")
            if isinstance(content, str):
                collected.append(content)
            elif isinstance(content, dict):
                text_value = content.get("text") or content.get("value")
                if isinstance(text_value, str):
                    collected.append(text_value)
        if collected:
            text = "\n".join(chunk.strip() for chunk in collected if chunk and chunk.strip())
            if text:
                return text
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
    return ""


def _validate_run_mode(mode: str) -> None:
    if (mode or "").lower() == "sync":
        return
    raise HTTPException(
        status_code=400,
        detail={"message": f"Unsupported run mode '{mode}'"},
    )


def _ensure_supported_model(model_id: str) -> None:
    if model_id in _MODEL_IDS:
        return
    raise HTTPException(
        status_code=422,
        detail={"message": f"Model '{model_id}' is not supported by {ECHO_AGENT_NAME}"},
    )


def _build_assistant_response(messages: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    assistant_text = _extract_user_text(messages).strip() or DEFAULT_ASSISTANT_FALLBACK
    assistant_message = {
        "role": "assistant",
        "parts": [
            {
                "content_type": "text/plain",
                "content": assistant_text,
                "content_encoding": "plain",
            }
        ],
    }
    events = [
        {
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": assistant_text,
                    }
                ],
            },
            "done": True,
        }
    ]
    return assistant_message, events


def _stream_update_payload(session_id: str, run_id: str, events: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {"sessionId": session_id, "promptId": run_id, RUN_EVENTS_KEY: events}


def _run_response_payload(
    session_id: str,
    run_id: str,
    assistant_message: Dict[str, Any],
    events: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "run": {
            RUN_ID_KEY: run_id,
            AGENT_NAME_KEY: ECHO_AGENT_NAME,
            SESSION_ID_KEY: session_id,
            RUN_STATUS_KEY: "completed",
            RUN_STOP_REASON_KEY: "done",
            RUN_OUTPUT_KEY: [assistant_message],
            RUN_EVENTS_KEY: events,
        }
    }


def _ensure_supported_mode(mode_id: str) -> None:
    if mode_id in _MODE_IDS:
        return
    raise HTTPException(
        status_code=422,
        detail={"message": f"Mode '{mode_id}' is not supported by {ECHO_AGENT_NAME}"},
    )


async def _ensure_session_mode(session: EchoSession, mode_id: str) -> None:
    _ensure_supported_mode(mode_id)
    previous = session.current_mode_id
    if mode_id == previous:
        return
    session.current_mode_id = mode_id
    await session.broadcast(
        {
            "sessionId": session.session_id,
            "events": [
                {"type": "mode_changed", "previousMode": previous, "nextMode": session.current_mode_id}
            ],
        }
    )


async def _ensure_agent_session(agent_name: str, session_id: str) -> Tuple[str, EchoSession]:
    agent = _ensure_agent(agent_name)
    session = await _resolve_session(agent, session_id)
    return agent, session


async def _sync_client_session_mapping(agent: str, session: EchoSession, desired_client_id: Optional[str]) -> None:
    if not desired_client_id:
        return
    async with _sessions_lock:
        old_key = _client_key(agent, session.client_session_id)
        new_key = _client_key(agent, desired_client_id)
        if old_key != new_key:
            _client_to_session.pop(old_key, None)
        session.client_session_id = desired_client_id
        _client_to_session[new_key] = session.session_id


def _ensure_session_model(session: EchoSession, model_id: Optional[str]) -> None:
    if not model_id or model_id == session.current_model_id:
        return
    _ensure_supported_model(model_id)
    session.current_model_id = model_id


class SessionNewRequest(BaseModel):
    agent_name: str = Field(alias=AGENT_NAME_KEY)
    client_session_id: str = Field(alias=CLIENT_SESSION_ID_KEY)
    mcpServers: Optional[List[Dict[str, Any]]] = Field(default=None, alias="mcpServers")

    class Config:
        populate_by_name = True


class SessionLoadRequest(BaseModel):
    agent_name: str = Field(alias=AGENT_NAME_KEY)
    session_id: str = Field(alias=SESSION_ID_KEY)
    client_session_id: Optional[str] = Field(default=None, alias=CLIENT_SESSION_ID_KEY)
    mcpServers: Optional[List[Dict[str, Any]]] = Field(default=None, alias="mcpServers")

    class Config:
        populate_by_name = True


class SessionModelRequest(BaseModel):
    agent_name: str = Field(alias=AGENT_NAME_KEY)
    session_id: str = Field(alias=SESSION_ID_KEY)
    model_id: str = Field(alias=MODEL_ID_KEY)
    client_session_id: Optional[str] = Field(default=None, alias=CLIENT_SESSION_ID_KEY)

    class Config:
        populate_by_name = True


class SessionModeRequest(BaseModel):
    agent_name: str = Field(alias=AGENT_NAME_KEY)
    session_id: str = Field(alias=SESSION_ID_KEY)
    mode_id: str = Field(alias=MODE_ID_KEY)
    client_session_id: Optional[str] = Field(default=None, alias=CLIENT_SESSION_ID_KEY)

    class Config:
        populate_by_name = True


class SessionCancelRequest(BaseModel):
    agent_name: str = Field(alias=AGENT_NAME_KEY)
    session_id: str = Field(alias=SESSION_ID_KEY)
    client_session_id: Optional[str] = Field(default=None, alias=CLIENT_SESSION_ID_KEY)

    class Config:
        populate_by_name = True


class SessionPruneRequest(BaseModel):
    keep_codex_session_ids: List[str] = Field(default_factory=list)


class SessionDeleteRequest(BaseModel):
    codex_session_ids: List[str] = Field(default_factory=list)


class RunRequest(BaseModel):
    agent_name: str = Field(alias=AGENT_NAME_KEY)
    mode: str
    session_id: Optional[str] = Field(default=None, alias=SESSION_ID_KEY)
    model_id: Optional[str] = Field(default=None, alias=MODEL_ID_KEY)
    input: List[Dict[str, Any]]
    attachments: List[Attachment] = Field(default_factory=list)

    class Config:
        populate_by_name = True


@app.get("/ping")
def ping() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/agents")
def list_agents() -> Dict[str, Any]:
    return {
        "agents": [
            {
                "name": ECHO_AGENT_NAME,
                "description": "Smith demo echo agent",
                "metadata": {"provider": "smith", "demo": True},
            }
        ]
    }


@app.post("/session/new")
async def create_session(
    payload: SessionNewRequest,
    authorization: Optional[str] = Header(default=None, alias="Authorization"),
) -> Dict[str, Any]:
    agent = _ensure_agent(payload.agent_name)
    remote_session_id = str(uuid.uuid4())
    session = EchoSession(agent_name=agent, session_id=remote_session_id, client_session_id=payload.client_session_id)
    await _store_session(session)
    return {"session": _session_payload(session)}


@app.post("/session/load")
async def load_session(
    payload: SessionLoadRequest,
    authorization: Optional[str] = Header(default=None, alias="Authorization"),
) -> Dict[str, Any]:
    agent, session = await _ensure_agent_session(payload.agent_name, payload.session_id)
    await _sync_client_session_mapping(agent, session, payload.client_session_id)
    return {"session": _session_payload(session)}


@app.post("/sessions/prune")
async def prune_sessions(
    payload: SessionPruneRequest,
    authorization: Optional[str] = Header(default=None, alias="Authorization"),
) -> JSONResponse:
    keep = {session_id.lower() for session_id in payload.keep_codex_session_ids if session_id}
    removed_ids = await _remove_sessions_matching(
        lambda session_id, keep=keep: session_id.lower() not in keep,
        reason="pruned",
    )
    return JSONResponse({"result": {"removed_codex_session_ids": removed_ids, "removed_artifacts": []}})


@app.post("/sessions/delete")
async def delete_sessions(
    payload: SessionDeleteRequest,
    authorization: Optional[str] = Header(default=None, alias="Authorization"),
) -> JSONResponse:
    targets = {session_id.lower() for session_id in payload.codex_session_ids if session_id}
    if not targets:
        return JSONResponse({"result": {"removed_codex_session_ids": [], "removed_artifacts": []}})
    removed_ids = await _remove_sessions_matching(
        lambda session_id, targets=targets: session_id.lower() in targets,
        reason="deleted",
    )
    return JSONResponse({"result": {"removed_codex_session_ids": removed_ids, "removed_artifacts": []}})


@app.post("/session/model")
async def set_session_model(
    payload: SessionModelRequest,
    authorization: Optional[str] = Header(default=None, alias="Authorization"),
) -> Dict[str, Any]:
    agent, session = await _ensure_agent_session(payload.agent_name, payload.session_id)
    _ensure_session_model(session, payload.model_id)
    await _sync_client_session_mapping(agent, session, payload.client_session_id)
    return {"session": _session_payload(session)}


@app.post("/session/mode")
async def set_session_mode(
    payload: SessionModeRequest,
    authorization: Optional[str] = Header(default=None, alias="Authorization"),
) -> Dict[str, Any]:
    agent, session = await _ensure_agent_session(payload.agent_name, payload.session_id)
    await _ensure_session_mode(session, payload.mode_id)
    await _sync_client_session_mapping(agent, session, payload.client_session_id)
    return {"session": _session_payload(session)}


@app.post("/session/cancel")
async def cancel_session_prompt(
    payload: SessionCancelRequest,
    authorization: Optional[str] = Header(default=None, alias="Authorization"),
) -> JSONResponse:
    agent, session = await _ensure_agent_session(payload.agent_name, payload.session_id)
    await _sync_client_session_mapping(agent, session, payload.client_session_id)
    await session.broadcast(
        {
            "sessionId": session.session_id,
            "events": [
                {
                    "type": "session_cancelled",
                    "reason": "cancelled_by_client",
                }
            ],
        }
    )
    return JSONResponse({"result": "cancelled"})


@app.post("/runs")
async def create_run(
    payload: RunRequest,
    authorization: Optional[str] = Header(default=None, alias="Authorization"),
) -> JSONResponse:
    agent = _ensure_agent(payload.agent_name)
    _validate_run_mode(payload.mode)
    session = await _resolve_session(agent, payload.session_id)
    _ensure_session_model(session, payload.model_id)
    assistant_message, events = _build_assistant_response(payload.input)

    run_id = str(uuid.uuid4())
    await session.broadcast(_stream_update_payload(session.session_id, run_id, events))

    return JSONResponse(_run_response_payload(session.session_id, run_id, assistant_message, events))


@app.get("/session/{session_id}/updates")
async def session_updates(session_id: str) -> StreamingResponse:
    session = await _session_by_remote(session_id)
    queue = session.register_watcher()

    async def event_stream():
        try:
            while True:
                update = await queue.get()
                if update is None:
                    break
                yield f"data: {json.dumps(update)}\n\n"
        finally:
            session.unregister_watcher(queue)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("agent:app", host="0.0.0.0", port=8080, reload=False)
