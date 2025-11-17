from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, List, Mapping, Optional, Set, Tuple

from acp_agents_common import discover_services_root
from acp_agents_common.attachments import (
    PROMPT_CAPABILITIES_KEY,
    PROMPT_CAPABILITY_EMBEDDED_CONTEXT,
    PROMPT_CAPABILITY_IMAGE,
)


SERVICES_ROOT = discover_services_root(Path(__file__).resolve().parent)
if str(SERVICES_ROOT) not in sys.path:
    sys.path.append(str(SERVICES_ROOT))

from acp_agents_common.mode_utils import (
    SESSION_ID_KEY,
    MODE_AUTO,
    MODE_FULL_ACCESS,
    MODE_READ_ONLY,
    MODE_RESPONSE_KEYS,
    MODELS_BLOCK_KEY,
    MODES_BLOCK_KEY,
    ModeCapabilities,
    canonicalize_models,
    canonicalize_modes,
)


CODEX_AGENT_NAME = "codex"
_SESSION_ID_PATTERN = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
    re.IGNORECASE,
)


_LOG = logging.getLogger("codex.bridge")


class CodexError(Exception):
    """Raised when the Codex ACP process returns a JSON-RPC error."""

    def __init__(self, code: int, message: str, data: Optional[Any] = None) -> None:
        super().__init__(message)
        self.code = code
        self.data = data

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"code": self.code, "message": str(self)}
        if self.data is not None:
            payload["data"] = self.data
        return payload


class CodexProcessClosed(Exception):
    """Raised when operations are attempted on a closed Codex process."""


NotificationHandler = Callable[[str, Dict[str, Any]], Awaitable[None]]


class CodexProcess:
    """Manages the codex-acp child process and JSON-RPC traffic."""

    def __init__(self, binary_path: Path) -> None:
        self._binary_path = binary_path
        self._proc: Optional[asyncio.subprocess.Process] = None
        self._stdout_task: Optional[asyncio.Task[None]] = None
        self._stderr_task: Optional[asyncio.Task[None]] = None
        self._pending: Dict[str, asyncio.Future[Any]] = {}
        self._notification_handler: Optional[NotificationHandler] = None
        self._id_counter = 0
        self._lock = asyncio.Lock()
        self._current_token: Optional[str] = None
        self._background_tasks: Set[asyncio.Task[Any]] = set()

    async def ensure_started(self, token: Optional[str]) -> bool:
        """Ensure the child process is running with the given token.

        Returns True if the process was started or restarted.
        """
        async with self._lock:
            if (
                self._proc is not None
                and self._proc.returncode is None
                and token == self._current_token
            ):
                return False

            await self._shutdown_locked()

            env = os.environ.copy()
            env.pop("CODEX_API_KEY", None)
            if token:
                env["OPENAI_API_KEY"] = token
            else:
                env.pop("OPENAI_API_KEY", None)

            _LOG.info(
                json.dumps(
                    {
                        "message": "Starting codex-acp process",
                        "binary_path": str(self._binary_path),
                        "with_token": bool(token),
                    }
                )
            )

            self._proc = await asyncio.create_subprocess_exec(
                str(self._binary_path),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
            self._current_token = token
            self._stdout_task = asyncio.create_task(self._read_stdout())
            self._stderr_task = asyncio.create_task(self._read_stderr())
            return True

    async def _shutdown_locked(self) -> None:
        if self._stdout_task:
            self._stdout_task.cancel()
        if self._stderr_task:
            self._stderr_task.cancel()

        if self._proc and self._proc.returncode is None:
            self._proc.terminate()
            try:
                await asyncio.wait_for(self._proc.wait(), timeout=5)
            except asyncio.TimeoutError:
                self._proc.kill()
                await self._proc.wait()

        if self._stdout_task:
            with contextlib.suppress(asyncio.CancelledError):
                await self._stdout_task
        if self._stderr_task:
            with contextlib.suppress(asyncio.CancelledError):
                await self._stderr_task

        self._stdout_task = None
        self._stderr_task = None
        self._proc = None
        self._current_token = None

        self._fail_pending_requests("codex-acp process stopped")
        for task in self._background_tasks:
            task.cancel()
        self._background_tasks.clear()

    async def stop(self) -> None:
        async with self._lock:
            await self._shutdown_locked()

    def set_notification_handler(self, handler: NotificationHandler) -> None:
        self._notification_handler = handler

    async def request(self, method: str, params: Optional[Dict[str, Any]]) -> Any:
        if not self._proc or self._proc.stdin is None:
            raise CodexProcessClosed("codex-acp process is not running")

        self._id_counter += 1
        request_id = str(self._id_counter)
        message = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params or {},
        }

        if _LOG.isEnabledFor(logging.DEBUG):
            _LOG.debug(json.dumps({"message": "codex_request", "payload": message}))

        future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future

        payload = (json.dumps(message) + "\n").encode("utf-8")
        try:
            self._proc.stdin.write(payload)
            await self._proc.stdin.drain()
        except BrokenPipeError as exc:
            if not future.done():
                future.set_exception(CodexProcessClosed(str(exc)))
            raise CodexProcessClosed(str(exc)) from exc

        return await future

    async def notify(self, method: str, params: Optional[Dict[str, Any]]) -> None:
        if not self._proc or self._proc.stdin is None:
            raise CodexProcessClosed("codex-acp process is not running")

        message = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {},
        }

        if _LOG.isEnabledFor(logging.DEBUG):
            _LOG.debug(json.dumps({"message": "codex_notify", "payload": message}))

        payload = (json.dumps(message) + "\n").encode("utf-8")
        try:
            self._proc.stdin.write(payload)
            await self._proc.stdin.drain()
        except BrokenPipeError as exc:
            raise CodexProcessClosed(str(exc)) from exc

    def _schedule_notification(self, method: str, params: Dict[str, Any]) -> None:
        handler = self._notification_handler
        if not handler:
            return
        task = asyncio.create_task(handler(method, params))
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    def _fail_pending_requests(self, reason: str) -> None:
        for future in self._pending.values():
            if not future.done():
                future.set_exception(CodexProcessClosed(reason))
        self._pending.clear()

    @staticmethod
    def _decode_stdout_line(line: bytes) -> Optional[Dict[str, Any]]:
        raw = line.decode("utf-8", errors="replace").strip()
        if not raw:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            _LOG.error(json.dumps({"message": "Failed to decode Codex output", "raw": raw}))
            return None

    def _handle_response_message(self, message: Dict[str, Any]) -> None:
        request_id = str(message.get("id"))
        future = self._pending.pop(request_id, None)
        if not future:
            _LOG.warning(json.dumps({"message": "No pending request for response", "id": request_id}))
            return

        error_obj = message.get("error")
        if error_obj:
            if not future.done():
                future.set_exception(
                    CodexError(
                        code=error_obj.get("code", -32001),
                        message=error_obj.get("message", "Unknown error"),
                        data=error_obj.get("data"),
                    )
                )
            if _LOG.isEnabledFor(logging.DEBUG):
                _LOG.debug(
                    json.dumps(
                        {
                            "message": "codex_response_error",
                            "id": request_id,
                            "error": error_obj,
                        }
                    )
                )
            return

        if _LOG.isEnabledFor(logging.DEBUG):
            _LOG.debug(
                json.dumps(
                    {
                        "message": "codex_response",
                        "id": request_id,
                        "result": message.get("result"),
                    }
                )
            )
        if not future.done():
            future.set_result(message.get("result"))

    def _dispatch_notification(self, message: Dict[str, Any]) -> None:
        method = message.get("method")
        if not method:
            _LOG.debug(json.dumps({"message": "Received unexpected message without id/method", "raw": message}))
            return
        params = message.get("params") or {}
        self._schedule_notification(method, params)

    async def _read_stdout(self) -> None:
        assert self._proc and self._proc.stdout
        reader = self._proc.stdout

        while True:
            line = await reader.readline()
            if not line:
                break

            message = self._decode_stdout_line(line)
            if message is None:
                continue

            if "id" in message:
                self._handle_response_message(message)
            elif "method" in message:
                self._dispatch_notification(message)
            else:
                _LOG.debug(json.dumps({"message": "Received unexpected message without id/method", "raw": message}))

        # Process exited, fail all pending futures.
        self._fail_pending_requests("codex-acp stdout closed")

    async def _read_stderr(self) -> None:
        assert self._proc and self._proc.stderr
        reader = self._proc.stderr

        while True:
            line = await reader.readline()
            if not line:
                break
            _LOG.info(
                json.dumps(
                    {
                        "message": "codex-acp stderr",
                        "line": line.decode("utf-8", errors="replace").rstrip(),
                    }
                )
            )

    @property
    def current_token(self) -> Optional[str]:
        return self._current_token


@dataclass
class RunResult:
    session_id: str
    codex_session_id: str
    stop_reason: str
    output_messages: List[Dict[str, Any]]
    events: List[Dict[str, Any]]


_CODEX_MODE_CAPABILITIES: Dict[str, ModeCapabilities] = {
    MODE_READ_ONLY: ModeCapabilities(
        filesystem_read=True,
        filesystem_write=False,
        terminal_exec=False,
        network_access=False,
        auto_apply=False,
    ),
    MODE_AUTO: ModeCapabilities(
        filesystem_read=True,
        filesystem_write=True,
        terminal_exec=True,
        network_access=False,
        auto_apply=True,
    ),
    MODE_FULL_ACCESS: ModeCapabilities(
        filesystem_read=True,
        filesystem_write=True,
        terminal_exec=True,
        network_access=True,
        auto_apply=True,
    ),
}

_PUBLIC_MODE_CHAT = "chat"
_PUBLIC_MODE_AGENT = "agent"

_PUBLIC_MODE_LABELS = {
    _PUBLIC_MODE_CHAT: "Chat",
    _PUBLIC_MODE_AGENT: "Agent",
}

_PUBLIC_MODE_DESCRIPTIONS = {
    _PUBLIC_MODE_CHAT: "Discuss without changing files or running commands.",
    _PUBLIC_MODE_AGENT: "Allow the assistant to edit files and execute commands.",
}

_MODE_KEY_CURRENT = MODE_RESPONSE_KEYS["current_mode_id"]
_MODE_KEY_AVAILABLE = MODE_RESPONSE_KEYS["available_modes"]
_MODE_KEY_ID = MODE_RESPONSE_KEYS["mode_id"]
_MODE_KEY_LABEL = MODE_RESPONSE_KEYS["mode_label"]
_MODE_KEY_DESCRIPTION = MODE_RESPONSE_KEYS["mode_description"]
_MODE_KEY_CAPABILITIES = MODE_RESPONSE_KEYS["mode_capabilities"]
_MODE_KEY_META = MODE_RESPONSE_KEYS["mode_meta"]

_PUBLIC_MODE_CANDIDATES: Dict[str, Tuple[str, ...]] = {
    _PUBLIC_MODE_CHAT: (MODE_READ_ONLY,),
    _PUBLIC_MODE_AGENT: (MODE_FULL_ACCESS, MODE_AUTO),
}

_PUBLIC_MODE_DEFAULT_VENDOR: Dict[str, str] = {
    _PUBLIC_MODE_CHAT: MODE_READ_ONLY,
    _PUBLIC_MODE_AGENT: MODE_FULL_ACCESS,
}


def _resolve_vendor_mode(public_mode_id: str, vendor_lookup: Dict[str, Dict[str, Any]]) -> Optional[str]:
    candidates = _PUBLIC_MODE_CANDIDATES.get(public_mode_id, ())
    for candidate in candidates:
        if candidate in vendor_lookup:
            return candidate
    return None


def _build_public_modes(
    vendor_modes: Optional[Dict[str, Any]]
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """Translate Codex vendor modes into the public Chat/Agent presets."""
    available_public: List[Dict[str, Any]] = []
    public_current: Optional[str] = None
    mapping: Dict[str, str] = {}

    if isinstance(vendor_modes, dict):
        vendor_current = vendor_modes.get(_MODE_KEY_CURRENT)
        vendor_available = vendor_modes.get(_MODE_KEY_AVAILABLE) or []
        vendor_lookup = {}
        for entry in vendor_available:
            if not isinstance(entry, dict):
                continue
            mode_id = entry.get(_MODE_KEY_ID)
            if isinstance(mode_id, str):
                vendor_lookup[mode_id] = entry

        for public_mode_id in (_PUBLIC_MODE_CHAT, _PUBLIC_MODE_AGENT):
            codex_mode_id = _resolve_vendor_mode(public_mode_id, vendor_lookup)
            vendor_entry = vendor_lookup.get(codex_mode_id) if codex_mode_id else None
            if codex_mode_id is None:
                codex_mode_id = _PUBLIC_MODE_DEFAULT_VENDOR.get(public_mode_id)
            capabilities_payload = (
                vendor_entry.get(_MODE_KEY_CAPABILITIES)
                if vendor_entry
                else _CODEX_MODE_CAPABILITIES.get(codex_mode_id, ModeCapabilities()).to_payload()
            )
            vendor_meta = vendor_entry.get(_MODE_KEY_META) if vendor_entry else {}
            meta_payload = dict(vendor_meta) if isinstance(vendor_meta, dict) else {}
            meta_payload.setdefault("vendor", CODEX_AGENT_NAME)
            if codex_mode_id:
                meta_payload["vendor_mode_id"] = codex_mode_id
            if not vendor_entry:
                meta_payload.setdefault("source", "smith_fallback")

            available_public.append(
                {
                    _MODE_KEY_ID: public_mode_id,
                    _MODE_KEY_LABEL: _PUBLIC_MODE_LABELS[public_mode_id],
                    _MODE_KEY_DESCRIPTION: _PUBLIC_MODE_DESCRIPTIONS[public_mode_id],
                    _MODE_KEY_CAPABILITIES: capabilities_payload,
                    _MODE_KEY_META: meta_payload,
                }
            )
            if vendor_entry and vendor_current == meta_payload.get("vendor_mode_id"):
                public_current = public_mode_id
            if codex_mode_id:
                mapping[public_mode_id] = codex_mode_id

        if public_current is None and vendor_modes.get(_MODE_KEY_CURRENT) == MODE_AUTO:
            # Auto mode is closer to Agent behaviour; prefer Agent if available.
            if any(entry.get(_MODE_KEY_ID) == _PUBLIC_MODE_AGENT for entry in available_public):
                public_current = _PUBLIC_MODE_AGENT

    if public_current is None and available_public:
        public_current = available_public[0][_MODE_KEY_ID]

    if not available_public:
        # Fallback: synthesize both modes from presets.
        for public_mode_id in (_PUBLIC_MODE_CHAT, _PUBLIC_MODE_AGENT):
            codex_mode_id = _PUBLIC_MODE_DEFAULT_VENDOR[public_mode_id]
            available_public.append(
                {
                    _MODE_KEY_ID: public_mode_id,
                    _MODE_KEY_LABEL: _PUBLIC_MODE_LABELS[public_mode_id],
                    _MODE_KEY_DESCRIPTION: _PUBLIC_MODE_DESCRIPTIONS[public_mode_id],
                    _MODE_KEY_CAPABILITIES: _CODEX_MODE_CAPABILITIES[codex_mode_id].to_payload(),
                    _MODE_KEY_META: {
                        "vendor": CODEX_AGENT_NAME,
                        "vendor_mode_id": codex_mode_id,
                        "source": "smith_fallback",
                    },
                }
            )
            mapping[public_mode_id] = codex_mode_id
        public_current = _PUBLIC_MODE_CHAT

    return (
        {
            _MODE_KEY_CURRENT: public_current,
            _MODE_KEY_AVAILABLE: available_public,
        },
        mapping,
    )


def _codex_mode_for_public(public_mode_id: str) -> str:
    candidates = _PUBLIC_MODE_CANDIDATES.get(public_mode_id)
    if not candidates:
        expected = ", ".join(sorted(_PUBLIC_MODE_CANDIDATES.keys()))
        raise ValueError(f"Unsupported mode '{public_mode_id}'. Expected one of: {expected}")
    return candidates[0]

class CodexManager:
    """High-level orchestrator that exposes prompt execution."""

    def __init__(self, binary_path: Path, workspace: Path) -> None:
        self._process = CodexProcess(binary_path)
        self._workspace = workspace
        self._workspace_root = workspace.resolve()
        default_home = workspace / ".codex"
        self._codex_home = Path(os.getenv("CODEX_HOME", str(default_home)))
        self._sessions_root = self._codex_home / "sessions"
        self._initialized = False
        self._client_to_codex: Dict[Tuple[str, str], str] = {}
        self._session_watchers: Dict[str, List[asyncio.Queue[Dict[str, Any]]]] = {}
        self._session_mcp_servers: Dict[str, List[Dict[str, Any]]] = {}
        self._session_mode_map: Dict[str, Dict[str, str]] = {}
        self._auth_methods: set[str] = set()
        self._prompt_capabilities: Dict[str, bool] = {}
        self._update_prompt_capabilities(None)
        self._lock = asyncio.Lock()

        self._process.set_notification_handler(self._handle_notification)

    async def stop(self) -> None:
        await self._process.stop()

    async def run_prompt(
        self,
        token: Optional[str],
        client_session_id: Optional[str],
        messages: List[Dict[str, Any]],
        attachments: Optional[List[Dict[str, Any]]] = None,
        model_id: Optional[str] = None,
    ) -> RunResult:
        client_session, codex_session, queue = await self._prepare_prompt_context(
            token, client_session_id, model_id
        )
        try:
            prompt_payload = _build_prompt_payload(messages, attachments or [])
        except ValueError:
            self._detach_watcher(codex_session, queue)
            raise

        self._log_prompt_dispatch(client_session, codex_session, len(prompt_payload))
        accumulator = _PromptAccumulator()
        stop_reason = "unknown"
        try:
            stop_reason = await self._consume_prompt_stream(
                codex_session, prompt_payload, queue, accumulator
            )
        finally:
            self._detach_watcher(codex_session, queue)
            self._drain_prompt_queue(queue, accumulator)

        output_messages = accumulator.build_output()
        events = list(accumulator.events)
        self._log_prompt_completion(
            client_session,
            codex_session,
            stop_reason,
            accumulator,
            output_messages,
            events,
        )

        return RunResult(
            session_id=client_session or codex_session,
            codex_session_id=codex_session,
            stop_reason=stop_reason,
            output_messages=output_messages,
            events=events,
        )

    async def stream_session(
        self, token: Optional[str], session_identifier: str
    ) -> AsyncIterator[Dict[str, Any]]:
        codex_session = await self._resolve_codex_session(token, session_identifier)
        queue = self._attach_watcher(codex_session)

        try:
            while True:
                update = await queue.get()
                if _LOG.isEnabledFor(logging.DEBUG):
                    _LOG.debug(
                        json.dumps(
                            {
                                "message": "session_update_dispatch",
                                "client_session_id": session_identifier or None,
                                "codex_session_id": codex_session,
                                "payload": update,
                            }
                        )
                    )
                yield update
        finally:
            self._detach_watcher(codex_session, queue)

    async def _ensure_initialized(self, token: Optional[str]) -> None:
        if self._initialized:
            return

        init_result = await self._process.request(
            "initialize",
            {
                "protocolVersion": 1,
                "clientCapabilities": {
                    "fs": {"readTextFile": False, "writeTextFile": False},
                    "terminal": False,
                },
            },
        )

        self._auth_methods = {
            method.get("id")
            for method in (init_result or {}).get("authMethods", [])
            if isinstance(method, dict) and "id" in method
        }

        if token:
            if "openai-api-key" not in self._auth_methods:
                raise CodexError(
                    code=-32000,
                    message="Codex agent does not support OPENAI_API_KEY authentication",
                )
            await self._process.request("authenticate", {"methodId": "openai-api-key"})

        self._initialized = True
        agent_caps = (init_result or {}).get("agentCapabilities", {})
        prompt_caps = agent_caps.get("promptCapabilities") if isinstance(agent_caps, dict) else {}
        self._update_prompt_capabilities(prompt_caps)

    async def _ensure_process_ready(self, token: Optional[str]) -> None:
        restarted = await self._process.ensure_started(token)
        if restarted:
            _LOG.info(
                json.dumps(
                    {
                        "message": "codex-acp restarted",
                        "reason": "token_changed" if token else "process_started",
                    }
                )
            )
            self._initialized = False
            self._client_to_codex.clear()
            self._session_watchers.clear()
            self._session_mcp_servers.clear()
            self._update_prompt_capabilities(None)
        await self._ensure_initialized(token)

    def _session_key(self, agent_name: str, client_session_id: str) -> Tuple[str, str]:
        return (agent_name.lower(), client_session_id)

    async def _ensure_session(self, agent_name: str, client_session_id: str) -> str:
        key = self._session_key(agent_name, client_session_id)
        if key in self._client_to_codex:
            return self._client_to_codex[key]

        default_servers: List[Dict[str, Any]] = []
        codex_session = await self._create_codex_session(default_servers)
        self._client_to_codex[key] = codex_session
        self._session_mcp_servers[codex_session] = default_servers
        _LOG.info(
            json.dumps(
                {
                    "message": "Created Codex session",
                    "client_session_id": client_session_id or None,
                    "codex_session_id": codex_session,
                }
            )
        )
        return codex_session

    async def create_session_with_models(
        self,
        token: Optional[str],
        agent_name: str,
        client_session_id: str,
        mcp_servers: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        await self._ensure_process_ready(token)
        codex_session = await self._create_codex_session(mcp_servers)
        self._client_to_codex[self._session_key(agent_name, client_session_id)] = codex_session
        self._session_mcp_servers[codex_session] = mcp_servers
        payload = await self._load_codex_session(codex_session, mcp_servers)
        return payload

    async def load_session_with_models(
        self,
        token: Optional[str],
        agent_name: str,
        codex_session_id: str,
        client_session_id: Optional[str] = None,
        mcp_servers: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        await self._ensure_process_ready(token)
        if client_session_id is not None:
            self._client_to_codex[self._session_key(agent_name, client_session_id)] = codex_session_id
        effective_servers = mcp_servers or self._session_mcp_servers.get(codex_session_id, [])
        if effective_servers:
            self._session_mcp_servers[codex_session_id] = effective_servers
        payload = await self._load_codex_session(codex_session_id, effective_servers)
        return payload

    async def set_session_model_for_agent(
        self,
        token: Optional[str],
        agent_name: str,
        codex_session_id: str,
        model_id: str,
        client_session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        await self._ensure_process_ready(token)
        if client_session_id is not None:
            self._client_to_codex[self._session_key(agent_name, client_session_id)] = codex_session_id
        await self._set_session_model(codex_session_id, model_id)
        payload = await self._load_codex_session(
            codex_session_id, self._session_mcp_servers.get(codex_session_id, [])
        )
        return payload

    async def set_session_mode_for_agent(
        self,
        token: Optional[str],
        agent_name: str,
        codex_session_id: str,
        mode_id: str,
        client_session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        await self._ensure_process_ready(token)
        if client_session_id is not None:
            self._client_to_codex[self._session_key(agent_name, client_session_id)] = codex_session_id
        mapping = self._session_mode_map.get(codex_session_id, {})
        codex_mode_id = mapping.get(mode_id)
        if codex_mode_id is None:
            codex_mode_id = _codex_mode_for_public(mode_id)
        await self._set_session_mode(codex_session_id, codex_mode_id)
        payload = await self._load_codex_session(
            codex_session_id, self._session_mcp_servers.get(codex_session_id, [])
        )
        return payload

    async def cancel_session_prompt_for_agent(
        self,
        token: Optional[str],
        agent_name: str,
        codex_session_id: str,
        client_session_id: Optional[str] = None,
    ) -> None:
        await self._ensure_process_ready(token)
        if client_session_id is not None:
            self._client_to_codex[self._session_key(agent_name, client_session_id)] = codex_session_id
        await self._cancel_session_prompt(codex_session_id)

    async def _create_codex_session(self, mcp_servers: List[Dict[str, Any]]) -> str:
        request_payload = {
            "cwd": str(self._workspace),
            "mcpServers": mcp_servers,
        }
        if _LOG.isEnabledFor(logging.DEBUG):
            _LOG.debug(
                json.dumps(
                    {
                        "message": "codex_session_new_request",
                        "payload": request_payload,
                    }
                )
            )
        response = await self._process.request("session/new", request_payload)
        codex_session_raw = response.get("sessionId")
        if not codex_session_raw:
            raise CodexError(-32603, "Codex session/new did not return a sessionId")
        return str(codex_session_raw)

    async def _load_codex_session(
        self, codex_session_id: str, mcp_servers: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        response = await self._process.request(
            "session/load",
            {
                "sessionId": codex_session_id,
                "cwd": str(self._workspace),
                "mcpServers": mcp_servers,
            },
        )
        if not isinstance(response, dict):
            raise CodexError(-32603, "Codex session/load returned invalid payload")
        payload = response.get("session") if isinstance(response.get("session"), dict) else response
        session_id = str(payload.get(SESSION_ID_KEY) or payload.get("sessionId") or codex_session_id)
        canonical_models = canonicalize_models(payload.get(MODELS_BLOCK_KEY))
        vendor_modes = canonicalize_modes(
            payload.get(MODES_BLOCK_KEY), _CODEX_MODE_CAPABILITIES, CODEX_AGENT_NAME
        )
        public_modes, mapping = _build_public_modes(vendor_modes)
        self._session_mode_map[codex_session_id] = mapping
        agent_caps = payload.get("agentCapabilities") if isinstance(payload, dict) else {}
        prompt_caps = agent_caps.get("promptCapabilities") if isinstance(agent_caps, dict) else {}
        self._update_prompt_capabilities(prompt_caps)
        return {
            SESSION_ID_KEY: session_id,
            MODELS_BLOCK_KEY: canonical_models,
            MODES_BLOCK_KEY: public_modes,
            PROMPT_CAPABILITIES_KEY: dict(self._prompt_capabilities),
        }

    @property
    def prompt_capabilities(self) -> Dict[str, bool]:
        return dict(self._prompt_capabilities)

    def _update_prompt_capabilities(self, prompt_caps: Optional[Mapping[str, Any]]) -> None:
        image_supported = True
        embedded_supported = True
        if isinstance(prompt_caps, Mapping):
            if PROMPT_CAPABILITY_IMAGE in prompt_caps:
                image_supported = bool(prompt_caps.get(PROMPT_CAPABILITY_IMAGE))
            if PROMPT_CAPABILITY_EMBEDDED_CONTEXT in prompt_caps:
                embedded_supported = bool(prompt_caps.get(PROMPT_CAPABILITY_EMBEDDED_CONTEXT))
        self._prompt_capabilities = {
            PROMPT_CAPABILITY_IMAGE: image_supported,
            PROMPT_CAPABILITY_EMBEDDED_CONTEXT: embedded_supported,
        }

    async def prune_sessions(self, keep_codex_session_ids: Set[str]) -> Dict[str, List[str]]:
        normalized_keep = {session_id.lower() for session_id in keep_codex_session_ids}
        removed_codex_session_ids: List[str] = []

        session_lookup: Dict[str, str] = {}
        for value in list(self._client_to_codex.values()):
            session_lookup[value.lower()] = value
        for session_id in list(self._session_mcp_servers.keys()):
            session_lookup.setdefault(session_id.lower(), session_id)

        async with self._lock:
            removable_lower = set(session_lookup.keys()) - normalized_keep
            for lowercase_id in removable_lower:
                original_id = session_lookup[lowercase_id]
                if self._clear_session_tracking(original_id):
                    removed_codex_session_ids.append(original_id)

        removed_artifacts = self._prune_session_files(normalized_keep)
        if removed_codex_session_ids or removed_artifacts:
            _LOG.info(
                json.dumps(
                    {
                        "message": "pruned_codex_sessions",
                        "removed_codex_session_ids": removed_codex_session_ids,
                        "removed_artifacts": removed_artifacts,
                    }
                )
            )

        return {
            "removed_codex_session_ids": removed_codex_session_ids,
            "removed_artifacts": removed_artifacts,
        }

    async def delete_sessions(self, codex_session_ids: Set[str]) -> Dict[str, List[str]]:
        normalized_targets = {session_id.lower() for session_id in codex_session_ids if session_id}
        if not normalized_targets:
            return {"removed_codex_session_ids": [], "removed_artifacts": []}

        session_lookup: Dict[str, str] = {}
        for value in list(self._client_to_codex.values()):
            session_lookup[value.lower()] = value
        for session_id in list(self._session_mcp_servers.keys()):
            session_lookup.setdefault(session_id.lower(), session_id)

        removed_codex_session_ids: List[str] = []
        async with self._lock:
            for lowercase_id in normalized_targets:
                original_id = session_lookup.get(lowercase_id)
                if not original_id:
                    continue
                if self._clear_session_tracking(original_id):
                    removed_codex_session_ids.append(original_id)

        removed_artifacts = self._remove_specific_session_artifacts(normalized_targets)
        if removed_codex_session_ids or removed_artifacts:
            _LOG.info(
                json.dumps(
                    {
                        "message": "deleted_codex_sessions",
                        "removed_codex_session_ids": removed_codex_session_ids,
                        "removed_artifacts": removed_artifacts,
                    }
                )
            )
        return {
            "removed_codex_session_ids": removed_codex_session_ids,
            "removed_artifacts": removed_artifacts,
        }

    async def _set_session_model(self, codex_session_id: str, model_id: str) -> None:
        await self._process.request(
            "session/set_model",
            {
                "sessionId": codex_session_id,
                "modelId": model_id,
            },
        )
        _LOG.info(
            json.dumps(
                {
                    "message": "Updated session model",
                    "codex_session_id": codex_session_id,
                    "model_id": model_id,
                }
            )
        )

    async def _set_session_mode(self, codex_session_id: str, mode_id: str) -> None:
        await self._process.request(
            "session/set_mode",
            {
                "sessionId": codex_session_id,
                "modeId": mode_id,
            },
        )
        _LOG.info(
            json.dumps(
                {
                    "message": "Updated session mode",
                    "codex_session_id": codex_session_id,
                    "mode_id": mode_id,
                }
            )
        )

    async def _cancel_session_prompt(self, codex_session_id: str) -> None:
        try:
            await self._process.notify(
                "session/cancel",
                {
                    "sessionId": codex_session_id,
                },
            )
        except CodexError as exc:
            _LOG.warning(
                json.dumps(
                    {
                        "message": "session_cancel_failed",
                        "codex_session_id": codex_session_id,
                        "error": exc.to_dict(),
                    }
                )
            )
            raise
        else:
            _LOG.info(
                json.dumps(
                    {
                        "message": "session_cancelled",
                        "codex_session_id": codex_session_id,
                    }
                )
            )

    def _clear_session_tracking(self, codex_session_id: str) -> bool:
        removed = False
        for key, value in list(self._client_to_codex.items()):
            if value == codex_session_id:
                del self._client_to_codex[key]
                removed = True
        if codex_session_id in self._session_mcp_servers:
            self._session_mcp_servers.pop(codex_session_id, None)
            removed = True
        if codex_session_id in self._session_watchers:
            self._session_watchers.pop(codex_session_id, None)
            removed = True
        if codex_session_id in self._session_mode_map:
            self._session_mode_map.pop(codex_session_id, None)
            removed = True
        return removed

    def _extract_session_id(self, path: Path) -> Optional[str]:
        match = _SESSION_ID_PATTERN.search(path.name)
        if match:
            return match.group(0).lower()
        return None

    def _prune_session_files(self, keep_session_ids: Set[str]) -> List[str]:
        if not self._sessions_root.exists():
            return []

        removed_paths: List[str] = []

        if not keep_session_ids:
            removed_paths.extend(self._remove_all_session_artifacts())
            return removed_paths

        candidates = [
            path
            for path in self._sessions_root.rglob("*")
            if self._extract_session_id(path) is not None
        ]
        candidates.sort(key=lambda path: len(path.parts), reverse=True)

        for path in candidates:
            session_id = self._extract_session_id(path)
            if session_id is None or session_id in keep_session_ids:
                continue
            if not path.exists():
                continue
            try:
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
                removed_paths.append(str(path))
            except FileNotFoundError:
                continue
            except OSError as exc:
                _LOG.warning(
                    json.dumps(
                        {
                            "message": "session_file_prune_failed",
                            "path": str(path),
                            "error": str(exc),
                        }
                    )
                )
                continue

            if path.parent != self._sessions_root:
                self._cleanup_empty_parents(path.parent)

        self._remove_empty_descendants(self._sessions_root)
        return removed_paths

    def _remove_specific_session_artifacts(self, session_ids: Set[str]) -> List[str]:
        if not self._sessions_root.exists():
            return []

        removed_paths: List[str] = []
        candidates = [
            path
            for path in self._sessions_root.rglob("*")
            if self._extract_session_id(path) in session_ids
        ]
        candidates.sort(key=lambda path: len(path.parts), reverse=True)

        for path in candidates:
            if not path.exists():
                continue
            try:
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
                removed_paths.append(str(path))
            except FileNotFoundError:
                continue
            except OSError as exc:
                _LOG.warning(
                    json.dumps(
                        {
                            "message": "session_file_delete_failed",
                            "path": str(path),
                            "error": str(exc),
                        }
                    )
                )
                continue

            if path.parent != self._sessions_root:
                self._cleanup_empty_parents(path.parent)

        self._remove_empty_descendants(self._sessions_root)
        return removed_paths

    def _remove_all_session_artifacts(self) -> List[str]:
        removed: List[str] = []
        if not self._sessions_root.exists():
            return removed

        for child in sorted(self._sessions_root.iterdir(), key=lambda p: len(p.parts), reverse=True):
            try:
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
                removed.append(str(child))
            except FileNotFoundError:
                continue
            except OSError as exc:
                _LOG.warning(
                    json.dumps(
                        {
                            "message": "session_artifact_prune_failed",
                            "path": str(child),
                            "error": str(exc),
                        }
                    )
                )
        return removed

    def _cleanup_empty_parents(self, start: Path) -> None:
        current = start
        while current != self._sessions_root:
            try:
                entries = list(current.iterdir())
            except FileNotFoundError:
                return
            except OSError:
                return
            if entries:
                return
            try:
                current.rmdir()
            except OSError:
                return
            current = current.parent

    def _remove_empty_descendants(self, root: Path) -> None:
        for path in sorted(root.rglob("*"), key=lambda p: len(p.parts), reverse=True):
            if path.is_dir() and path != root:
                try:
                    path.rmdir()
                except OSError:
                    continue

    async def _handle_notification(self, method: str, params: Dict[str, Any]) -> None:
        if method != "session/update":
            _LOG.debug(json.dumps({"message": "Ignoring Codex notification", "method": method}))
            return

        session_id = str(params.get("sessionId", ""))
        if not session_id:
            return

        if _LOG.isEnabledFor(logging.DEBUG):
            _LOG.debug(
                json.dumps(
                    {
                        "message": "codex_notification",
                        "method": method,
                        "session_id": session_id,
                        "payload": params,
                    }
                )
            )

        watchers = list(self._session_watchers.get(session_id, []))
        if not watchers:
            return

        await self._preflight_apply_patch(session_id, params)
        await asyncio.gather(*(queue.put(params) for queue in watchers), return_exceptions=True)
        if _LOG.isEnabledFor(logging.DEBUG):
            _LOG.debug(
                json.dumps(
                    {
                        "message": "codex_notification",
                        "method": method,
                        "params": params,
                    }
                )
            )

    async def _preflight_apply_patch(self, session_id: str, params: Mapping[str, Any]) -> None:
        raw_inputs = self._extract_apply_patch_inputs(params)
        if not raw_inputs:
            if _LOG.isEnabledFor(logging.DEBUG):
                _LOG.debug(
                    json.dumps(
                        {
                            "message": "apply_patch_preflight_skip",
                            "reason": "no_apply_patch_payload",
                            "session_id": session_id,
                        }
                    )
                )
            return

        if _LOG.isEnabledFor(logging.DEBUG):
            _LOG.debug(
                json.dumps(
                    {
                        "message": "apply_patch_preflight_inputs",
                        "session_id": session_id,
                        "input_count": len(raw_inputs),
                    }
                )
            )

        paths_to_check: Set[Path] = set()
        for raw_input in raw_inputs:
            base_dir = self._resolve_apply_patch_base_dir(session_id, raw_input)
            if base_dir is None:
                if _LOG.isEnabledFor(logging.DEBUG):
                    _LOG.debug(
                        json.dumps(
                            {
                                "message": "apply_patch_preflight_skip_input",
                                "reason": "base_dir_outside_workspace",
                                "session_id": session_id,
                                "raw_input": raw_input,
                            }
                        )
                    )
                continue
            raw_changes = raw_input.get("changes")
            normalized_changes = self._normalize_apply_patch_changes(raw_changes)
            if not normalized_changes:
                if _LOG.isEnabledFor(logging.DEBUG):
                    _LOG.debug(
                        json.dumps(
                            {
                                "message": "apply_patch_preflight_skip_input",
                                "reason": "no_changes",
                                "session_id": session_id,
                                "raw_input": raw_input,
                            }
                        )
                    )
                continue
            for change in normalized_changes:
                if not isinstance(change, Mapping):
                    continue
                change_type = self._normalize_change_type(change)
                if change_type == "update":
                    self._accumulate_path(
                        session_id,
                        base_dir,
                        change,
                        ("path", "targetPath", "target_path"),
                        paths_to_check,
                    )
                elif change_type == "delete":
                    if self._change_allows_missing(change):
                        continue
                    self._accumulate_path(
                        session_id,
                        base_dir,
                        change,
                        ("path", "targetPath", "target_path"),
                        paths_to_check,
                    )
                elif change_type == "move":
                    self._accumulate_path(
                        session_id,
                        base_dir,
                        change,
                        (
                            "fromPath",
                            "sourcePath",
                            "oldPath",
                            "source",
                            "from_path",
                            "source_path",
                            "old_path",
                            "path",
                        ),
                        paths_to_check,
                    )
                    destination_path = self._resolve_apply_patch_path(
                        session_id,
                        base_dir,
                        change,
                        (
                            "toPath",
                            "destinationPath",
                            "newPath",
                            "destination",
                            "to_path",
                            "destination_path",
                            "new_path",
                        ),
                    )
                    if destination_path is not None:
                        parent = destination_path.parent
                        if parent != destination_path:
                            paths_to_check.add(parent)

        if not paths_to_check:
            return

        await self._ensure_paths_available(session_id, paths_to_check)

    def _extract_apply_patch_inputs(self, payload: Any) -> List[Mapping[str, Any]]:
        results: List[Mapping[str, Any]] = []
        stack: List[Any] = [payload]
        seen: Set[int] = set()

        while stack:
            current = stack.pop()
            if isinstance(current, Mapping):
                if self._is_apply_patch_payload(current) and id(current) not in seen:
                    seen.add(id(current))
                    results.append(current)
                raw_input = current.get("rawInput") or current.get("raw_input")
                if isinstance(raw_input, Mapping) and id(raw_input) not in seen:
                    if self._is_apply_patch_payload(raw_input):
                        seen.add(id(raw_input))
                        results.append(raw_input)
                stack.extend(
                    value
                    for value in current.values()
                    if isinstance(value, (Mapping, list, tuple))
                )
            elif isinstance(current, (list, tuple)):
                stack.extend(
                    item for item in current if isinstance(item, (Mapping, list, tuple))
                )

        return results

    @staticmethod
    def _is_apply_patch_payload(candidate: Mapping[str, Any]) -> bool:
        if not isinstance(candidate, Mapping):
            return False

        for key in ("kind", "tool", "toolName", "tool_name", "name"):
            value = candidate.get(key)
            if isinstance(value, str):
                normalized = value.strip().lower().replace("-", "_")
                if normalized == "apply_patch":
                    return True

        changes = candidate.get("changes")
        if isinstance(changes, list):
            for entry in changes:
                if isinstance(entry, Mapping) and any(
                    key in entry
                    for key in (
                        "path",
                        "fromPath",
                        "toPath",
                        "sourcePath",
                        "destinationPath",
                        "oldPath",
                        "newPath",
                        "from_path",
                        "to_path",
                        "source_path",
                        "destination_path",
                        "old_path",
                        "new_path",
                    )
                ):
                    return True
        if isinstance(changes, Mapping):
            for descriptor in changes.values():
                if isinstance(descriptor, Mapping):
                    if any(
                        key in descriptor
                        for key in (
                            "path",
                            "fromPath",
                            "toPath",
                            "sourcePath",
                            "destinationPath",
                            "oldPath",
                            "newPath",
                            "from_path",
                            "to_path",
                            "source_path",
                            "destination_path",
                            "old_path",
                            "new_path",
                        )
                    ):
                        return True
                    nested_update = descriptor.get("update") or descriptor.get("delete") or descriptor.get("add") or descriptor.get("move")
                    if isinstance(nested_update, Mapping):
                        return True
            if changes:
                return True

        return False

    def _resolve_apply_patch_base_dir(
        self, session_id: str, raw_input: Mapping[str, Any]
    ) -> Optional[Path]:
        base_dir_value = (
            raw_input.get("cwd")
            or raw_input.get("workingDir")
            or raw_input.get("workingDirectory")
        )
        if isinstance(base_dir_value, str) and base_dir_value.strip():
            candidate = Path(base_dir_value.strip())
            if candidate.is_absolute():
                resolved = candidate.resolve()
            else:
                resolved = (self._workspace_root / candidate).resolve()
        else:
            resolved = self._workspace_root

        try:
            resolved.relative_to(self._workspace_root)
        except ValueError:
            _LOG.warning(
                json.dumps(
                    {
                        "message": "apply_patch_preflight_base_outside_workspace",
                        "session_id": session_id,
                        "base_dir": str(resolved),
                        "workspace": str(self._workspace_root),
                    }
                )
            )
            return None

        return resolved

    def _accumulate_path(
        self,
        session_id: str,
        base_dir: Path,
        change: Mapping[str, Any],
        keys: Tuple[str, ...],
        accumulator: Set[Path],
    ) -> Optional[Path]:
        resolved = self._resolve_apply_patch_path(session_id, base_dir, change, keys)
        if resolved is not None:
            accumulator.add(resolved)
        return resolved

    def _resolve_apply_patch_path(
        self,
        session_id: str,
        base_dir: Path,
        change: Mapping[str, Any],
        keys: Tuple[str, ...],
    ) -> Optional[Path]:
        for key in keys:
            value = change.get(key)
            if isinstance(value, str):
                resolved = self._resolve_path_value(session_id, base_dir, value.strip())
                if resolved is not None:
                    return resolved
        return None

    def _resolve_path_value(
        self, session_id: str, base_dir: Path, value: str
    ) -> Optional[Path]:
        if not value:
            return None

        candidate = Path(value)
        if candidate.is_absolute():
            resolved = candidate.resolve()
        else:
            resolved = (base_dir / candidate).resolve()

        try:
            resolved.relative_to(self._workspace_root)
        except ValueError:
            _LOG.warning(
                json.dumps(
                    {
                        "message": "apply_patch_preflight_path_outside_workspace",
                        "session_id": session_id,
                        "requested": value,
                        "resolved": str(resolved),
                        "workspace": str(self._workspace_root),
                    }
                )
            )
            return None

        return resolved

    @staticmethod
    def _normalize_change_type(change: Mapping[str, Any]) -> str:
        raw_type = (
            change.get("type")
            or change.get("changeType")
            or change.get("change_type")
            or change.get("operation")
        )
        if isinstance(raw_type, str):
            return raw_type.strip().lower().replace("-", "_")
        return ""

    @staticmethod
    def _change_allows_missing(change: Mapping[str, Any]) -> bool:
        allow_value = change.get("allowMissing")
        if allow_value is None:
            allow_value = change.get("allow_missing")
        if isinstance(allow_value, bool):
            return allow_value
        if isinstance(allow_value, str):
            lowered = allow_value.strip().lower()
            return lowered in {"1", "true", "yes", "on"}
        if isinstance(allow_value, (int, float)):
            return bool(allow_value)
        return False

    async def _ensure_paths_available(self, session_id: str, paths: Set[Path]) -> None:
        if not paths:
            return

        timeout_seconds = 2.0
        poll_interval = 0.1
        missing = {path for path in paths if not path.exists()}
        if not missing:
            return

        if _LOG.isEnabledFor(logging.DEBUG):
            _LOG.debug(
                json.dumps(
                    {
                        "message": "apply_patch_preflight_wait",
                        "session_id": session_id,
                        "paths": sorted(str(path) for path in missing),
                        "timeout_seconds": timeout_seconds,
                    }
                )
            )

        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_seconds
        while missing and loop.time() < deadline:
            await asyncio.sleep(poll_interval)
            missing = {path for path in paths if not path.exists()}

        if missing:
            _LOG.warning(
                json.dumps(
                    {
                        "message": "apply_patch_preflight_timeout",
                        "session_id": session_id,
                        "paths": sorted(str(path) for path in missing),
                        "timeout_seconds": timeout_seconds,
                    }
                )
            )
        elif _LOG.isEnabledFor(logging.DEBUG):
            _LOG.debug(
                json.dumps(
                    {
                        "message": "apply_patch_preflight_ready",
                        "session_id": session_id,
                        "paths": sorted(str(path) for path in paths),
                    }
                )
            )

    def _normalize_apply_patch_changes(self, raw_changes: Any) -> List[Mapping[str, Any]]:
        if isinstance(raw_changes, list):
            return [change for change in raw_changes if isinstance(change, Mapping)]
        if isinstance(raw_changes, Mapping):
            normalized: List[Mapping[str, Any]] = []
            for path_key, descriptor in raw_changes.items():
                if not isinstance(path_key, str):
                    continue
                normalized.extend(self._normalize_change_mapping_entry(path_key, descriptor))
            return normalized
        return []

    def _normalize_change_mapping_entry(self, path_key: str, descriptor: Any) -> List[Mapping[str, Any]]:
        results: List[Mapping[str, Any]] = []
        if not isinstance(descriptor, Mapping):
            results.append({"path": path_key})
            return results

        base_allow_missing = descriptor.get("allowMissing")
        if base_allow_missing is None:
            base_allow_missing = descriptor.get("allow_missing")

        def with_defaults(payload: Mapping[str, Any], change_type: str) -> Mapping[str, Any]:
            normalized_payload: Dict[str, Any] = dict(payload)
            normalized_payload.setdefault("path", path_key)
            normalized_payload.setdefault("type", change_type)
            if base_allow_missing is not None and "allowMissing" not in normalized_payload and "allow_missing" not in normalized_payload:
                normalized_payload["allowMissing"] = base_allow_missing
            return normalized_payload

        update_payload = descriptor.get("update")
        if update_payload is not None:
            if isinstance(update_payload, Mapping):
                results.append(with_defaults(update_payload, "update"))
            else:
                results.append({"path": path_key, "type": "update"})

        add_payload = descriptor.get("add")
        if add_payload is not None:
            if isinstance(add_payload, Mapping):
                results.append(with_defaults(add_payload, "add"))
            else:
                results.append({"path": path_key, "type": "add"})

        delete_payload = descriptor.get("delete")
        if delete_payload is None:
            delete_payload = descriptor.get("remove")
        if delete_payload is not None:
            payload: Dict[str, Any] = {}
            if isinstance(delete_payload, Mapping):
                payload.update(delete_payload)
            elif isinstance(delete_payload, bool):
                payload["allowMissing"] = delete_payload
            results.append(with_defaults(payload, "delete"))

        move_payload = descriptor.get("move")
        if move_payload is not None:
            if isinstance(move_payload, Mapping):
                results.append(with_defaults(move_payload, "move"))
            else:
                results.append({"path": path_key, "type": "move"})

        rename_payload = descriptor.get("rename")
        if rename_payload is not None:
            if isinstance(rename_payload, Mapping):
                results.append(with_defaults(rename_payload, "move"))
            else:
                results.append({"path": path_key, "type": "move"})

        if not results:
            fallback_payload = dict(descriptor)
            fallback_type = fallback_payload.get("type", "update")
            results.append(with_defaults(fallback_payload, fallback_type))

        return results

    def _attach_watcher(self, session_id: str) -> asyncio.Queue[Dict[str, Any]]:
        queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()
        self._session_watchers.setdefault(session_id, []).append(queue)
        return queue

    def _detach_watcher(self, session_id: str, queue: asyncio.Queue[Dict[str, Any]]) -> None:
        watchers = self._session_watchers.get(session_id)
        if not watchers:
            return
        with contextlib.suppress(ValueError):
            watchers.remove(queue)
        if not watchers:
            self._session_watchers.pop(session_id, None)

    async def _prepare_prompt_context(
        self,
        token: Optional[str],
        client_session_id: Optional[str],
        model_id: Optional[str],
    ) -> Tuple[str, str, asyncio.Queue[Dict[str, Any]]]:
        client_session = client_session_id or ""
        await self._ensure_process_ready(token)
        codex_session = await self._ensure_session(CODEX_AGENT_NAME, client_session)
        if model_id:
            await self._set_session_model(codex_session, model_id)
        queue = self._attach_watcher(codex_session)
        return client_session, codex_session, queue

    def _log_prompt_dispatch(self, client_session: str, codex_session: str, block_count: int) -> None:
        _LOG.info(
            json.dumps(
                {
                    "message": "Dispatching prompt to codex-acp",
                    "session_id": client_session or None,
                    "codex_session_id": codex_session,
                    "content_blocks": block_count,
                }
            )
        )

    async def _consume_prompt_stream(
        self,
        codex_session: str,
        prompt_payload: List[Dict[str, Any]],
        queue: asyncio.Queue[Dict[str, Any]],
        accumulator: _PromptAccumulator,
    ) -> str:
        response_task = asyncio.create_task(
            self._process.request(
                "session/prompt",
                {
                    "sessionId": codex_session,
                    "prompt": prompt_payload,
                },
            )
        )
        queue_task: Optional[asyncio.Task[Dict[str, Any]]] = None
        stop_reason = "unknown"

        try:
            while True:
                if queue_task is None:
                    queue_task = asyncio.create_task(queue.get())
                done, _ = await asyncio.wait(
                    {response_task, queue_task}, return_when=asyncio.FIRST_COMPLETED
                )

                if queue_task in done:
                    accumulator.ingest(queue_task.result())
                    queue_task = None

                if response_task in done:
                    prompt_response = response_task.result()
                    stop_reason = (prompt_response or {}).get("stopReason", "unknown")
                    break
        finally:
            if queue_task:
                queue_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await queue_task
            if not response_task.done():
                response_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await response_task
        return stop_reason

    @staticmethod
    def _drain_prompt_queue(queue: asyncio.Queue[Dict[str, Any]], accumulator: _PromptAccumulator) -> None:
        while not queue.empty():
            accumulator.ingest(queue.get_nowait())

    def _log_prompt_completion(
        self,
        client_session: str,
        codex_session: str,
        stop_reason: str,
        accumulator: _PromptAccumulator,
        output_messages: List[Dict[str, Any]],
        events: List[Dict[str, Any]],
    ) -> None:
        _LOG.info(
            json.dumps(
                {
                    "message": "Prompt completed",
                    "session_id": client_session or None,
                    "codex_session_id": codex_session,
                    "stop_reason": stop_reason,
                    "assistant_tokens": accumulator.assistant_token_count,
                }
            )
        )
        if _LOG.isEnabledFor(logging.DEBUG):
            _LOG.debug(
                json.dumps(
                    {
                        "message": "prompt_stream_summary",
                        "codex_session_id": codex_session,
                        "output_messages": output_messages,
                        "events": events,
                    }
                )
            )

    async def _resolve_codex_session(self, token: Optional[str], session_identifier: str) -> str:
        effective_token = token if token is not None else self._process.current_token
        await self._ensure_process_ready(effective_token)
        if session_identifier in self._session_watchers:
            return session_identifier
        for (_, client_session), value in self._client_to_codex.items():
            if client_session == session_identifier:
                return value
        return await self._ensure_session(CODEX_AGENT_NAME, session_identifier)


class _PromptAccumulator:
    """Collects updates streaming from Codex and renders final output."""

    def __init__(self) -> None:
        self._assistant_text_parts: List[str] = []
        self.events: List[Dict[str, Any]] = []

    def ingest(self, notification: Dict[str, Any]) -> None:
        if not isinstance(notification, dict):
            return

        update = notification.get("update") or {}
        if not isinstance(update, dict):
            return

        kind = update.get("sessionUpdate")
        if kind == "agent_message_chunk":
            content = update.get("content") or {}
            if isinstance(content, dict) and content.get("type") == "text":
                text = content.get("text")
                if isinstance(text, str) and text:
                    self._assistant_text_parts.append(text)
        self.events.append(notification)

    def build_output(self) -> List[Dict[str, Any]]:
        if not self._assistant_text_parts:
            return []

        message = {
            "role": "assistant",
            "parts": [
                {
                    "content_type": "text/plain",
                    "content_encoding": "plain",
                    "content": "".join(self._assistant_text_parts),
                }
            ],
        }
        return [message]

    @property
    def assistant_token_count(self) -> int:
        return sum(len(part) for part in self._assistant_text_parts)


def _build_prompt_payload(
    messages: List[Dict[str, Any]],
    attachment_blocks: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    content_blocks: List[Dict[str, Any]] = []

    for message in messages:
        parts = message.get("parts") or []
        for part in parts:
            if not isinstance(part, dict):
                continue
            content_type = part.get("content_type") or part.get("type")
            if content_type not in {"text/plain", "text"}:
                continue
            text = part.get("content") or part.get("text") or ""
            if text:
                content_blocks.append({"type": "text", "text": text})

    if not content_blocks and not attachment_blocks:
        raise ValueError("Prompt payload did not contain any content")

    content_blocks.extend(attachment_blocks)
    return content_blocks
