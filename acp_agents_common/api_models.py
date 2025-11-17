from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .attachments import Attachment
from .mode_utils import (
    AGENT_NAME_KEY,
    CLIENT_SESSION_ID_KEY,
    MODE_ID_KEY,
    MODEL_ID_KEY,
    SESSION_ID_KEY,
)


class _AliasConfig:
    populate_by_name = True


class SessionNewRequest(BaseModel):
    agent_name: str = Field(alias=AGENT_NAME_KEY)
    client_session_id: str = Field(alias=CLIENT_SESSION_ID_KEY)
    mcpServers: Optional[List[Dict[str, Any]]] = None

    class Config(_AliasConfig):
        pass


class SessionLoadRequest(BaseModel):
    agent_name: str = Field(alias=AGENT_NAME_KEY)
    session_id: str = Field(alias=SESSION_ID_KEY)
    client_session_id: Optional[str] = Field(default=None, alias=CLIENT_SESSION_ID_KEY)
    mcpServers: Optional[List[Dict[str, Any]]] = None

    class Config(_AliasConfig):
        pass


class SessionModelRequest(BaseModel):
    agent_name: str = Field(alias=AGENT_NAME_KEY)
    session_id: str = Field(alias=SESSION_ID_KEY)
    model_id: str = Field(alias=MODEL_ID_KEY)
    client_session_id: Optional[str] = Field(default=None, alias=CLIENT_SESSION_ID_KEY)

    class Config(_AliasConfig):
        pass


class SessionModeRequest(BaseModel):
    agent_name: str = Field(alias=AGENT_NAME_KEY)
    session_id: str = Field(alias=SESSION_ID_KEY)
    mode_id: str = Field(alias=MODE_ID_KEY)
    client_session_id: Optional[str] = Field(default=None, alias=CLIENT_SESSION_ID_KEY)

    class Config(_AliasConfig):
        pass


class SessionCancelRequest(BaseModel):
    agent_name: str = Field(alias=AGENT_NAME_KEY)
    session_id: str = Field(alias=SESSION_ID_KEY)
    client_session_id: Optional[str] = Field(default=None, alias=CLIENT_SESSION_ID_KEY)

    class Config(_AliasConfig):
        pass


class AgentDescribeRequest(BaseModel):
    agent_name: str = Field(alias=AGENT_NAME_KEY)

    class Config(_AliasConfig):
        pass


class SessionPruneRequest(BaseModel):
    keep_session_ids: List[str] = Field(default_factory=list)

    class Config(_AliasConfig):
        pass


class SessionDeleteRequest(BaseModel):
    session_ids: List[str] = Field(default_factory=list)

    class Config(_AliasConfig):
        pass


class RunRequest(BaseModel):
    agent_name: str = Field(alias=AGENT_NAME_KEY)
    mode: str
    session_id: Optional[str] = Field(default=None, alias=SESSION_ID_KEY)
    model_id: Optional[str] = Field(default=None, alias=MODEL_ID_KEY)
    input: List[Dict[str, Any]]
    attachments: List[Attachment] = Field(default_factory=list)

    class Config(_AliasConfig):
        pass
