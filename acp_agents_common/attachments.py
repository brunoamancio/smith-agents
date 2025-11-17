from __future__ import annotations

import base64
import binascii
import os
import re
import uuid
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

DEFAULT_MAX_INLINE_BYTES = int(os.getenv("ACPA_ATTACHMENT_MAX_INLINE_BYTES", 5 * 1024 * 1024))

ATTACHMENT_TYPE_IMAGE = "image"
ATTACHMENT_TYPE_RESOURCE = "resource"
ATTACHMENT_TYPE_RESOURCE_LINK = "resource_link"
ATTACHMENT_MIME_FIELD = "mimeType"
ATTACHMENT_URI_FIELD = "uri"
ATTACHMENT_NAME_FIELD = "name"
PROMPT_CAPABILITIES_KEY = "prompt_capabilities"
PROMPT_CAPABILITY_IMAGE = "image"
PROMPT_CAPABILITY_EMBEDDED_CONTEXT = "embeddedContext"
ATTACHMENT_METADATA_KEY = "attachments"
ATTACHMENT_METADATA_INLINE_LIMIT_KEY = "inlineLimitBytes"


class AttachmentError(Exception):
    """Raised when attachment payloads are invalid."""

    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


class AttachmentPayloadKind(str, Enum):
    IMAGE_BASE64 = "image_base64"
    IMAGE_URL = "image_url"
    RESOURCE_INLINE_TEXT = "resource_inline_text"
    RESOURCE_INLINE_BASE64 = "resource_inline_base64"
    RESOURCE_LINK = "resource_link"


class Attachment(BaseModel):
    id: Optional[str] = None
    display_name: Optional[str] = Field(default=None, alias="display_name")
    mime_type: Optional[str] = Field(default=None, alias="mime_type")
    size: Optional[int] = None
    payload_kind: AttachmentPayloadKind = Field(alias="payload_kind")
    data: Optional[str] = None
    text: Optional[str] = None
    uri: Optional[str] = None

    class Config:
        populate_by_name = True


def _decode_base64_size(label: str, value: str) -> int:
    try:
        return len(base64.b64decode(value, validate=True))
    except binascii.Error as exc:  # pragma: no cover - defensive
        raise AttachmentError(
            f"Attachment '{label}' contains invalid base64 data",
            status_code=422,
        ) from exc


def _ensure_inline_size(label: str, size: int, limit: int) -> None:
    if size > limit:
        raise AttachmentError(
            (
                f"Attachment '{label}' exceeds the maximum inline size of "
                f"{limit} bytes"
            ),
            status_code=413,
        )


def _safe_inline_uri(source: Attachment, index: int) -> str:
    if source.uri:
        return source.uri
    candidate = source.id or source.display_name
    if candidate:
        slug = re.sub(r"[^a-zA-Z0-9._-]", "-", candidate.strip())
        if slug:
            return f"urn:smith:inline:{slug}"
    return f"urn:smith:inline:{uuid.uuid4()}"


def normalize_attachments(
    attachments: List[Attachment],
    inline_limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Convert attachment definitions into ACP prompt content blocks."""

    if not attachments:
        return []

    limit = inline_limit if inline_limit is not None else DEFAULT_MAX_INLINE_BYTES

    normalized: List[Dict[str, Any]] = []

    for index, attachment in enumerate(attachments):
        label = attachment.display_name or attachment.id or f"attachment-{index + 1}"
        kind = attachment.payload_kind

        if kind is AttachmentPayloadKind.IMAGE_BASE64:
            if not attachment.data:
                raise AttachmentError(
                    f"Attachment '{label}' is missing base64 image data",
                    status_code=422,
                )
            size = _decode_base64_size(label, attachment.data)
            _ensure_inline_size(label, size, limit)
            block: Dict[str, Any] = {
                "type": ATTACHMENT_TYPE_IMAGE,
                "data": attachment.data,
                ATTACHMENT_MIME_FIELD: attachment.mime_type or "image/png",
            }
            if attachment.uri:
                block[ATTACHMENT_URI_FIELD] = attachment.uri
            normalized.append(block)
            continue

        if kind is AttachmentPayloadKind.IMAGE_URL:
            if not attachment.uri:
                raise AttachmentError(
                    f"Attachment '{label}' is missing an image URI",
                    status_code=422,
                )
            block = {
                "type": ATTACHMENT_TYPE_IMAGE,
                ATTACHMENT_URI_FIELD: attachment.uri,
            }
            if attachment.mime_type:
                block[ATTACHMENT_MIME_FIELD] = attachment.mime_type
            normalized.append(block)
            continue

        if kind is AttachmentPayloadKind.RESOURCE_INLINE_TEXT:
            if attachment.text is None:
                raise AttachmentError(
                    f"Attachment '{label}' is missing inline text content",
                    status_code=422,
                )
            inline_uri = _safe_inline_uri(attachment, index)
            block = {
                "type": ATTACHMENT_TYPE_RESOURCE,
                "resource": {
                    "type": "text",
                    ATTACHMENT_URI_FIELD: inline_uri,
                    "text": attachment.text,
                    ATTACHMENT_MIME_FIELD: attachment.mime_type or "text/plain",
                },
            }
            normalized.append(block)
            continue

        if kind is AttachmentPayloadKind.RESOURCE_INLINE_BASE64:
            if not attachment.data:
                raise AttachmentError(
                    f"Attachment '{label}' is missing inline resource data",
                    status_code=422,
                )
            size = _decode_base64_size(label, attachment.data)
            _ensure_inline_size(label, size, limit)
            inline_uri = _safe_inline_uri(attachment, index)
            block = {
                "type": ATTACHMENT_TYPE_RESOURCE,
                "resource": {
                    "type": "blob",
                    ATTACHMENT_URI_FIELD: inline_uri,
                    "blob": attachment.data,
                    ATTACHMENT_MIME_FIELD: attachment.mime_type or "application/octet-stream",
                },
            }
            normalized.append(block)
            continue

        if kind is AttachmentPayloadKind.RESOURCE_LINK:
            if not attachment.uri:
                raise AttachmentError(
                    f"Attachment '{label}' is missing a resource URI",
                    status_code=422,
                )
            block = {
                "type": ATTACHMENT_TYPE_RESOURCE_LINK,
                ATTACHMENT_URI_FIELD: attachment.uri,
                ATTACHMENT_NAME_FIELD: attachment.display_name or attachment.id or attachment.uri,
            }
            if attachment.mime_type:
                block[ATTACHMENT_MIME_FIELD] = attachment.mime_type
            if attachment.size is not None:
                block["size"] = attachment.size
            normalized.append(block)
            continue

        raise AttachmentError(
            f"Attachment '{label}' has unsupported payload kind '{kind.value}'",
            status_code=422,
        )

    return normalized


__all__ = [
    "Attachment",
    "AttachmentPayloadKind",
    "AttachmentError",
    "normalize_attachments",
    "DEFAULT_MAX_INLINE_BYTES",
    "ATTACHMENT_TYPE_IMAGE",
    "ATTACHMENT_TYPE_RESOURCE",
    "ATTACHMENT_TYPE_RESOURCE_LINK",
    "ATTACHMENT_MIME_FIELD",
    "ATTACHMENT_URI_FIELD",
    "ATTACHMENT_NAME_FIELD",
    "PROMPT_CAPABILITIES_KEY",
    "PROMPT_CAPABILITY_IMAGE",
    "PROMPT_CAPABILITY_EMBEDDED_CONTEXT",
    "ATTACHMENT_METADATA_KEY",
    "ATTACHMENT_METADATA_INLINE_LIMIT_KEY",
]
