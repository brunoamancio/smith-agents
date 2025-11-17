import base64

import pytest

from .attachments import (
    DEFAULT_MAX_INLINE_BYTES,
    Attachment,
    AttachmentError,
    AttachmentPayloadKind,
    normalize_attachments,
)


def test_normalize_base64_image() -> None:
    payload = Attachment(
        display_name="snapshot.png",
        mime_type="image/png",
        payload_kind=AttachmentPayloadKind.IMAGE_BASE64,
        data=base64.b64encode(b"png-bytes").decode("ascii"),
    )

    blocks = normalize_attachments([payload])
    assert len(blocks) == 1
    block = blocks[0]
    assert block["type"] == "image"
    assert block["mimeType"] == "image/png"
    assert block["data"] == payload.data


def test_normalize_inline_text_defaults_mime() -> None:
    payload = Attachment(
        display_name="notes.txt",
        payload_kind=AttachmentPayloadKind.RESOURCE_INLINE_TEXT,
        text="Example content",
    )

    blocks = normalize_attachments([payload])
    assert len(blocks) == 1
    block = blocks[0]
    resource = block["resource"]
    assert resource["type"] == "text"
    assert resource["text"] == "Example content"
    assert resource["mimeType"] == "text/plain"


def test_normalize_rejects_oversized_base64() -> None:
    raw = b"z" * (DEFAULT_MAX_INLINE_BYTES + 1)
    payload = Attachment(
        display_name="big.bin",
        payload_kind=AttachmentPayloadKind.RESOURCE_INLINE_BASE64,
        data=base64.b64encode(raw).decode("ascii"),
    )

    with pytest.raises(AttachmentError) as exc_info:
        normalize_attachments([payload])

    assert exc_info.value.status_code == 413
