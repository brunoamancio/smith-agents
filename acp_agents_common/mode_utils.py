from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

MODE_READ_ONLY = "read-only"
MODE_AUTO = "auto"
MODE_FULL_ACCESS = "full-access"

AGENT_NAME_KEY = "agent_name"
CLIENT_SESSION_ID_KEY = "client_session_id"
SESSION_ID_KEY = "session_id"
MODEL_ID_KEY = "model_id"
MODE_ID_KEY = "mode_id"
MODELS_BLOCK_KEY = "models"
MODES_BLOCK_KEY = "modes"
RUN_ID_KEY = "run_id"
RUN_STATUS_KEY = "status"
RUN_STOP_REASON_KEY = "stop_reason"
RUN_OUTPUT_KEY = "output"
RUN_EVENTS_KEY = "events"

MODE_RESPONSE_KEYS = {
    "current_mode_id": "current_mode_id",
    "available_modes": "available_modes",
    "mode_id": "id",
    "mode_label": "label",
    "mode_description": "description",
    "mode_capabilities": "capabilities",
    "mode_meta": "meta",
}

@dataclass
class ModeCapabilities:
    filesystem_read: bool = False
    filesystem_write: bool = False
    terminal_exec: bool = False
    network_access: bool = False
    auto_apply: bool = False

    def to_payload(self) -> Dict[str, Any]:
        return {
            "filesystem": {
                "read": self.filesystem_read,
                "write": self.filesystem_write,
            },
            "terminal": {
                "exec": self.terminal_exec,
            },
            "network": {
                "access": self.network_access,
            },
            "auto_apply": self.auto_apply,
        }

    @classmethod
    def from_payload(cls, payload: Optional[Dict[str, Any]]) -> "ModeCapabilities":
        if not isinstance(payload, dict):
            return cls()
        filesystem = payload.get("filesystem", {})
        terminal = payload.get("terminal", {})
        network = payload.get("network", {})
        return cls(
            filesystem_read=bool(filesystem.get("read")),
            filesystem_write=bool(filesystem.get("write")),
            terminal_exec=bool(terminal.get("exec")),
            network_access=bool(network.get("access")),
            auto_apply=bool(payload.get("auto_apply")),
        )


def canonicalize_modes(
    modes_state: Optional[Dict[str, Any]],
    presets: Dict[str, ModeCapabilities],
    vendor: str,
) -> Dict[str, Any]:
    if not isinstance(modes_state, dict):
        current = None
        available_raw: Iterable[Dict[str, Any]] = []
    else:
        current = modes_state.get("current_mode_id") or modes_state.get("currentModeId")
        available_raw = modes_state.get("available_modes") or modes_state.get("availableModes") or []

    available: List[Dict[str, Any]] = []
    available_ids = set()
    for entry in available_raw:
        if not isinstance(entry, dict):
            continue
        mode_id = entry.get("id") or entry.get("modeId")
        if not isinstance(mode_id, str) or not mode_id:
            continue
        available_ids.add(mode_id)
        label = entry.get("label") or entry.get("name") or mode_id
        description = entry.get("description")
        preset_capabilities = presets.get(mode_id, ModeCapabilities())
        capabilities = preset_capabilities.to_payload()
        meta_payload = {"vendor": vendor, "vendor_mode_id": mode_id}
        if isinstance(entry.get("meta"), dict):
            meta_payload.update({k: v for k, v in entry["meta"].items() if k not in meta_payload})
        available.append(
            {
                MODE_RESPONSE_KEYS["mode_id"]: mode_id,
                MODE_RESPONSE_KEYS["mode_label"]: label,
                MODE_RESPONSE_KEYS["mode_description"]: description,
                MODE_RESPONSE_KEYS["mode_capabilities"]: capabilities,
                MODE_RESPONSE_KEYS["mode_meta"]: meta_payload,
            }
        )

    if current not in available_ids:
        current = next(iter(available_ids), None)

    return {
        MODE_RESPONSE_KEYS["current_mode_id"]: current,
        MODE_RESPONSE_KEYS["available_modes"]: available,
    }


def canonicalize_models(models_state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(models_state, dict):
        current = None
        available_raw: Iterable[Dict[str, Any]] = []
    else:
        current = models_state.get("current_model_id") or models_state.get("currentModelId")
        available_raw = models_state.get("available_models") or models_state.get("availableModels") or []

    available: List[Dict[str, Any]] = []
    available_ids = set()
    for entry in available_raw:
        if not isinstance(entry, dict):
            continue
        model_id = entry.get("model_id") or entry.get("modelId")
        if not isinstance(model_id, str) or not model_id:
            continue
        available_ids.add(model_id)
        name = entry.get("name") or model_id
        description = entry.get("description")
        available.append(
            {
                "model_id": model_id,
                "name": name,
                "description": description,
            }
        )

    if current not in available_ids:
        current = next(iter(available_ids), None)

    return {
        "current_model_id": current,
        "available_models": available,
    }

