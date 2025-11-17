from pathlib import Path


def discover_services_root(anchor: Path) -> Path:
    """Return the nearest ancestor containing acp_agents_common."""
    for candidate in (anchor, *anchor.parents):
        if (candidate / "acp_agents_common").exists():
            return candidate
    return anchor
