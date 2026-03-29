import os
from pathlib import Path


def _parse_dotenv_line(line: str) -> tuple[str, str] | None:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    if "=" not in line:
        return None
    key, value = line.split("=", 1)
    key = key.strip()
    value = value.strip()
    if not key:
        return None
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        value = value[1:-1]
    return key, value


def load_project_dotenv(project_dir: Path, override: bool = False) -> None:
    """Load ``<project_dir>/.env`` into ``os.environ`` (override=False 时保留已有变量)."""
    env_path = project_dir / ".env"
    if not env_path.exists():
        return
    try:
        content = env_path.read_text(encoding="utf-8")
    except OSError:
        return
    for raw_line in content.splitlines():
        parsed = _parse_dotenv_line(raw_line)
        if parsed is None:
            continue
        key, value = parsed
        if override or key not in os.environ:
            os.environ[key] = value
