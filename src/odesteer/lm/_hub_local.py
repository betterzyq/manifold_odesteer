"""
将 Hugging Face repo id 解析为 hub 缓存下的本地 snapshot 目录，避免误走 Hub。
支持 refs/main 为 commit SHA、字面 ``main``、以及不完整 hash 目录时回退到 ``snapshots/main``。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def repo_id_to_hub_folder_name(repo_id: str) -> str:
    return "models--" + repo_id.replace("/", "--")


def _read_ref_revision(refs_dir: Path, name: str) -> str | None:
    p = refs_dir / name
    if not p.is_file():
        return None
    try:
        text = p.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return text or None


def _load_safetensors_weight_shards(index_path: Path) -> list[str] | None:
    try:
        data = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    wm = data.get("weight_map")
    if not isinstance(wm, dict) or not wm:
        return None
    return list(dict.fromkeys(str(v) for v in wm.values()))


def snapshot_has_complete_weights(snapshot: Path) -> bool:
    if not (snapshot / "config.json").is_file():
        return False

    index = snapshot / "model.safetensors.index.json"
    if index.is_file():
        shards = _load_safetensors_weight_shards(index)
        if shards:
            for name in shards:
                sp = snapshot / name
                if not sp.is_file() or sp.stat().st_size < 1:
                    return False
            return True

    single = snapshot / "model.safetensors"
    if single.is_file() and single.stat().st_size > 1_000_000:
        return True

    consolidated = snapshot / "consolidated.safetensors"
    if consolidated.is_file() and consolidated.stat().st_size > 500_000_000:
        return True

    shards = list(snapshot.glob("model-*-of-*.safetensors")) + list(
        snapshot.glob("pytorch_model-*-of-*.bin")
    )
    if shards and any(s.stat().st_size > 50_000_000 for s in shards):
        return True

    # GPT-2 系：单文件 pytorch_model.bin
    legacy = snapshot / "pytorch_model.bin"
    if legacy.is_file() and legacy.stat().st_size > 100_000_000:
        return True

    return False


def resolve_hub_snapshot_dir(repo_id: str, hub_cache: Path | None) -> Path | None:
    if hub_cache is None or not hub_cache.is_dir():
        return None
    if "/" not in repo_id or Path(repo_id).exists():
        return None

    folder = hub_cache / repo_id_to_hub_folder_name(repo_id)
    if not folder.is_dir():
        return None

    snapshots_root = folder / "snapshots"
    if not snapshots_root.is_dir():
        return None

    refs_dir = folder / "refs"
    candidates: list[Path] = []

    for ref_name in ("main", "master"):
        rev = _read_ref_revision(refs_dir, ref_name) if refs_dir.is_dir() else None
        if rev:
            candidates.append(snapshots_root / rev)

    main_snap = snapshots_root / "main"
    if main_snap.is_dir():
        candidates.append(main_snap)

    seen: set[str] = set()
    ordered: list[Path] = []
    for c in candidates:
        key = str(c.resolve())
        if key not in seen:
            seen.add(key)
            ordered.append(c)

    for cand in ordered:
        if cand.is_dir() and snapshot_has_complete_weights(cand):
            return cand.resolve()

    try:
        subs = sorted(p for p in snapshots_root.iterdir() if p.is_dir())
    except OSError:
        return None
    for sub in subs:
        if snapshot_has_complete_weights(sub):
            return sub.resolve()

    return None


def get_hub_cache_dir_str() -> str | None:
    """HF 模型缓存的 ``hub`` 目录（``models--*`` 所在层）。优先环境变量，否则探测常见本地路径。"""
    c = os.environ.get("HF_HUB_CACHE")
    if c:
        return c
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return str(Path(hf_home) / "hub")
    # 未配置 HF_HOME / HF_HUB_CACHE 时：数据盘（AutoDL 等）与默认 ~/.cache
    for candidate in (
        Path("/root/autodl-tmp/.cache/huggingface/hub"),
        Path.home() / ".cache" / "huggingface" / "hub",
    ):
        if candidate.is_dir():
            return str(candidate.resolve())
    return None


def maybe_resolve_pretrained_to_local_path(
    pretrained_name: str,
    hub_cache: Path | str | None,
) -> str:
    if hub_cache is None:
        hcs = get_hub_cache_dir_str()
        hub_cache = Path(hcs) if hcs else None

    if hub_cache is None or not Path(hub_cache).is_dir():
        return pretrained_name

    resolved = resolve_hub_snapshot_dir(pretrained_name, Path(hub_cache))
    if resolved is None:
        return pretrained_name

    if os.environ.get("ODESteer_DEBUG_HUB", "0") == "1":
        print(
            f"[ODESteer] Resolved Hub repo {pretrained_name!r} -> local snapshot {resolved}",
            flush=True,
        )
    return str(resolved)


def _is_resolved_local_model_dir(path: str) -> bool:
    p = Path(path)
    return p.is_dir() and (p / "config.json").is_file()


def hf_pretrained_hub_kwargs(model_id: str) -> tuple[str, dict]:
    """
    返回 ``(pretrained_name_or_path, kwargs)`` 供 ``from_pretrained`` 使用。

    - 加载项目根 ``.env``，使用 ``HF_HOME/hub`` 解析本地 snapshot。
    - 若已解析到带 ``config.json`` 的本地目录，则 ``local_files_only=True``，**不再访问 Hub**。
    - 另可通过 ``HF_HUB_OFFLINE`` / ``ODESteer_OFFLINE`` / ``ODESteer_LOCAL_ONLY=1`` 强制仅本地。
    """
    from ..utils.env import load_project_dotenv

    load_project_dotenv(_PROJECT_ROOT, override=False)
    cache_dir_str = get_hub_cache_dir_str()
    hub_cache = Path(cache_dir_str) if cache_dir_str else None

    pretrained = model_id
    if not Path(model_id).exists():
        pretrained = maybe_resolve_pretrained_to_local_path(model_id, hub_cache)
    if Path(pretrained).exists():
        pretrained = str(Path(pretrained).resolve())

    force_local = (
        os.environ.get("HF_HUB_OFFLINE", "0") == "1"
        or os.environ.get("ODESteer_OFFLINE", "0") == "1"
        or os.environ.get("ODESteer_LOCAL_ONLY", "0") == "1"
        or _is_resolved_local_model_dir(pretrained)
    )

    extra: dict = {"local_files_only": force_local}
    if cache_dir_str:
        extra["cache_dir"] = cache_dir_str
    return pretrained, extra
