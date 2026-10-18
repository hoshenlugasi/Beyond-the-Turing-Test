"""Git-independent source fingerprints for reproducible experiments."""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT_SOURCE_FILES = (
    "llm_interface.py",
    "stt_main.py",
    "web_player.py",
    "evaluate_agent_performance.py",
    "prepare_config.py",
    "scripts/live_experiment_runner.py",
    "scripts/generate_deployment_manifest.py",
    "requirements-server.txt",
    "requirements.txt",
    "Dockerfile",
    "docker-compose.prod.yml",
    "generate_hackathon_dashboard.py",
    "hackathon_dashboard.html",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_source_bytes(path: Path) -> bytes:
    """Return source bytes with platform checkout line endings normalized.

    Git may materialize the same text as CRLF on Windows and LF in the Linux
    deployment checkout.  Line-ending representation is not agent behavior, so
    hashing raw checkout bytes made an otherwise identical release fail closed.
    All fingerprint candidates are text source/configuration files.
    """
    return path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def _candidate_files(root: Path) -> list[Path]:
    candidates: list[Path] = []
    for directory in (root / "llm_players", root / "core"):
        if directory.exists():
            candidates.extend(
                path for path in directory.rglob("*.py")
                if "__pycache__" not in path.parts
            )
    for name in ROOT_SOURCE_FILES:
        path = root / name
        if path.exists():
            candidates.append(path)
    config_dir = root / "configurations"
    if config_dir.exists():
        candidates.extend(
            path for path in config_dir.rglob("*.json")
            # ``web_*`` and ``agents_*`` are generated per-run configs and are
            # gitignored.  They are neither release source nor part of the
            # frozen publication preset, and may exist only on the machine
            # where a manifest happens to be generated.
            if not path.name.startswith(("web_", "agents_", "_"))
        )
    template_dir = root / "templates"
    if template_dir.exists():
        candidates.extend(template_dir.rglob("*.html"))
    return sorted(
        set(candidates), key=lambda path: path.relative_to(root).as_posix()
    )


def relevant_source_fingerprint(project_root: Path | None = None) -> dict[str, Any]:
    """Hash all source/config/dependency bytes that can affect a live game."""
    root = Path(project_root or Path(__file__).resolve().parent.parent).resolve()
    digest = hashlib.sha256()
    files: list[dict[str, Any]] = []
    for path in _candidate_files(root):
        relative = path.relative_to(root).as_posix()
        data = _canonical_source_bytes(path)
        file_hash = hashlib.sha256(data).hexdigest()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(data)
        digest.update(b"\0")
        files.append({"path": relative, "sha256": file_hash, "size": len(data)})

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
            text=True, encoding="utf-8", errors="replace", timeout=5,
            check=False,
        )
        commit = result.stdout.strip() if result.returncode == 0 else "unknown"
        commit = commit or "unknown"
    except Exception:
        commit = "unknown"

    return {
        "schema_version": 1,
        "sha256": digest.hexdigest(),
        "file_count": len(files),
        "git_commit": commit,
        "captured_at": _utc_now(),
        "files": files,
    }


def load_deployment_manifest(project_root: Path | None = None) -> dict[str, Any] | None:
    root = Path(project_root or Path(__file__).resolve().parent.parent).resolve()
    path = root / "deployment_manifest.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    return data if isinstance(data, dict) else None


def deployment_manifest_status(project_root: Path | None = None) -> dict[str, Any]:
    """Compare current source to the checked-in deployment manifest."""
    current = relevant_source_fingerprint(project_root)
    manifest = load_deployment_manifest(project_root)
    expected = (manifest or {}).get("source_fingerprint_sha256")
    expected_files = {
        item.get("path"): item.get("sha256")
        for item in ((manifest or {}).get("source_files") or [])
        if isinstance(item, dict) and item.get("path")
    }
    actual_files = {
        item["path"]: item["sha256"] for item in current["files"]
    }
    file_differences = {
        "changed": sorted(
            path for path in expected_files.keys() & actual_files.keys()
            if expected_files[path] != actual_files[path]
        ),
        "missing": sorted(expected_files.keys() - actual_files.keys()),
        "unexpected": sorted(actual_files.keys() - expected_files.keys()),
    }
    return {
        "ok": bool(expected) and expected == current["sha256"],
        "expected_sha256": expected,
        "actual_sha256": current["sha256"],
        "manifest": manifest,
        "fingerprint": current,
        "file_differences": file_differences,
        "reason": (
            "ok" if expected and expected == current["sha256"]
            else "deployment manifest missing" if not manifest
            else "deployment manifest has no source fingerprint" if not expected
            else "source fingerprint mismatch"
        ),
    }
