"""Small, cross-process-safe file replacement helpers.

Windows readers can briefly hold a destination file without delete sharing,
which makes an otherwise atomic ``os.replace`` fail with ``PermissionError``.
Game state writers retry that transient condition and use a unique temporary
file so concurrent writers never contend for the same ``*.tmp`` path.
"""

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path


def atomic_write_text(
    path: str | os.PathLike[str],
    text: str,
    *,
    encoding: str = "utf-8",
) -> None:
    """Atomically replace *path*, tolerating transient Windows share locks."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        temporary.write_text(text, encoding=encoding)
        for attempt in range(8):
            try:
                os.replace(temporary, destination)
                return
            except PermissionError:
                if attempt == 7:
                    raise
                time.sleep(0.02 * (2**attempt))
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
