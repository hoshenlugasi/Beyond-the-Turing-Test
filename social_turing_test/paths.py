"""Packaged resources and private local runtime locations."""

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = PACKAGE_ROOT.parent
CONFIGURATIONS = PACKAGE_ROOT / "configurations"
