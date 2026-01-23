"""Centralized path management."""

from pathlib import Path
from typing import Final

from pyprojroot import find_root, has_file

PROJECT_ROOT: Final[Path] = find_root(has_file("uv.lock"))
