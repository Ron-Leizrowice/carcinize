"""Base mixins for Rust-like types."""

from __future__ import annotations

from copy import deepcopy
from typing import Self


class RustType:
    """Mixin providing Rust-like behavior for types.

    Provides:
    - clone(): Create a deep copy of the value
    """

    __slots__ = ()

    def clone(self) -> Self:
        """Clone the value (deep copy)."""
        return deepcopy(self)
