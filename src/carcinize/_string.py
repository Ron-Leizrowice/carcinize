"""Rust-like string type.

Work in progress, not yet publicly exported.

"""

from carcinize._base import RustType


class String(str, RustType):
    """A string type that owns its own value.

    Inherits from both str and RustType, providing:
    - All standard str methods
    - clone() for explicit copying
    """

    __slots__ = ()
