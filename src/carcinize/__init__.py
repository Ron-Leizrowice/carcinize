"""Carcinize - Rust-like types for Python.

This package provides Rust-inspired data types for Python:

- Result: A type representing either success (Ok) or failure (Err)
- Option: A type representing an optional value (Some or Nothing)
- Struct/FrozenStruct: Pydantic-based structs with Rust-like semantics
- Iter: Fluent iterator with chainable combinators
- Lazy/OnceCell: Thread-safe lazy initialization primitives
"""

from carcinize._exceptions import UnwrapError
from carcinize._iter import Iter
from carcinize._lazy import Lazy, OnceCell, OnceCellAlreadyInitializedError
from carcinize._option import Nothing, Option, Some
from carcinize._result import Err, Ok, Result
from carcinize._struct import Struct

__all__ = [
    "Err",
    "Iter",
    "Lazy",
    "Nothing",
    "Ok",
    "OnceCell",
    "OnceCellAlreadyInitializedError",
    "Option",
    "Result",
    "Some",
    "Struct",
    "UnwrapError",
]
