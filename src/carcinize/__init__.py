"""Carcinize - Rust-like types for Python.

This package provides Rust-inspired data types for Python:

- Result: A type representing either success (Ok) or failure (Err)
- Option: A type representing an optional value (Some or Nothing)
- Struct/MutStruct: Pydantic-based structs with Rust-like semantics
- Iter: Fluent iterator with chainable combinators
- Lazy/OnceCell: Thread-safe lazy initialization primitives
"""

from carcinize.exceptions import UnwrapError
from carcinize.iter import Iter
from carcinize.lazy import Lazy, OnceCell, OnceCellAlreadyInitializedError
from carcinize.option import Nothing, Option, Some
from carcinize.result import Err, Ok, Result
from carcinize.struct import MutStruct, Struct

__all__ = [
    "Err",
    "Iter",
    "Lazy",
    "MutStruct",
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
