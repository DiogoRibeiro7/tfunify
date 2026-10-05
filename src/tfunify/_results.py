"""Tuple-like access to the arrays a result unpacks into."""

from __future__ import annotations

from collections.abc import Iterator
from typing import ClassVar, overload

from ._validation import FloatArray


class Unpackable:
    """Lets a result be unpacked, indexed and measured like the tuple it replaces.

    The systems used to return plain tuples. A result object keeps that
    protocol for the arrays named in `_unpacked`, so `pnl, weights = result`,
    `result[0]` and `len(result)` all work, while further arrays are reached by
    name only.
    """

    _unpacked: ClassVar[tuple[str, ...]]

    def _as_tuple(self) -> tuple[FloatArray, ...]:
        return tuple(getattr(self, name) for name in self._unpacked)

    def __iter__(self) -> Iterator[FloatArray]:
        return iter(self._as_tuple())

    def __len__(self) -> int:
        return len(self._unpacked)

    @overload
    def __getitem__(self, index: int) -> FloatArray: ...
    @overload
    def __getitem__(self, index: slice) -> tuple[FloatArray, ...]: ...
    def __getitem__(self, index: int | slice) -> FloatArray | tuple[FloatArray, ...]:
        return self._as_tuple()[index]
