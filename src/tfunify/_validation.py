"""Argument checks shared by the public functions.

Every public function validates its inputs here, so that a wrong argument is
reported as a `ValueError` that names it instead of surfacing later as a NaN
or as an exception from inside NumPy.
"""

from __future__ import annotations

import math
from numbers import Integral, Real

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def as_real_array(values: ArrayLike, name: str) -> FloatArray:
    """Return `values` as an array of floats; reject what is not a real number.

    Integers and floats are accepted, in any container NumPy understands (a
    pandas Series is read by position). Booleans, strings, complex numbers and
    arrays with masked entries are errors, not conversions.
    """
    if isinstance(values, np.ma.MaskedArray) and bool(np.any(values.mask)):
        raise ValueError(f"{name} has masked values; fill or drop them first")
    try:
        raw = np.asarray(values)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric: {error}") from error
    if raw.dtype.kind == "c":
        raise ValueError(f"{name} must be real, got complex values")
    if raw.dtype.kind not in "fiuO":
        raise ValueError(f"{name} must be numeric, got values of type {raw.dtype}")
    if raw.dtype.kind == "O":
        # Python objects, as in a pandas column of mixed content: each one has
        # to be a number, because float() would also convert "1.5" and True
        for item in raw.flat:
            if isinstance(item, (bool, np.bool_)) or not isinstance(item, Real):
                raise ValueError(
                    f"{name} must be numeric, got a value of type {type(item).__name__}"
                )
    try:
        return np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be numeric: {error}") from error


def as_series(values: ArrayLike, name: str, *, min_length: int = 1) -> FloatArray:
    """Return `values` as a one-dimensional array of finite floats."""
    array = as_real_array(values, name)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {array.shape}")
    if array.size < min_length:
        raise ValueError(f"{name} must have at least {min_length} value(s), got {array.size}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return array


def as_prices(values: ArrayLike, name: str = "prices", *, min_length: int = 1) -> FloatArray:
    """Return `values` as a one-dimensional array of finite, positive floats."""
    array = as_series(values, name, min_length=min_length)
    if np.any(array <= 0.0):
        raise ValueError(f"{name} must be strictly positive")
    return array


def check_real(
    value: object,
    name: str,
    *,
    minimum: float | None = None,
    strict: bool = False,
) -> float:
    """Return `value` as a finite float, optionally bounded below."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real number, got {value!r}")
    try:
        number = float(value)
    except OverflowError:  # an integer beyond the range of a float
        number = math.inf
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite, got {str(value)[:40]}")
    if minimum is not None:
        if strict and number <= minimum:
            raise ValueError(f"{name} must be greater than {minimum:g}, got {value!r}")
        if not strict and number < minimum:
            raise ValueError(f"{name} must be at least {minimum:g}, got {value!r}")
    return number


def check_positive(value: object, name: str) -> float:
    """Return `value` as a finite float greater than zero."""
    return check_real(value, name, minimum=0.0, strict=True)


def check_integer(value: object, name: str, *, minimum: int = 1) -> int:
    """Return `value` as an `int` not below `minimum` (booleans are rejected)."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    number = int(value)
    if number < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {number}")
    return number


def check_nu(value: object, name: str = "nu") -> float:
    """Return a smoothing parameter in `[0, 1)`."""
    nu = check_real(value, name)
    if not 0.0 <= nu < 1.0:
        raise ValueError(f"{name} must satisfy 0 <= {name} < 1, got {value!r}")
    return nu


def check_span(value: object, name: str = "span") -> float:
    """Return a filter span, a real number of at least one observation."""
    return check_real(value, name, minimum=1.0)


def nu_from_span(value: object, name: str = "span") -> float:
    """Smoothing parameter `1 - 2 / (span + 1)` of a span, in `[0, 1)`."""
    nu = 1.0 - 2.0 / (check_span(value, name) + 1.0)
    if not nu < 1.0:
        raise ValueError(
            f"{name} is too large: for {value!r} the smoothing parameter rounds to one"
        )
    return nu


def check_span_order(span_short: object, span_long: object) -> None:
    """Raise unless the fast span is smaller than the slow one, as floats see them."""
    if not nu_from_span(span_short, "span_short") < nu_from_span(span_long, "span_long"):
        raise ValueError(
            "span_short must be smaller than span_long, got "
            f"span_short={span_short!r}, span_long={span_long!r}"
        )


def check_optional_cap(value: object, name: str) -> float | None:
    """Return `None` or a positive float."""
    if value is None:
        return None
    return check_positive(value, name)


def checked_span(value: object, name: str) -> float:
    """Return a span as a float, after checking that its filter exists."""
    nu_from_span(value, name)
    return check_span(value, name)


def store(config: object, **values: object) -> None:
    """Set fields of a frozen configuration from inside its `__post_init__`.

    The checks return plain Python numbers. Storing those, and not what was
    passed in, keeps a NumPy scalar of another precision (a `float32` read
    from a file, say) from pulling the arithmetic of a system down to it.
    """
    for name, value in values.items():
        object.__setattr__(config, name, value)
