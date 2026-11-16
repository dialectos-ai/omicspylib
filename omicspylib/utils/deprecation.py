"""
Helpers for renaming public arguments without breaking existing code.
"""
from __future__ import annotations

import warnings
from typing import Any

_NOT_SET: Any = object()


def resolve_renamed_arg(
    new_value: Any,
    old_value: Any,
    new_name: str,
    old_name: str,
    default: Any = None,
    stacklevel: int = 3,
) -> Any:
    """
    Return the value of a renamed argument, accepting the old name too.

    If the old argument is used, a ``DeprecationWarning`` is raised pointing
    to the new name. If both are used, a ``TypeError`` is raised.

    Parameters
    ----------
    new_value : Any
        Value passed with the new argument name.
    old_value : Any
        Value passed with the deprecated argument name. ``None`` means it
        was not used, so deprecated arguments should default to ``None``.
    new_name : str
        The new argument name, used in messages.
    old_name : str
        The deprecated argument name, used in messages.
    default : Any, optional
        Default value of the new argument. A new value equal to it is treated
        as not passed. Default is ``None``.
    stacklevel : int, optional
        Stack level of the warning. The default points to the caller of the
        public function that calls this helper.

    Returns
    -------
    Any
        The value to use.
    """
    old_used = old_value is not _NOT_SET and old_value is not None
    if not old_used:
        return new_value

    new_used = new_value is not _NOT_SET and not (
        new_value is default or new_value == default
    )
    if new_used:
        raise TypeError(
            f"Got values for both '{new_name}' and its deprecated alias "
            f"'{old_name}'. Use only '{new_name}'."
        )

    warnings.warn(
        f"'{old_name}' is deprecated and will be removed in a future release. "
        f"Use '{new_name}' instead.",
        category=DeprecationWarning,
        stacklevel=stacklevel,
    )
    return old_value
