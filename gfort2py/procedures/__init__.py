# SPDX-License-Identifier: GPL-2.0+
from typing import Any

import gfModParser as gf

from .functions import fFunc
from .procedures import fProcedure
from .subroutine import fSub

__all__ = ["factory", "fProcedure", "register_proc", "fFunc", "fSub"]

# Registry mapping frozenset of flag names -> fProcedure subclass.
# Flags come from attributes on gf.Symbol (e.g. is_subroutine, bind_c).
# The factory tries each registered key in insertion order;
# the first key whose flags are ALL True on the symbol wins.
_proc_registry: dict[frozenset[str], type[fProcedure]] = {
    frozenset({"is_subroutine"}): fSub,
    frozenset(): fFunc,  # default: any non-subroutine is a function
}


def register_proc(flags: frozenset[str], cls: type[fProcedure]) -> None:
    """Register a fProcedure subclass for a given set of Symbol flag names.

    The entry is inserted *before* the default catch-all so it takes priority.

    Args:
        flags: Frozenset of attribute names on ``gf.Symbol`` that must all be
               ``True`` for this class to be selected.
        cls: The :class:`fProcedure` subclass to use when the flags match.
    """
    # Insert before the final catch-all entry (empty frozenset)
    items = list(_proc_registry.items())
    items.insert(len(items) - 1, (flags, cls))
    _proc_registry.clear()
    _proc_registry.update(items)


def factory(procedure: gf.Symbol) -> type[fProcedure]:
    for flags, cls in _proc_registry.items():
        if all(getattr(procedure, flag, False) for flag in flags):
            return cls
    # Unreachable: empty frozenset always matches
    return fFunc
