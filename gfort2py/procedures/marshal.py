# SPDX-License-Identifier: GPL-2.0

import ctypes
from typing import Any


def resolve_procedure_address(value: Any, *, name: str) -> int:
    """Resolve a callable wrapper to a concrete C function address."""
    source_addr = getattr(value, "address", None)
    if source_addr is not None:
        return int(source_addr)

    ctype = getattr(value, "ctype", None)
    if ctype is None:
        raise TypeError(f"Expected a procedure-like value for {name}")

    addr = ctypes.cast(ctype, ctypes.c_void_p).value
    if addr is None:
        raise TypeError(f"Expected a valid procedure address for {name}")

    return int(addr)


def marshal_dummy_procedure_argument(
    value: Any,
    *,
    name: str,
    is_proc_pointer: bool,
    is_optional: bool,
) -> tuple[Any, Any, ctypes.c_void_p | None]:
    """Prepare ctypes value for a dummy procedure/procedure-pointer argument."""
    if value is None and is_optional:
        return None, None, None

    cproc = getattr(value, "ctype", None)
    if cproc is None:
        raise TypeError(f"Expected a procedure-like value for {name}")

    if not is_proc_pointer:
        return cproc, value, None

    # gfortran lowers dummy procedure pointers as a pointer to the
    # procedure-pointer slot, not as a bare function address.
    pointer_definition = getattr(value, "pointer_definition", None)
    proc_lib = getattr(value, "_lib", None)

    if pointer_definition is not None and proc_lib is not None:
        slot = ctypes.c_void_p.in_dll(proc_lib, pointer_definition.mangled_name)
    else:
        slot = ctypes.c_void_p(resolve_procedure_address(value, name=name))

    return ctypes.pointer(slot), value, slot
