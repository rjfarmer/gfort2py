# SPDX-License-Identifier: GPL-2.0+

import ctypes
import itertools


def copy_array(src, dst, length, size):
    ctypes.memmove(
        dst,
        src,
        length * size,
    )


def resolve_other_args(obj, other_args):
    """
    We want to iterate over the components of obj
    and if they are symbol_refs look them up in other_args.

    This way we can resolve attributes that are runtime set,
    for instance dimension(n) arrays, where n is a dummy argument

    For now limit to array bounds
    """

    if not obj.is_array():
        return obj

    for i in itertools.chain(obj.sym.array_spec.lower, obj.sym.array_spec.upper):
        if not isinstance(i.value, int):
            ref = i.value.ref
            for j in other_args:
                if ref == j.symbol_ref:
                    i.value = j.value

    return obj
