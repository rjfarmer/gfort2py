# SPDX-License-Identifier: GPL-2.0+

import ctypes
import itertools

from .fUnary import run_unary


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
        i = _resolve_arg(i, other_args)

    return obj


def _resolve_arg(arg, other_args):
    if arg.exp_type == "CONSTANT":
        return arg
    elif arg.exp_type == "VARIABLE":
        ref = arg.value.ref
        for j in other_args:
            if ref == j.symbol_ref:
                arg.value = j.value
    elif arg.exp_type == "OP":
        # Unary operator
        op = arg.unary_op
        arg1 = _resolve_arg(arg.unary_args[0], other_args)
        arg2 = _resolve_arg(arg.unary_args[1], other_args)

        # print(op,arg1.value,arg2.value)
        # print(run_unary(op,arg1.value,arg2.value))
        arg.value = run_unary(op, arg1.value, arg2.value)

    return arg
