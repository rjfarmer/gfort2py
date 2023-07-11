# SPDX-License-Identifier: GPL-2.0+

import ctypes
import itertools
from pprint import pprint

from .fUnary import run_unary


def copy_array(src, dst, length, size):
    ctypes.memmove(
        dst,
        src,
        length * size,
    )


def resolve_other_args(obj, other_args, module, lib, fProc):
    """
    We want to iterate over the components of obj
    and if they are symbol_refs look them up in other_args.

    This way we can resolve attributes that are runtime set,
    for instance dimension(n) arrays, where n is a dummy argument

    For now limit to array bounds

    fProc is needed in case we have to call a user function. We can't
    import it as utils.py is imported by fProc.py itself
    """

    if obj.is_array():
        for i in itertools.chain(obj.sym.array_spec.lower, obj.sym.array_spec.upper):
            i = _resolve_arg(i, other_args, module, lib, fProc)

    if obj.is_char():
        obj.sym.ts.charlen = _resolve_arg(
            obj.sym.ts.charlen, other_args, module, lib, fProc
        )

    return obj


def _resolve_arg(arg, other_args, module, lib, fProc):
    if not hasattr(arg, "exp_type"):
        return arg

    if arg.exp_type == "CONSTANT":
        return arg
    elif arg.exp_type == "VARIABLE":
        # Sometimes we try re-resolving already resolved arguments
        # so skip if value is not a symbol_ref
        if hasattr(arg.value, "ref"):
            ref = arg.value.ref
            for j in other_args:
                if ref == j.symbol_ref:
                    arg.value = j.value
    elif arg.exp_type == "OP":
        # Unary operator
        op = arg.unary_op
        arg1 = _resolve_arg(arg.unary_args[0], other_args, module, lib, fProc)
        arg2 = _resolve_arg(arg.unary_args[1], other_args, module, lib, fProc)
        arg.value = run_unary(op, arg1.value, arg2.value)
    elif arg.exp_type == "FUNCTION":
        # User supplied function
        func_ref = arg.value.ref
        # Lookup function
        func_sym = module[func_ref]
        func = fProc(lib, func_sym, module)
        # Lookup args
        # TODO: Only works for single argument functions for now
        arg = _resolve_arg(module[arg.args.value.ref], other_args, module, lib, fProc)
        for a in other_args:
            if a.symbol_ref == arg.head.id:
                func_arg = a.value
                break

        arg.value = func(func_arg).result

    return arg
