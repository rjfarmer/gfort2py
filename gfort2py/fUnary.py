# SPDX-License-Identifier: GPL-2.0+

import operator

default_ops = {
    "UPLUS": operator.__add__,
    "UMINUS": operator.__sub__,
    "PLUS": operator.__add__,
    "MINUS": operator.__sub__,
    "TIMES": operator.__mul__,
    "DIVIDE": operator.__truediv__,
    "POWER": operator.__pow__,
    "CONCAT": operator.__add__,  # Only for strings
    "AND": operator.__and__,
    "OR": operator.__or__,
    "EQV": operator.__eq__,
    "NEQV": operator.__ne__,
    "EQ_SIGN": operator.__eq__,
    "EQ": operator.__eq__,
    "NE_SIGN": operator.__ne__,
    "NE": operator.__ne__,
    "GT_SIGN": operator.__gt__,
    "GT": operator.__gt__,
    "GE_SIGN": operator.__ge__,
    "GE": operator.__ge__,
    "LT_SIGN": operator.__le__,
    "LT": operator.__le__,
    "LE_SIGN": operator.__le__,
    "LE": operator.__le__,
    "NOT": operator.__not__,
    "PARENTHESES": None,
    "USER": None,
    "NULL": None,
}


def run_unary(unary, x, y, *, ops=default_ops):
    if unary == "PARENTHESES":
        return x

    op = ops[unary]

    if op is None:
        raise NotImplementedError(f"Unary op {unary} not implemented")

    return op(x, y)
