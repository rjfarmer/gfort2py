# SPDX-License-Identifier: GPL-2.0+

import abc
import sys
from typing import Any, cast

import gfModParser as gf

from .arguments import fArguments
from .extra_arguments import fArgumentsExtra
from .return_arguments import (
    fReturnArrayArguments,
    fReturnCharArguments,
    fReturnDTArguments,
)


def factory_return(
    procedure: gf.Symbol,
    module: gf.Module,
    values: tuple[tuple[Any, ...], dict[str, Any]],
) -> fArguments:

    rt = cast(Any, procedure).return_type

    if rt.ftype == "character":
        return fReturnCharArguments(procedure, module, values)
    elif rt.is_array:
        return fReturnArrayArguments(procedure, module, values)
    elif rt.is_dt:
        return fReturnDTArguments(procedure, module, values)
    else:
        raise ValueError(f"Unknown return argument type {rt}")
