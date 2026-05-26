# SPDX-License-Identifier: GPL-2.0+

from typing import Any

import gfModParser as gf

from .arguments import fArguments
from .extra_arguments import fArgumentsExtra
from .return_arguments import (
    fReturnArguments,
    fReturnArrayArguments,
    fReturnCharArguments,
    fReturnDTArguments,
)


def factory_return(
    procedure: gf.Symbol,
    module: gf.Module,
    values: tuple[tuple[Any, ...], dict[str, Any]],
) -> fReturnArguments | None:

    if procedure.is_subroutine:
        return None

    key = procedure.properties.symbol_reference
    rt = module[key]

    if rt.type.lower() == "character":
        return fReturnCharArguments(procedure, module, values, rt)
    elif rt.is_array:
        return fReturnArrayArguments(procedure, module, values, rt)
    elif rt.is_dt and rt.is_array:
        return fReturnDTArguments(procedure, module, values, rt)
    else:
        return None
