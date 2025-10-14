# SPDX-License-Identifier: GPL-2.0+

from typing import Type
import functools

import gfModParser as gf

__all__ = ["get_module"]


@functools.lru_cache(maxsize=None)
def get_module(name: str) -> Type[gf.Module]:
    return gf.Module(name)
