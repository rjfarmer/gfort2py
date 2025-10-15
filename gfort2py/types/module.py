# SPDX-License-Identifier: GPL-2.0+

import functools

import gfModParser as gf

__all__ = ["get_module"]


@functools.lru_cache(maxsize=None)
def get_module(name: str) -> gf.Module:
    return gf.Module(name)
