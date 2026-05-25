# SPDX-License-Identifier: GPL-2.0+

import functools

import gfModParser as gf

__all__ = ["get_module", "register_module_alias"]


_module_aliases: dict[str, gf.Module] = {}


@functools.lru_cache(maxsize=None)
def get_module(name: str) -> gf.Module:
    key = name.lower().strip()

    # Dummy/procedure-local symbols in .mod files can have an empty module field.
    # If we only have one registered module alias, use it as the resolution target.
    if key in {"", "."} and len(_module_aliases) == 1:
        return next(iter(_module_aliases.values()))

    if key in _module_aliases:
        return _module_aliases[key]
    return gf.Module(name)


def register_module_alias(name: str, module: gf.Module) -> None:
    _module_aliases[name.lower()] = module
