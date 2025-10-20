# SPDX-License-Identifier: GPL-2.0+

import sys
from typing import Type, Any, Iterable
import abc
from dataclasses import dataclass

import gfModParser as gf

from ..types import factory

from .argument import fArg

# Handle converting procedure agurements into correct type to pass to ctypes


@dataclass
class Arg:
    definition: gf.Symbol
    argument: fArg
    value: Any = None
    set: bool = False
    actual: bool = True


class ArgumentError(Exception):
    pass


class fArguments:
    def __init__(
        self,
        procedure: gf.Symbol,
        module: Type[gf.Module],
        values: list[tuple, dict[str, Any]],
    ):
        self.procedure: gf.Symbol = procedure
        self.module: Type[gf.Module] = module
        self._values = values

        self.args: dict[str, Arg] = {}

        self.expand_values()

        self.set_vars()

    def expand_values(self):
        """
        Takes the args and kwargs in self._values and match with the
        procedure names
        """
        for i in self.procedure.properties.formal_argument:
            name = self.module[i].name
            self.args[name] = Arg(
                definition=self.module[i],
                argument=fArg(self.module[i], self.procedure, self.module),
            )

        # First compoent of self._values is the args so expand in order
        for key, value in zip(self.args.keys(), self._values[0]):
            self.args[key].value = value
            self.args[key].set = True

        # Next loop over the kwargs and set, if not already set
        for key, value in self._values[1].items():
            if key not in self.args:
                raise ArgumentError(f"{key} not an argument to the procedure")

            if self.args[key].set:
                raise ArgumentError(
                    f"Keyword arg {key} already set via non-keyword arguments"
                )

            self.args[key].value = value
            self.args[key].set = True

        for key, arg in self.args.items():
            if arg.set == False and not arg.argument.can_be_unset():
                raise ArgumentError(f"Argument {key} must have a value set")

    def set_vars(self):
        for key in self.args.keys():
            self.args[key].argument.ctype = self.args[key].value

    def arg_list(self):
        return [c.argument.ctype for c in self.args.values()]

    def __len__(self):
        return len(self.args)

    def convert_args_back(self):
        res = {}
        for key, arg in self.args.items():
            if arg.actual:
                res[key] = arg.argument.value()
        return res
