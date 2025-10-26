# SPDX-License-Identifier: GPL-2.0+

import sys
from typing import Type, Any, Iterable
import abc
from dataclasses import dataclass

import gfModParser as gf

from ...types import factory

from .argument import fArg

# Handle converting procedure agurements into correct type to pass to ctypes


class ArgumentError(Exception):
    pass


@dataclass
class Arg:
    definition: gf.Symbol
    argument: fArg
    value: Any = None
    set: bool = False
    actual: bool = True


class fArgumentsAbstract(abc.ABC):
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

    def get_ctypes(self) -> list[Any]:
        return [c.argument.ctype for c in self.args.values()]

    def __len__(self):
        return len(self.args)

    @abc.abstractmethod
    def set_values(self):
        """
        Sets all argument ctypes with thier value
        """
        raise NotImplemented

    @abc.abstractmethod
    def get_values(self) -> dict[str, Any]:
        """
        Converts ctypes back into thier base type and return the dict with thier values
        """
        raise NotImplemented


class fArguments(fArgumentsAbstract):
    def set_values(self):
        """
        Sets all argument ctypes with thier value
        """
        self.expand_values()

        for key in self.args.keys():
            self.args[key].argument.ctype = self.args[key].value

    def get_values(self) -> dict[str, Any]:
        """
        Converts ctypes back into thier base type and return the dict with thier values
        """

        res = {}
        for key, arg in self.args.items():
            if arg.actual:
                res[key] = arg.argument.value()
        return res
