# SPDX-License-Identifier: GPL-2.0+

import sys
from typing import Type, Any, Iterable
import abc
from dataclasses import dataclass

import gfModParser as gf

from ...types import factory

from .argument import fArg
from .arguments import fArguments, Arg


class fArgumentsExtra(fArguments):
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
