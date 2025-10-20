# SPDX-License-Identifier: GPL-2.0+
import ctypes
import os
import abc
import collections
from typing import List, Any, NamedTuple, Type
from functools import cache

import gfModParser as gf
from ..types import factory as type_factory

from .arguments import fArguments


class Result(NamedTuple):
    result: Any
    args: dict[str, Any]


class fProcedure(abc.ABC):

    def __init__(
        self, lib: ctypes.CDLL, definition: gf.Symbol, module: gf.Module, **kwargs
    ):
        self._module = module
        self.definition = definition
        self._lib = lib
        self._result = None

        self._proc = getattr(self._lib, self.definition.mangled_name)
        self._set_return()

    def __call__(self, *args, **kwargs) -> Result:

        self.args = fArguments(
            procedure=self.definition, module=self._module, values=[args, kwargs]
        )

        if len(self.args):
            self.result = self._proc(*self.args.arg_list())
        else:
            self.result = self._proc()

        return Result(self.result, self.args.convert_args_back())

    @property
    def ctype(self):
        return self._proc

    def __repr__(self):
        return self.__doc__

    @abc.abstractmethod
    def _set_return(self):
        raise NotImplementedError

    def _doc_args(self):
        args = []
        for key in self.definition.properties.formal_argument:
            arg = type_factory(self._module[key])()
            args.append(f"{str(arg)} :: {self._module[key].name}")

        args = ", ".join(args)
        return args

    @property
    @abc.abstractmethod
    def result(self):
        raise NotImplementedError

    @result.setter
    @abc.abstractmethod
    def result(self, value):
        raise NotImplementedError

    # def __call__(self, *args, **kwargs) -> Result:
    #     self._args = args
    #     self._kwargs = kwargs

    #     # Unpack args and kwargs
    #     self._convert_args_in()

    #     # Convert vlaues and typses into ctypes
    #     self._set_args()

    #     if len(self._proc_args):
    #         res = self._proc(*self._proc_args)
    #     else:
    #         res = self._proc()

    #     # Convert return type
    #     ret = self._convert_return(res)

    #     # Convert arguments
    #     args = self._convert_args_out()

    #     return Result(ret, args)

    # def _set_argument_types(self):
    #     """Sets the procedures argument types

    #     Note none of these should set pointer status yet, this is only setting up the f_type's
    #     """
    #     self._pre_args_types = self._pre_set_args_types()
    #     self._middle_args_types = self._set_args_types()
    #     self._post_args_types = self._post_set_args_types()

    # def _pre_set_args_types(self) -> List[type[f_type]]:
    #     """Arguments that get added to the start of the argument list"""

    #     res = []

    #     if self.definition.is_function:
    #         if self.return_type.type == "character":
    #             # Add a character and a strlen for functions returning characters
    #             res.append(self.return_var)
    #             res.append(ftype_strlen)
    #         elif self.return_type.is_array:
    #             # Need an empty array for explicit arrays
    #             res.append(self.return_var)

    #     return res

    # def _set_args_types(self) -> List[type[f_type]]:
    #     """Arguments that go in the middle"""

    #     res = []

    #     for key in self.definition.properties.formal_argument:
    #         res.append(factory(self._module[key]))

    #     return res

    # def _post_set_args_types(self) -> List[type[f_type]]:
    #     """Arguments that get added to the end of the argument list"""
    #     res = []

    #     for key in self.definition.properties.formal_argument:
    #         arg = self._module[key]
    #         if arg.type == "character":
    #             res.append(ftype_strlen)
    #         if arg.properties.attributes.optional and not arg.type == "character":
    #             res.append(ftype_optional)

    #     return res

    # def _convert_args_in(self):
    #     """
    #     Resolves input arguments into dict.

    #     If argument is unset then we store a None
    #     """

    #     arg_values = {}
    #     for key in self.definition.properties.formal_argument:
    #         arg_values[key] = None

    #     for value, key in zip(self._args, arg_values.keys()):
    #         arg_values[key] = value

    #     for k, v in self._kwargs.items():
    #         if k in arg_values:
    #             if arg_values[k] is not None:
    #                 raise ValueError(f"Argument {k} set twice")
    #             arg_values[k] = v
    #         else:
    #             raise ValueError(f"Argument {k} not found in call signature")

    #     self._arg_values = arg_values

    # def _set_args(self):
    #     pre = []
    #     middle = []
    #     post = []

    #     for arg in self._pre_args_types:
    #         pre.append(arg()._ctype)

    #     for index, (key, value) in enumerate(self._arg_values.items()):
    #         arg = self._module[key]
    #         if arg.properties.attributes.pointer:
    #             middle.append(self._middle_args_types[index](value).pointer2())
    #         else:
    #             middle.append(self._middle_args_types[index](value).pointer())

    #     index = 0
    #     for key in self.definition.properties.formal_argument:
    #         arg = self._module[key]
    #         if arg.type == "character":
    #             post.append(
    #                 self._post_args_types[index].from_param(
    #                     len(self._arg_values[arg.name])
    #                 )
    #             )
    #             index += 1
    #         if arg.properties.attributes.optional:
    #             if self._arg_values[arg.name] is None:
    #                 post.append(self._post_args_types[index].from_param(None))
    #             else:
    #                 post.append(self._post_args_types[index].from_param(1))
    #             index += 1

    #     self._proc_args = pre + middle + post

    # def _convert_return(self, res):
    #     # TODO: handle returning characters and arrays
    #     return res

    # def _convert_args_out(self):
    #     res = {}
    #     for key, ftype, value in zip(
    #         self._arg_values.keys(), self._middle_args_types, self._proc_args
    #     ):
    #         arg = self._module[key]
    #         if arg.properties.attributes.pointer:
    #             t = ftype.from_ctype(value.contents.contents)
    #         else:
    #             t = ftype.from_ctype(value.contents)
    #         res[self._module[key].name] = t.value

    #     return res
