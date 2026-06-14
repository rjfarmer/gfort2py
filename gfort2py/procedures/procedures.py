# SPDX-License-Identifier: GPL-2.0+
import abc
import collections
import ctypes
import os
from functools import cache
from typing import Any, List, NamedTuple, Type

import gfModParser as gf

from ..types import factory as type_factory
from .arguments import factory_return, fArguments, fArgumentsExtra


class Result(NamedTuple):
    result: Any
    args: dict[str, Any]


class fProcedure(metaclass=abc.ABCMeta):

    def _cleanup_argument_container(self, container: Any) -> None:
        if container is None:
            return

        args = getattr(container, "args", None)
        if not args:
            return

        for entry in args.values():
            cleanup = getattr(entry.argument, "cleanup", None)
            if callable(cleanup):
                cleanup()

    def _cleanup_return_arguments(self) -> None:
        release_args_start = getattr(self._args_start, "release", None)
        if callable(release_args_start):
            try:
                release_args_start()
            except (
                AttributeError,
                TypeError,
                ValueError,
                OSError,
                ctypes.ArgumentError,
            ):
                pass

        if self._args_start is None:
            return

        result_type = getattr(self._args_start, "_result_type", None)
        release = getattr(result_type, "release", None)
        if callable(release):
            try:
                release()
            except (AttributeError, TypeError, ValueError, ctypes.ArgumentError):
                pass

    def __init__(
        self, lib: ctypes.CDLL, definition: gf.Symbol, module: gf.Module, **kwargs
    ):
        self._module = module
        self.definition = definition
        self._lib = lib
        self._result = None
        self._args_start: Any = None

        self._proc = getattr(self._lib, self.definition.mangled_name)
        self._set_return()

    def __call__(self, *args, **kwargs) -> Result:
        self._args_start = factory_return(
            procedure=self.definition,
            module=self._module,
            lib=self._lib,
            values=(args, kwargs),
        )

        self.args = fArguments(
            procedure=self.definition, module=self._module, values=(args, kwargs)
        )

        self.args_end = fArgumentsExtra(
            procedure=self.definition,
            module=self._module,
            values=(args, kwargs),
            arguments=self.args,
        )

        try:
            if self._args_start is not None:
                self._args_start.set_values()
            self.args.set_values()
            self.args_end.set_values()

            args_start: list[Any] = []
            if self._args_start is not None:
                args_start = self._args_start.get_ctypes()

            all_args = [
                *args_start,
                *self.args.get_ctypes(),
                *self.args_end.get_ctypes(),
            ]

            if len(all_args):
                self.result = self._proc(*all_args)
            else:
                self.result = self._proc()

            resolved_return = self.resolve_return()
            resolved_args = self.resolve_args()
            return Result(resolved_return, resolved_args)
        finally:
            self._cleanup_argument_container(self.args_end)
            self._cleanup_argument_container(self.args)
            self._cleanup_return_arguments()

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

    def resolve_return(self):
        res = self.result

        if (
            res is None
            and not self.definition.is_subroutine
            and self._args_start is not None
        ):
            values = self._args_start.get_values()
            if len(values):
                res = values[0]

        return res

    def resolve_args(self):
        return self.args.get_values()
