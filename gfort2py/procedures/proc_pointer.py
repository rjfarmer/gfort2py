# SPDX-License-Identifier: GPL-2.0+

import ctypes
from typing import Any

import gfModParser as gf

from .arguments import factory_return, fArguments, fArgumentsExtra
from .functions import fFunc
from .procedures import Result


class fProcPointer(fFunc):
    """Procedure pointer wrapper backed by a module variable slot."""

    def __init__(self, lib: ctypes.CDLL, definition: gf.Symbol, module: gf.Module):
        self.pointer_definition = definition
        interface_ref = int(definition.properties.typespec.interface)

        if interface_ref <= 0:
            raise TypeError(
                f"Procedure pointer {definition.name} has no callable interface"
            )

        interface_definition = module[interface_ref]
        super().__init__(lib, interface_definition, module)

    def _set_return(self):
        if self.definition.is_subroutine:
            self._proc.argtypes = None
            return

        super()._set_return()

    def _pointer_slot(self) -> ctypes.c_void_p:
        return ctypes.c_void_p.in_dll(self._lib, self.pointer_definition.mangled_name)

    @property
    def address(self) -> int | None:
        return self._pointer_slot().value

    def bind(self, value: Any) -> None:
        """Bind this pointer to another callable procedure-like object."""
        if value is None:
            self._pointer_slot().value = None
            return

        source_addr = getattr(value, "address", None)
        if source_addr is not None:
            self._pointer_slot().value = source_addr
            return

        ctype = getattr(value, "ctype", None)
        if ctype is None:
            raise TypeError(
                f"Can not bind non-procedure value to {self.pointer_definition.name}"
            )

        addr = ctypes.cast(ctype, ctypes.c_void_p).value
        self._pointer_slot().value = addr

    def _make_proc(self, addr: int, all_args: list[Any]):
        # Build a concrete callback signature from prepared ctypes values.
        # This preserves pointer depths expected by Fortran ABI lowering.
        argtypes = [ctypes.c_void_p if arg is None else type(arg) for arg in all_args]

        restype: Any = None
        if not self.definition.is_subroutine:
            if (
                self.return_type.type.lower() == "character"
                or self.return_type.is_array
            ):
                restype = None
            else:
                if (
                    self.return_type.type.lower() == "real"
                    and self.return_type.kind == 16
                ):
                    raise TypeError(
                        "Can not return a quad from a procedure, it must be an argument or module variable"
                    )
                restype = self.return_var.ctype

        proc_type = ctypes.CFUNCTYPE(restype, *argtypes)
        return proc_type(addr)

    def __call__(self, *args, **kwargs) -> Result:
        addr = self.address
        if addr is None:
            raise AttributeError(
                f"Procedure pointer {self.pointer_definition.name} is not associated"
            )

        self._args_start = factory_return(
            procedure=self.definition,
            module=self._module,
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

            self._proc = self._make_proc(addr, all_args)

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
    def result(self):
        if self.definition.is_subroutine:
            return None

        return super().result

    @result.setter
    def result(self, value):
        if self.definition.is_subroutine:
            self._result = None
            return

        fFunc.result.fset(self, value)

    @property
    def __doc__(self):
        if self.definition.is_subroutine:
            ftype = f"subroutine {self.pointer_definition.name}"
        else:
            ftype = f"{str(self.return_var)} function {self.pointer_definition.name}"
        return f"{ftype} ({self._doc_args()})"
