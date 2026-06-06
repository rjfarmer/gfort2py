# SPDX-License-Identifier: GPL-2.0+

import ctypes
from abc import ABCMeta
from typing import Any, cast

import gfModParser as gf

from ...types import factory
from ...utils import get_c_runtime, strlen_ctype


class fArg(metaclass=ABCMeta):
    def __init__(self, definition: gf.Symbol, procedure: gf.Symbol, module: gf.Module):
        self.definition: gf.Symbol = definition
        self.module: gf.Module = module
        self.procedure: gf.Symbol = procedure

        cls = factory(self.definition)
        c = cls.__new__(cls)
        c._symbol = self.definition
        c._module_obj = self.module
        type(c).__init__(c)  # type: ignore[misc]
        self.base = c
        self._ctype = None
        self._procedure_value = None
        self._procedure_pointer_slot = None
        self._alloc_char_input_buf = None
        self._alloc_char_data = None
        self._alloc_char_data_ptr = None
        self._alloc_char_len = None
        self._alloc_char_len_ptr = None

    def _libc(self):
        libc = get_c_runtime()
        libc.malloc.argtypes = [ctypes.c_size_t]
        libc.malloc.restype = ctypes.c_void_p
        return libc

    def _setup_allocatable_character(self):
        if self._alloc_char_data is None:
            self._alloc_char_data = ctypes.c_void_p(0)
            self._alloc_char_data_ptr = ctypes.pointer(self._alloc_char_data)
            self._alloc_char_len = strlen_ctype()(0)
            self._alloc_char_len_ptr = ctypes.pointer(self._alloc_char_len)

    def _uses_scalar_allocatable_character_abi(self) -> bool:
        if not (self.is_allocatable and self.is_character):
            return False
        # Character arrays are descriptor-backed; only scalar deferred-length
        # characters use the lowered char** + strlen* ABI.
        if self.definition.is_array:
            return False
        return self.definition.properties.typespec.charlen.value <= 0

    @property
    def is_optional(self) -> bool:
        return self.definition.properties.attributes.optional

    @property
    def is_intent_out(self) -> bool:
        return self.definition.properties.attributes.intent == "OUT"

    @property
    def is_pointer(self) -> bool:
        return self.definition.properties.attributes.pointer

    @property
    def is_proc_pointer(self) -> bool:
        return self.definition.properties.attributes.proc_pointer

    @property
    def is_allocatable(self) -> bool:
        return self.definition.properties.attributes.allocatable

    def can_be_unset(self) -> bool:
        return self.is_intent_out or self.is_allocatable or self.is_optional

    @property
    def is_value(self) -> bool:
        return self.definition.properties.attributes.value

    @property
    def is_return(self) -> bool:
        return False

    @property
    def is_character(self) -> bool:
        return self.definition.type.lower() == "character"

    @property
    def is_procedure(self) -> bool:
        return self.definition.is_procedure

    @property
    def needs_resolving_late(self) -> bool:
        return False

    @property
    def ctype(self):
        return self._ctype

    @ctype.setter
    def ctype(self, value):
        if self.is_procedure:
            self._set_procedure(value)
            return

        if self._uses_scalar_allocatable_character_abi():
            self._set_allocatable_character(value)
            return

        if value is None and self.is_optional:
            self._set_optional_value()
            return

        if self.is_wrapper(value):
            if self.is_dt_like:
                self.base = value
            # Class-like dont need special handling

        self.base.value = value
        self._set_value()

    def value(self):
        if self._ctype is None:
            return None

        if self.is_procedure:
            return self._procedure_value

        if self._uses_scalar_allocatable_character_abi():
            return self._get_allocatable_character_value()

        if (
            self.is_optional
            and self.is_character
            and isinstance(self._ctype, ctypes.c_void_p)
            and self._ctype.value is None
        ):
            return None

        return self._get_value()

    def cleanup(self) -> None:
        release = getattr(self.base, "release", None)
        if callable(release):
            try:
                release()
            except (AttributeError, TypeError, ValueError, ctypes.ArgumentError):
                # Cleanup is best-effort and should not mask call results.
                pass

    def _set_procedure(self, value):
        if value is None and self.is_optional:
            self._procedure_value = None
            self._procedure_pointer_slot = None
            self._ctype = None
            return

        cproc = getattr(value, "ctype", None)
        if cproc is None:
            raise TypeError(
                f"Expected a procedure-like value for {self.definition.name}"
            )

        self._procedure_value = value

        if self.is_proc_pointer:
            # gfortran lowers dummy procedure pointers as a pointer to the
            # procedure-pointer slot, not as a bare function address.
            pointer_definition = getattr(value, "pointer_definition", None)
            proc_lib = getattr(value, "_lib", None)

            if pointer_definition is not None and proc_lib is not None:
                slot = ctypes.c_void_p.in_dll(proc_lib, pointer_definition.mangled_name)
            else:
                addr = ctypes.cast(cproc, ctypes.c_void_p).value
                slot = ctypes.c_void_p(addr)

            self._procedure_pointer_slot = slot
            self._ctype = ctypes.pointer(slot)
        else:
            self._procedure_pointer_slot = None
            self._ctype = cproc
        return

    def _set_allocatable_character(self, value):
        self._setup_allocatable_character()

        if value is None:
            self._alloc_char_input_buf = None
            self._alloc_char_data.value = None
            self._alloc_char_len.value = 0
        else:
            if hasattr(value, "encode"):
                value = value.encode(self.base._char.encoding)

            length = len(value)
            self._alloc_char_len.value = length

            if length > 0:
                self._alloc_char_input_buf = self._libc().malloc(length)
                ctypes.memmove(self._alloc_char_input_buf, value, length)
                self._alloc_char_data.value = self._alloc_char_input_buf
            else:
                self._alloc_char_input_buf = None
                self._alloc_char_data.value = None

        self._ctype = (self._alloc_char_data_ptr, self._alloc_char_len_ptr)
        return

    def _set_optional_value(self):
        if self.is_character:
            self._ctype = ctypes.c_void_p(0)
        elif self.is_value:
            self.base.value = 0
            self._ctype = self.base._ctype
        else:
            self._ctype = None
        return

    @property
    def is_dt_like(self) -> bool:
        return self.definition.is_dt

    @property
    def is_class_like(self) -> bool:
        return self.definition.type.lower() == "class"

    def is_wrapper(self, value) -> bool:
        return (
            hasattr(value, "_ctype")
            and callable(getattr(value, "pointer", None))
            and callable(getattr(value, "pointer2", None))
        )

    def _set_value(self):
        if self.is_value:
            self._ctype = self.base._ctype
        elif self.is_pointer:
            if self.definition.is_array:
                self._ctype = self.base.pointer()
            else:
                self._ctype = self.base.pointer2()
        else:
            self._ctype = self.base.pointer()

    def _get_allocatable_character_value(self):
        self._setup_allocatable_character()

        if self._alloc_char_data.value is None:
            return None

        length = int(self._alloc_char_len.value)
        if length <= 0:
            return ""

        data = ctypes.string_at(self._alloc_char_data.value, length)
        return data.decode(self.base._char.encoding)

    def _get_value(self):
        if self.is_value:
            c = self._ctype
        elif self.is_pointer:
            p = cast(Any, self._ctype)
            if self.definition.is_array:
                c = p.contents
            else:
                c = p.contents.contents
        else:
            p = cast(Any, self._ctype)
            c = p.contents

        if self.definition.is_array or self.definition.is_dt:
            self.base._ctype = c
            return self.base.value

        return self.base.from_ctype(c, symbol=self.definition).value
