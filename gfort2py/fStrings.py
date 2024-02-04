# SPDX-License-Identifier: GPL-2.0+

import ctypes
import numpy as np

from .fVar_t import fVar_t
from .fArrays import fAssumedShape
from .utils import copy_array, is_64bit


class fStr(fVar_t):
    """ """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._len = None

    def ctype(self):
        return self._ctype_base * self.len()

    def from_param(self, value):
        if self.obj.is_deferred_len():
            self._len = len(value)

        if self.cvalue is None:
            self.cvalue = self.ctype()()

        self._value = value

        if self._value is None:
            return None

        if hasattr(self._value, "encode"):
            self._value = self._value.encode()

        if len(self._value) > self.len():
            self._value = self._value[: self.len()]
        else:
            self._value = self._value + b" " * (self.len() - len(self._value))

        # self._buf = bytearray(self._value)  # Need to keep hold of the reference
        self.cvalue.value = self._value

        return self.cvalue

    @property
    def value(self):
        try:
            return self.cvalue.value.decode()
        except AttributeError:
            return str(self.cvalue)  # Functions returning str's give us str not bytes

    @value.setter
    def value(self, value):
        self.from_param(value)

    def len(self):
        if self._len is None:
            if self.obj.is_deferred_len():
                self._len = len(self.cvalue)
            else:
                self._len = self.obj.strlen.value
        return self._len

    def ctype_len(self, *args):
        if is_64bit():
            return ctypes.c_int64(self.len())
        else:
            return ctypes.c_int32(self.len())

    @property
    def __doc__(self):
        try:
            return f"{self.type}(LEN={self.obj.len}) :: {self.name}"
        except AttributeError:
            return f"{self.type}(LEN=:) :: {self.name}"

    def sizeof(self):
        return ctypes.sizeof(self.ctype)


class fAllocStr(fStr):
    def __init__(self, *args, **kwargs):
        self._len = None
        self._len_ctype = None
        super().__init__(*args, **kwargs)

    def ctype(self):
        return self._ctype_base

    @property
    def _ctype_base(self):
        len = self.len()
        if len is None:
            return ctypes.c_char_p
        else:
            return ctypes.c_char_p * len

    @_ctype_base.setter
    def _ctype_base(self, value):
        return self._ctype_base

    def from_param(self, value):
        if value is None:
            self.cvalue = ctypes.c_char_p(None)
            return self.cvalue

        self._len = len(value)

        self._value = value

        if hasattr(self._value, "encode"):
            self._value = self._value.encode()

        if self.cvalue is None:
            self.cvalue = self.ctype()()

        self.cvalue = self.ctype().from_address(ctypes.addressof(self.cvalue))

        if len(self._value) > self.len():
            self._value = self._value[: self.len()]
        else:
            self._value = self._value + b" " * (self.len() - len(self._value))

        self.cvalue[0] = self._value

        return self.cvalue

    @property
    def value(self):
        try:
            x = self.cvalue[0]
        except Exception:
            return None

        if x is None:
            return None
        else:
            if self.len() is None:
                return b""
            else:
                # We can get warnings from vlgrind around here (Invalid read of size 1) and (Address 0x.... is 0 bytes after a block of size 6 alloc'd).
                # The problem is various string handling routines
                # assume a c-ctyle string with a null pointer
                # at the end. But Fortran strings dont have that,
                # so Python tries to read beyond the end of the string
                # as long as we use the self.len() for lengths we should be
                # okay as thats set by Fortran and is the correct
                # length.
                return x[: self.len()].decode()

    @value.setter
    def value(self, value):
        self.from_param(value)

    def len(self):
        if self._len_ctype is not None:
            self._len = self._len_ctype.contents.value

        return self._len

    def in_dll(self, lib):
        self.cvalue = self.ctype().in_dll(lib, self.mangled_name)
        return self.cvalue

    @property
    def __doc__(self):
        return f"character(LEN=(:)), allocatable :: {self.name}"

    def sizeof(self):
        return ctypes.sizeof(self.ctype)

    def to_proc(self, value, other_args):
        if is_64bit():
            lsize = ctypes.c_int64
        else:
            lsize = ctypes.c_int32

        if value is None:
            self._len = None
            self.cvalue = ctypes.pointer(ctypes.c_char_p(None))
            self._len_ctype = ctypes.pointer(lsize(0))

        else:
            self._len = len(value)
            self.cvalue = self.from_param(value)
            self._len_ctype = ctypes.pointer(lsize(self._len))

        return self.Args(None, self.cvalue, self._len_ctype)


class fStrExplicit(fStr):
    def __init__(self, *args, **kwargs):
        self._len_ctype = None
        self._len = None
        super().__init__(*args, **kwargs)
        self.unpack = False

    def ctype(self):
        return self._ctype_base

    @property
    def _ctype_base(self):
        return ctypes.c_char * self.len() * self.obj.size

    @_ctype_base.setter
    def _ctype_base(self, value):
        try:
            return ctypes.c_char * self.len() * self.obj.size
        except TypeError:
            pass  # Dont allways now size when loading class

    def _array_check(self, value, know_shape=True):
        shape = self.obj.shape()
        ndim = self.obj.ndim

        if not np.issubdtype(value.dtype, np.bytes_):
            raise TypeError("Character strings must be bytes (S dtype)")

        if not value.flags["F_CONTIGUOUS"]:
            value = np.asfortranarray(value)

        if value.ndim != ndim:
            raise ValueError(
                f"Wrong number of dimensions, got {value.ndim} expected {ndim}"
            )

        if know_shape:
            if not self.obj.is_allocatable and list(value.shape) != shape:
                raise ValueError(f"Wrong shape, got {value.shape} expected {shape}")

        value = value.ravel(order="F")
        return value

    def from_param(self, value):
        if value is None:
            raise ValueError("Character array must not be None")

        self._value = self._array_check(value)

        self._len = self._value.dtype.itemsize

        if self.cvalue is None:
            self.cvalue = self.ctype()()

        copy_array(
            self._value.ctypes.data,
            ctypes.addressof(self.cvalue),
            ctypes.sizeof(self._ctype_base),
            1,
        )

        return self.cvalue

    def from_ctype(self, ct):
        self.cvalue = ct

        return self.value

    @property
    def value(self):
        z = np.zeros(self.obj.shape(), dtype=f"S{self.len()}")

        copy_array(
            ctypes.addressof(self.cvalue),
            z.ctypes.data,
            ctypes.sizeof(self._ctype_base),
            1,
        )

        return z

    @value.setter
    def value(self, value):
        self.from_param(value)

    def len(self):
        if self._len_ctype is not None:
            self._len = self._len_ctype.value

        if self._len is None:
            if self.obj.is_deferred_len():
                self._len = len(self.cvalue)
            else:
                self._len = self.obj.strlen.value

        return self._len

    def in_dll(self, lib):
        self.cvalue = self.ctype().in_dll(lib, self.mangled_name)
        return self.cvalue

    @property
    def __doc__(self):
        return (
            f"character(LEN=({self.len()})),dimension({self.obj.shape}) :: {self.name}"
        )

    def sizeof(self):
        return ctypes.sizeof(self.ctype)

    def to_proc(self, value, other_args):
        if value is None:
            l = 0
        else:
            l = value.dtype.itemsize

        if is_64bit():
            self._len_ctype = ctypes.c_int64(l)
        else:
            self._len_ctype = ctypes.c_int32(l)

        self.cvalue = self.from_param(value)

        return self.Args(None, self.cvalue, self._len_ctype)

    def __del__(self):
        self.cvalue = None
        self._len_ctype = None


class fStrAssumedShape(fAssumedShape):
    def __init__(self, *args, **kwargs):
        self._len_ctype = None
        self._len = None
        super().__init__(*args, **kwargs)
        self.unpack = False
        self.is_array = True

    @property
    def _ctype_base(self):
        # print(self.len(),self.obj.size)
        return ctypes.c_char * int(self.len() * self.obj.size)

    @_ctype_base.setter
    def _ctype_base(self, value):
        try:
            return ctypes.c_char * int(self.len() * self.obj.size)
        except TypeError:
            pass  # Dont always now the size when loading the class

    def _array_check(self, value, know_shape=True):
        if not np.issubdtype(value.dtype, np.bytes_):
            raise TypeError("Character strings must be bytes (S dtype)")

        return super()._array_check(value, know_shape)

    @property
    def value(self):
        if hasattr(self.cvalue, "contents"):
            cv = self.cvalue.contents
        else:
            cv = self.cvalue

        if cv.base_addr is None:
            return None

        shape = []
        for i in range(self.obj.ndim):
            shape.append(cv.dims[i].ubound - cv.dims[i].lbound + 1)

        shape = tuple(shape)
        size = (np.prod(shape),)

        z = np.zeros(shape, dtype=f"S{self.len()}")

        copy_array(
            cv.base_addr,
            z.ctypes.data,
            ctypes.sizeof(ctypes.c_char * int(self.len())),
            np.prod(shape),
        )

        return z

    @value.setter
    def value(self, value):
        self.from_param(value)

    def len(self):
        if self._len_ctype is not None:
            self._len = self._len_ctype.value

        if self._len is None:
            if self.obj.is_deferred_len():
                self._len = len(self.cvalue)
            else:
                self._len = self.obj.strlen.value

        return self._len

    def sizeof(self):
        return ctypes.sizeof(self.ctype)

    def to_proc(self, value, other_args):
        if value is None:
            l = self.obj.strlen.value
        else:
            l = value.dtype.itemsize

        if is_64bit():
            self._len_ctype = ctypes.c_int64(l)
        else:
            self._len_ctype = ctypes.c_int32(l)

        self.cvalue = ctypes.pointer(self.from_param(value))

        return self.Args(None, self.cvalue, self._len_ctype)
