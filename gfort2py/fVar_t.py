# SPDX-License-Identifier: GPL-2.0+
import ctypes
import numpy as np
import weakref

from .fUnary import run_unary
from .allocate import alloc, dealloc


_index_t = ctypes.c_int64
_size_t = ctypes.c_int64


class _bounds14(ctypes.Structure):
    _fields_ = [("stride", _index_t), ("lbound", _index_t), ("ubound", _index_t)]


class _dtype_type(ctypes.Structure):
    _fields_ = [
        ("elem_len", _size_t),
        ("version", ctypes.c_int32),
        ("rank", ctypes.c_byte),
        ("type", ctypes.c_byte),
        ("attribute", ctypes.c_ushort),
    ]


def _make_fAlloc15(ndims):
    class _fAllocArray(ctypes.Structure):
        _fields_ = [
            ("base_addr", ctypes.c_void_p),
            ("offset", _size_t),
            ("dtype", _dtype_type),
            ("span", _index_t),
            ("dims", _bounds14 * ndims),
        ]

    return _fAllocArray


def deallocate(var):
    pass


class fVar_t:
    def __init__(self, obj):
        self.obj = obj

        self.type, self.kind = self.obj.type_kind()

    def name(self):
        return self.obj.name

    def mangled_name(self):
        return self.obj.mangled_name

    def module(self):
        return self.obj.module

    def _array_check(self, value, know_shape=True):
        value = value.astype(self.obj.dtype())
        shape = self.obj.shape()
        ndim = self.obj.ndim

        if not value.flags["F_CONTIGUOUS"]:
            value = np.asfortranarray(value)

        if value.ndim != ndim:
            raise ValueError(
                f"Wrong number of dimensions, got {value.ndim} expected {ndim}"
            )

        if know_shape:
            if list(value.shape) != shape:
                raise ValueError(f"Wrong shape, got {value.shape} expected {shape}")

        value = value.ravel(order="F")
        return value

    def from_param(self, value, ctype=None):

        if self.obj.is_optional() and value is None:
            return None

        if self.obj.is_array():
            if self.obj.is_explicit():
                value = self._array_check(value)
                if ctype is None:
                    ctype = self.ctype(value)()
                self.copy_array(
                    value.ctypes.data,
                    ctypes.addressof(ctype),
                    self.sizeof,
                    self.obj.size,
                )
                return ctype
            elif self.obj.is_assumed_size():
                value = self._array_check(value, know_shape=False)
                if ctype is None:
                    ctype = self.ctype(value)()

                self.copy_array(
                    value.ctypes.data,
                    ctypes.addressof(ctype),
                    self.sizeof,
                    np.size(value),
                )

                return ctype

            elif self.obj.needs_array_desc():
                shape = self.obj.shape
                ndim = self.obj.ndim

                if ctype is None:
                    ctype = _make_fAlloc15(ndim)()

                if value is None:
                    return ctype
                else:
                    shape = value.shape
                    value = self._array_check(value, False)

                    alloc("alloc", ctype, self.type, self.kind, shape)

                    self.copy_array(
                        value.ctypes.data, ctype.base_addr, self.sizeof, np.size(value)
                    )

                    weakref.finalize(
                        ctype,
                        dealloc,
                        "alloc",
                        ctype,
                        self.type,
                        self.kind,
                        shape,
                        head="1",
                    )

                    return ctype

        if ctype is None:
            ctype = self.ctype(value)

        if self.type == "INTEGER":
            return ctype(value)
        elif self.type == "REAL":
            if self.kind == 16:
                print(
                    f"Object of type {self.type} and kind {self.kind} not supported yet, passing None"
                )
                return ctype(None)

            return ctype(value)
        elif self.type == "LOGICAL":
            if value:
                return ctype(1)
            else:
                return ctype(0)
        elif self.type == "CHARACTER":
            strlen = self.len(value).value

            if hasattr(value, "encode"):
                value = value.encode()

            if len(value) > strlen:
                value = value[:strlen]
            else:
                value = value + b" " * (strlen - len(value))

            self._buf = bytearray(value)  # Need to keep hold of the reference

            return ctype.from_buffer(self._buf)
        elif self.type == "COMPLEX":
            return ctype(value.real, value.imag)

        raise NotImplementedError(
            f"Object of type {self.type} and kind {self.kind} not supported yet"
        )

    def len(self, value=None):
        if self.obj.is_char():
            if self.obj.is_defered_len():
                l = len(value)
            else:
                l = self.obj.strlen.value

        elif self.obj.is_array():
            if self.obj.is_assumed_size():
                l = np.size(value)
        else:
            l = None

        return ctypes.c_int64(l)

    @property
    def ctype(self):
        cb_var = None
        cb_arr = None

        if self.type == "INTEGER":
            if self.kind == 4:

                def callback(*args):
                    return ctypes.c_int32

                cb_var = callback
            elif self.kind == 8:

                def callback(*args):
                    return ctypes.c_int64

                cb_var = callback
        elif self.type == "REAL":
            if self.kind == 4:

                def callback(*args):
                    return ctypes.c_float

                cb_var = callback
            elif self.kind == 8:

                def callback(*args):
                    return ctypes.c_double

                cb_var = callback
            elif self.kind == 16:
                # Although we dont support quad yet we can keep things aligned
                def callback(*args):
                    return ctypes.c_ubyte * 16

                cb_var = callback
        elif self.type == "LOGICAL":

            def callback(*args):
                return ctypes.c_int32

            cb_var = callback
        elif self.type == "CHARACTER":
            try:
                strlen = (
                    self.obj.sym.ts.charlen.value
                )  # We know the string length at compile time

                def callback(*args):
                    return ctypes.c_char * strlen

                cb_var = callback
            except AttributeError:

                def callback(
                    value, *args
                ):  # We de not know the string length at compile time
                    return ctypes.c_char * len(value)

                cb_var = callback
        elif self.type == "COMPLEX":
            if self.kind == 4:

                def callback(*args):
                    class complex(ctypes.Structure):
                        _fields_ = [("real", ctypes.c_float), ("imag", ctypes.c_float)]

                    return complex

                cb_var = callback
            elif self.kind == 8:

                def callback(*args):
                    class complex(ctypes.Structure):
                        _fields_ = [
                            ("real", ctypes.c_double),
                            ("imag", ctypes.c_double),
                        ]

                    return complex

                cb_var = callback
            elif self.kind == 16:

                def callback(*args):
                    class complex(ctypes.Structure):
                        _fields_ = [
                            ("real", ctypes.c_ubyte * 16),
                            ("imag", ctypes.c_ubyte * 16),
                        ]

                    return complex

                cb_var = callback

        if self.obj.is_array():
            if self.obj.is_explicit():

                def callback(*args):
                    c = cb_var()
                    for dim in self.obj.shape():
                        c = c * dim

                    return c

                cb_arr = callback
            elif self.obj.is_assumed_size():

                def callback(value, *args):
                    return cb_var() * np.size(value)

                cb_arr = callback

            elif self.obj.needs_array_desc():

                def callback(*args):
                    return _make_fAlloc15(self.obj.ndim)

                cb_arr = callback

        else:
            cb_arr = cb_var

        if cb_arr is None:
            raise NotImplementedError(
                f"Object of type {self.type} and kind {self.kind} not supported yet"
            )
        else:
            return cb_arr

    def from_ctype(self, value):
        if value is None:
            return None

        x = value

        if hasattr(value, "contents"):
            if hasattr(value.contents, "contents"):
                x = value.contents.contents
            else:
                x = value.contents

        if self.obj.is_array():
            if self.obj.is_explicit():
                # If x is a 1d array of prod(shape) then force a reshape
                return np.ctypeslib.as_array(
                    x, shape=self.obj.shape(), order="F", dtype=self.obj.dtype()
                ).T
            elif self.obj.is_assumed_size():
                return np.ctypeslib.as_array(
                    x, shape=self.obj.shape(), order="F", dtype=self.obj.dtype()
                ).T
            elif self.obj.needs_array_desc():
                if x.base_addr is None:
                    return None

                shape = []
                for i in range(self.obj.ndim):
                    shape.append(x.dims[i].ubound - x.dims[i].lbound + 1)

                shape = tuple(shape)

                PTR = ctypes.POINTER(ctypes.c_void_p)
                x_ptr = ctypes.cast(x.base_addr, PTR)

                return np.ctypeslib.as_array(
                    x_ptr, shape=shape, order="F", dtype=self.obj.dtype()
                )

        if self.type == "COMPLEX":
            return complex(x.real, x.imag)

        if hasattr(x, "value") and not self.type == "CHARACTER":
            x = x.value

        if self.type == "INTEGER":
            return int(x)
        elif self.type == "REAL":
            if self.kind == 16:
                raise NotImplementedError(
                    f"Object of type {self.type} and kind {self.kind} not supported yet"
                )
            return float(x)
        elif self.type == "LOGICAL":
            return x == 1
        elif self.type == "CHARACTER":
            return "".join([i.decode() for i in x])
        else:
            raise NotImplementedError(
                f"Object of type {self.type} and kind {self.kind} not supported yet"
            )

    @property
    def __doc__(self):
        return f"{self.obj.head.name}={self.typekind}"

    @property
    def typekind(self):
        if self.type == "INTEGER" or self.type == "REAL":
            return f"{self.type}(KIND={self.kind})"
        elif self.type == "LOGICAL":
            return f"{self.type}"
        elif self.type == "CHARACTER":
            try:
                strlen = (
                    self.obj.sym.ts.charlen.value
                )  # We know the string length at compile time
                return f"{self.type}(LEN={strlen})"
            except AttributeError:
                return f"{self.type}(LEN=:)"

    @property
    def sizeof(self):
        return self.kind

    def set_ctype(self, ctype, value):
        if self.obj.is_array():
            v = self.from_param(value, ctype)
            return
        elif isinstance(ctype, ctypes.Structure):
            for k in ctype.__dir__():
                if not k.startswith("_") and hasattr(value, k):
                    setattr(ctype, k, getattr(value, k))
        else:
            ctype.value = self.from_param(value).value
            return

    def copy_array(self, src, dst, length, size):
        ctypes.memmove(
            dst,
            src,
            length * size,
        )
