# SPDX-License-Identifier: GPL-2.0+
import ctypes
import numpy as np
import platform
import weakref


from .fVar_t import fVar_t
from .utils import copy_array, is_64bit


if is_64bit():
    _index_t = ctypes.c_int64
    _size_t = ctypes.c_int64
else:
    _index_t = ctypes.c_int32
    _size_t = ctypes.c_int32


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


class fArray_t(fVar_t):
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
            if not self.obj.is_allocatable and list(value.shape) != shape:
                raise ValueError(f"Wrong shape, got {value.shape} expected {shape}")

        value = value.ravel(order="F")
        return value

    @property
    def ndim(self):
        return self.obj.ndim


class fExplicitArr(fArray_t):
    def ctype(self):
        return self._ctype_base * self.obj.size

    def from_param(self, value):
        if self.cvalue is None:
            self.cvalue = self.ctype()()

        self._value = self._array_check(value)
        copy_array(
            self._value.ctypes.data,
            ctypes.addressof(self.cvalue),
            ctypes.sizeof(self._ctype_base),
            self.obj.size,
        )
        return self.cvalue

    @property
    def value(self):
        return np.ctypeslib.as_array(self.cvalue).reshape(self.obj.shape(), order="F")

    @value.setter
    def value(self, value):
        self.from_param(value)

    @property
    def __doc__(self):
        return f"{self.type}(KIND={self.kind})({self.obj.shape()}) :: {self.name}"

    def sizeof(self):
        return ctypes.sizeof(self.ctype)

    def len(self):
        return len(self._value)


class fAssumedShape(fArray_t):
    _BT_UNKNOWN = 0
    _BT_INTEGER = _BT_UNKNOWN + 1
    _BT_LOGICAL = _BT_INTEGER + 1
    _BT_REAL = _BT_LOGICAL + 1
    _BT_COMPLEX = _BT_REAL + 1
    _BT_DERIVED = _BT_COMPLEX + 1
    _BT_CHARACTER = _BT_DERIVED + 1
    _BT_CLASS = _BT_CHARACTER + 1
    _BT_PROCEDURE = _BT_CLASS + 1
    _BT_HOLLERITH = _BT_PROCEDURE + 1
    _BT_VOID = _BT_HOLLERITH + 1
    _BT_ASSUMED = _BT_VOID + 1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_array = True

    def ctype(self):
        return _make_fAlloc15(self.obj.ndim)

    def from_param(self, value):
        if self.cvalue is None:
            self.cvalue = self.ctype()()

        if value is not None:
            self._value = self._array_check(value, False)

            # copy_array(
            #     self._value.ctypes.data,
            #     self.cvalue.base_addr,
            #     ctypes.sizeof(self._ctype_base()),
            #     np.size(value)
            # )
            self.cvalue.base_addr = self._value.ctypes.data

            self.cvalue.span = ctypes.sizeof(self._ctype_base())

            strides = []
            shape = np.shape(value)
            for i in range(self.ndim):
                self.cvalue.dims[i].lbound = _index_t(1)
                self.cvalue.dims[i].ubound = _index_t(shape[i])
                strides.append(
                    self.cvalue.dims[i].ubound - self.cvalue.dims[i].lbound + 1
                )

            spans = []
            for i in range(self.ndim):
                spans.append(int(np.prod(strides[:i])))
                self.cvalue.dims[i].stride = _index_t(spans[-1])

            self.cvalue.offset = -np.sum(spans)

        self.cvalue.dtype.elem_len = self.cvalue.span
        self.cvalue.dtype.version = 0
        self.cvalue.dtype.rank = self.ndim
        self.cvalue.dtype.type = self.ftype()
        self.cvalue.dtype.attribute = 0

        return self.cvalue

    @property
    def value(self):
        if self.cvalue.base_addr is None:
            return None

        shape = []
        for i in range(self.obj.ndim):
            shape.append(self.cvalue.dims[i].ubound - self.cvalue.dims[i].lbound + 1)

        shape = tuple(shape)
        size = (np.prod(shape),)

        PTR = ctypes.POINTER(self._ctype_base)
        x_ptr = ctypes.cast(self.cvalue.base_addr, PTR)

        array = self._make_empty(shape)

        copy_array(
            self.cvalue.base_addr,
            array.ctypes.data,
            ctypes.sizeof(self._ctype_base()),
            size[0],
        )

        # self._array = np.ctypeslib.as_array(x_ptr, shape=size).reshape(shape, order="F")

        return array

    @value.setter
    def value(self, value):
        self.from_param(value)

    @property
    def __doc__(self):
        return f"{self.type}(KIND={self.kind})(:) :: {self.name}"

    def ftype(self):
        if self.obj.type() == "INTEGER":
            return self._BT_INTEGER
        elif self.obj.type() == "LOGICAL":
            return self._BT_LOGICAL
        elif self.obj.type() == "REAL":
            return self._BT_REAL
        elif self.obj.type() == "COMPLEX":
            return self._BT_COMPLEX
        elif self.obj.type() == "CHARACTER":
            return self._BT_CHARACTER

        raise NotImplementedError(
            f"Assumed shape array of type {self.type} and kind {self.kind} not supported yet"
        )

    def __del__(self):
        if self.cvalue is not None:
            self.cvalue.base_addr = None

    def print(self):
        if self.cvalue is None:
            return ""

        print(f"base_addr {self.cvalue.base_addr}")
        print(f"offset {self.cvalue.offset}")
        print(f"dtype")
        print(f"\t elem_len {self.cvalue.dtype.elem_len}")
        print(f"\t version {self.cvalue.dtype.version}")
        print(f"\t rank {self.cvalue.dtype.rank}")
        print(f"\t type {self.cvalue.dtype.type}")
        print(f"\t attribute {self.cvalue.dtype.attribute}")
        print(f"span {self.cvalue.span}")
        print(f"dims {self.ndim}")
        for i in range(self.ndim):
            print(f"\t lbound {self.cvalue.dims[i].lbound}")
            print(f"\t ubound {self.cvalue.dims[i].ubound}")
            print(f"\t stride {self.cvalue.dims[i].stride}")

    def _make_empty(self, shape=None):
        dtype = self.obj.dtype()
        if shape is None:
            shape = self.obj.shape()

        return np.zeros(shape, dtype=dtype, order="F")


class fAssumedSize(fArray_t):
    def ctype(self):
        return self._ctype_base * np.prod(self._value.shape)

    def from_param(self, value):
        self._value = self._array_check(value)
        if self.cvalue is None:
            self.cvalue = self.ctype()()

        copy_array(
            self._value.ctypes.data,
            ctypes.addressof(self.cvalue),
            ctypes.sizeof(self._ctype_base),
            np.size(value),
        )
        return self.cvalue

    @property
    def value(self):
        return np.ctypeslib.as_array(self.cvalue, shape=np.size(self._value)).reshape(
            self._value.shape, order="F"
        )

    @value.setter
    def value(self, value):
        self.from_param(value)

    @property
    def __doc__(self):
        return f"{self.type}(KIND={self.kind})(*) :: {self.name}"

    def sizeof(self):
        return ctypes.sizeof(self.ctype)

    def len(self):
        return len(self._value)

    def ctype_len(self, *args):
        return ctypes.c_int64(self.len())

    @property
    def ndim(self):
        return 1
