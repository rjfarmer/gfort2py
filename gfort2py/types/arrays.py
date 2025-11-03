# SPDX-License-Identifier: GPL-2.0+

import ctypes
import numpy as np
from abc import ABCMeta, abstractmethod
from typing import Type, Tuple

import gfModParser as gf

from .base import f_type

from ..utils import copy_array, is_64bit
from ..allocate import allocate_var, allocate_char
from .character import ftype_character


class ftype_explicit_array(f_type, metaclass=ABCMeta):
    dtype = None
    ftype = None
    kind = None

    def __init__(self, value=None):
        self.base = self._base()
        super().__init__(value=value)

    @abstractmethod
    def _base(self):
        raise NotImplementedError

    @property
    def ctype(self):
        return self.base.ctype * self.size

    def __repr__(self):
        s = ",".join([str(i) for i in self.shape])
        return f"{self.base.ftype}(kind={self.base.kind})({s})"

    @property
    def value(self) -> np.ndarray:
        self._value = (
            np.ctypeslib.as_array(self._ctype)
            .reshape(self.shape, order="F")
            .astype(self.base.dtype)
        )
        return self._value

    @value.setter
    def value(self, value: np.ndarray):
        if value is None:
            return None

        self._value = self._array_check(value)
        copy_array(
            self._value.ctypes.data,
            ctypes.addressof(self._ctype),
            ctypes.sizeof(self.base.ctype),
            self.size,
        )

    def _array_check(self, value):
        value = np.asfortranarray(value)
        value = value.astype(self.base.dtype, copy=False)

        if value.ndim != self.ndims:
            raise ValueError(
                f"Wrong number of dimensions, got {value.ndim} expected {self.ndims}"
            )

        value = value.ravel(order="F")
        return value

    @property
    def shape(self) -> tuple[int, ...]:
        return self.definition().properties.array_spec.pyshape

    @property
    def ndims(self) -> int:
        return self.definition().properties.array_spec.rank

    @property
    def size(self) -> int:
        return np.prod(self.shape)


class ftype_assumed_shape(f_type, metaclass=ABCMeta):
    dtype = None
    ftype = None
    kind = None

    def _init__(self, value=None):
        self._value = value
        super().__init__()

    @abstractmethod
    def _base(self):
        raise NotImplementedError

    @property
    def base(self):
        return self._base()

    @property
    def ctype(self):
        if is_64bit():
            _index_t = ctypes.c_int64
            _size_t = ctypes.c_int64
        else:
            _index_t = ctypes.c_int32
            _size_t = ctypes.c_int32

        class _bounds14(ctypes.Structure):
            _fields_ = [
                ("stride", _index_t),
                ("lbound", _index_t),
                ("ubound", _index_t),
            ]

        class _dtype_type(ctypes.Structure):
            _fields_ = [
                ("elem_len", _size_t),
                ("version", ctypes.c_int32),
                ("rank", ctypes.c_byte),
                ("type", ctypes.c_byte),
                ("attribute", ctypes.c_ushort),
            ]

        class _fAllocArray(ctypes.Structure):
            _fields_ = [
                ("base_addr", ctypes.c_void_p),
                ("offset", _size_t),
                ("dtype", _dtype_type),
                ("span", _index_t),
                ("dims", _bounds14 * self.ndims),
            ]

        return _fAllocArray

    def __repr__(self):
        s = ",".join([":" for i in range(self.ndims)])
        return f"{self.ftype}(kind={self.kind})({s})"

    @property
    def value(self) -> np.ndarray:
        if self.ctype.base_addr is None:
            return None

        shape = []
        for i in range(self.ndims):
            shape.append(self.ctype.dims[i].ubound - self.ctype.dims[i].lbound + 1)

        shape = tuple(shape)

        array = np.zeros(shape, dtype=self.base.dtype, order="F")

        copy_array(
            self.ctype.base_addr,
            array.ctypes.data,
            ctypes.sizeof(self.base.ctype),
            np.prod(shape),
        )
        self._value = array
        return array

    @value.setter
    def value(self, value: np.ndarray):
        shape = np.shape(value)

        self._allocate(shape)

        self._value = np.asfortranarray(value).ravel("F")
        copy_array(
            self._value.ctypes.data,
            ctypes.addressof(self.ctype),
            ctypes.sizeof(self.base.ctype),
            np.prod(shape),
        )

    @abstractmethod
    def _allocate(self, shape):
        raise NotImplementedError

    @property
    def shape(self) -> tuple[int, ...]:
        return self.definition().properties.array_spec.pyshape

    @property
    def ndims(self) -> int:
        return self.definition().properties.array_spec.rank


class ftype_character_assumed_shape(ftype_assumed_shape):

    def _allocate(self, shape):
        allocate_char(
            self.ctype,
            kind=self.kind,
            length=self.base.len(),
            shape=shape,
            default=self.base.default,
        )


class ftype_number_assumed_shape(ftype_assumed_shape):

    def _allocate(self, shape):
        print(self._base)

        allocate_var(
            self.ctype,
            kind=self.kind,
            type=self.base.ftype,
            shape=shape,
            default=self.base.default,
        )
