# SPDX-License-Identifier: GPL-2.0+

import ctypes
import numpy as np
from abc import ABCMeta, abstractmethod

from .base import f_type

from ..utils import copy_array, is_64bit
from ..allocate import allocate_var, allocate_char


class f_explicit_array(f_type, metaclass=ABCMeta):
    dtype = None

    @property
    @abstractmethod
    def _base(self):
        raise NotImplementedError

    def _init__(self, base: f_type, shape=None, value=None):
        self._base = base
        self._value = value
        if shape is None:
            shape = self._value.shape()
        self.shape = shape
        super().__init__()

    @property
    def ctype(self):
        return self._base.ctype * np.product(self.shape)

    def __repr__(self):
        s = ",".join([i for i in self.shape])
        return f"{self.ftype}(kind={self.kind})({s})"

    @property
    def value(self):
        self._value = (
            np.ctypeslib.as_array(self._ctype)
            .reshape(self.shape, order="F")
            .as_dtype(self._base.dtype)
        )
        return self._value

    @value.setter
    def value(self, value):
        self._value = np.asfortranarray(value).ravel("F")
        copy_array(
            self._value.ctypes.data,
            ctypes.addressof(self._ctype),
            ctypes.sizeof(self._base.ctype),
            np.product(self.shape),
        )


class f_assumed_shape(f_type, metaclass=ABCMeta):
    dtype = None

    def _init__(self, base: f_type, value=None, ndims=None):
        self._base = base
        self._value = value
        if ndims is None:
            ndims = value.ndim
        self.ndims = ndims
        super().__init__()

    @property
    @abstractmethod
    def _base(self):
        raise NotImplementedError

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
    def value(self):
        if self.ctype.base_addr is None:
            return None

        shape = []
        for i in range(self.ndims):
            shape.append(self.ctype.dims[i].ubound - self.ctype.dims[i].lbound + 1)

        shape = tuple(shape)

        array = np.zeros(shape, dtype=self._base.dtype, order="F")

        copy_array(
            self.ctype.base_addr,
            array.ctypes.data,
            ctypes.sizeof(self._base.ctype),
            np.product(shape),
        )
        self._value = array
        return array

    @value.setter
    def value(self, value):
        shape = np.shape(value)

        self._allocate(shape)

        self._value = np.asfortranarray(value).ravel("F")
        copy_array(
            self._value.ctypes.data,
            ctypes.addressof(self.ctype),
            ctypes.sizeof(self._base.ctype),
            np.product(shape),
        )

    @abstractmethod
    def _allocate(self, shape):
        raise NotImplementedError


class f_character_assumed_shape(f_assumed_shape):
    def _allocate(self, shape):
        allocate_char(
            self.ctype,
            kind=self.kind,
            length=self._base.len(),
            shape=shape,
            default=self._base.default,
        )


class f_number_assumed_shape(f_assumed_shape):
    def _allocate(self, shape):
        allocate_var(
            self.ctype,
            kind=self.kind,
            type=self._base.ftype,
            shape=shape,
            default=self._base.default,
        )
