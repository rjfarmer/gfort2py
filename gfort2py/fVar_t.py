# SPDX-License-Identifier: GPL-2.0+
import ctypes
import numpy as np

from .fUnary import run_unary

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

def _make_dt():
    class _fDerivedType(ctypes.Structure):
        pass
    return _fDerivedType

class fVar_t():
    def __init__(self, obj, cvalue=None):
        self.obj = obj
        self._cvalue = cvalue

        self.type, self.kind = self.obj.type_kind()

        self._ctype_base = ctype_map(self.type, self.kind)

    @property
    def name(self):
        return self.obj.name

    @property
    def mangled_name(self):
        return self.obj.mangled_name

    @property
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
            if not self.obj.is_allocatable and list(value.shape) != shape:
                raise ValueError(f"Wrong shape, got {value.shape} expected {shape}")

        value = value.ravel(order="F")
        return value

    def ctype_len(self):
        return None

    def from_ctype(self, ct):
        self._cvalue = ct
        return self.value

    def from_address(self, addr):
        self._cvalue = self.ctype().from_address(addr)
        return self._cvalue

    def in_dll(self, lib):
        self._cvalue =  self.ctype().in_dll(lib, self.mangled_name)
        return  self._cvalue 

class fScalar(fVar_t):
    def ctype(self):
        return self._ctype_base

    def from_param(self, param):
        if self._cvalue is None:
            self._cvalue = self.ctype()(param)
        else:
            self._cvalue.value = param
        return self._cvalue

    @property
    def value(self):
        x = self._cvalue.value

        if self.type == "INTEGER":
            return int(x)
        elif self.type == "REAL":
            if self.kind == 16:
                raise NotImplementedError(
                    f"Quad precision floats not supported yet"
                )
            return float(x)
        elif self.type == "LOGICAL":
            return x == 1

    @value.setter
    def value(self, value):
        self.from_param(value)

    def sizeof(self):
        return ctypes.sizeof(self.ctype)

    def __doc__(self):
        return f"{self.type}(KIND={self.kind}) :: {self.name}"


class fCmplx(fVar_t):
    def ctype(self):
        return self._ctype_base

    def from_param(self, param):
        if self._cvalue is None:
            self._cvalue = self.ctype()()

        self._cvalue.real = param.real
        self._cvalue.imag = param.imag
        return self._cvalue

    @property
    def value(self):
        x = self._cvalue

        if self.kind == 16:
            raise NotImplementedError(
                f"Quad precision complex numbers not supported yet"
            )
        return complex(x.real, x.imag)

    @value.setter
    def value(self, value):
        self.from_param(value)

    def sizeof(self):
        return ctypes.sizeof(self.ctype)

    def __doc__(self):
        return f"{self.type}(KIND={self.kind}) :: {self.name}"



class fExplicitArr(fVar_t):
    def ctype(self):
        return self._ctype_base * self.obj.size

    def from_param(self, value):
        if self._cvalue is None:
            self._cvalue = self.ctype()()

        self._value = self._array_check(value)
        _copy_array(
            self._value.ctypes.data,
            ctypes.addressof(self._cvalue),
            ctypes.sizeof(self._ctype_base),
            self.obj.size,
        )
        return self._cvalue

    @property
    def value(self):
        return np.ctypeslib.as_array(self._cvalue).reshape(self.obj.shape(),order='F')

    @value.setter
    def value(self, value):
        self.from_param(value)

    def __doc__(self):
        return f"{self.type}(KIND={self.kind})({self.obj.shape()}) :: {self.name}"

    def sizeof(self):
        return ctypes.sizeof(self.ctype)

    def len(self):
        return len(self._value)


class fAssumedShape(fVar_t):

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

    def ctype(self):
        return _make_fAlloc15(self.obj.ndim)

    def from_param(self, value):
        if self._cvalue is None:
            self._cvalue = self.ctype()()

        if value is not None:
            self._value = self._array_check(value, False)

            _copy_array(
                self._value.ctypes.data, 
                self._cvalue.base_addr, 
                ctypes.sizeof(self._ctype_base()), 
                np.size(value)
            )

            self._cvalue.offset = -np.prod(np.shape(value))
            self._cvalue.span = ctypes.sizeof(self._ctype_base())

            strides = []
            shape = np.shape(value)
            for i in range(self.ndim):
                self._cvalue.dims[i].lbound = _index_t(1)
                self._cvalue.dims[i].ubound = _index_t(shape[i]) 
                strides.append(self._cvalue.dims[i].ubound - self._cvalue.dims[i].lbound + 1)

            for i in range(self.ndim):
                self._cvalue.dims[i].span = _index_t(int(np.prod(strides[:i])))

        self._cvalue.dtype.elem_len = self._cvalue.span
        self._cvalue.dtype.version = 0
        self._cvalue.dtype.rank = self.ndim
        self._cvalue.dtype.type = self.ftype()
        self._cvalue.dtype.attribute = 0

        return self._cvalue

    @property
    def value(self):
        if self._cvalue.base_addr is None:
            return None

        shape = []
        for i in range(self.obj.ndim):
            shape.append(self._cvalue.dims[i].ubound - self._cvalue.dims[i].lbound + 1)

        shape = tuple(shape)
        size = (np.prod(shape),)

        PTR = ctypes.POINTER(self._ctype_base)
        x_ptr = ctypes.cast(self._cvalue.base_addr, PTR)

        return np.ctypeslib.as_array(x_ptr,shape=size).reshape(shape,order='F')


    @value.setter
    def value(self, value):
        self.from_param(value)

    def __doc__(self):
        return f"{self.type}(KIND={self.kind})(:) :: {self.name}"

    @property
    def ndim(self):
        return self.obj.ndim


    def ftype(self):
        if self.obj.type() == "INTEGER":
            return self._BT_INTEGER
        elif self.obj.type() == "LOGICAL":
            return self._BT_LOGICAL
        elif self.obj.type() == "REAL":
            return self._BT_REAL
        elif self.obj.type() == "COMPLEX":
            return self._BT_COMPLEX

        raise NotImplementedError(f"Assume shape array of type {self.type} and kind {self.kind} not supported yet")



class fAssumedSize(fVar_t):
    def ctype(self):
        return self._ctype_base() * np.prod(self._value.shape())

    def from_param(self, value):
        if self._cvalue is None:
            self._cvalue = self.ctype()()

        self._value = self._array_check(value)
        _copy_array(
            self._value.ctypes.data,
            ctypes.addressof(self._cvalue),
            self.sizeof,
            np.size(value),
        )
        return self._cvalue

    @property
    def value(self):
        return np.ctypeslib.as_array(self._cvalue,shape=np.prod(self.obj.shape())).reshape(self.obj.shape(),order='F')

    @value.setter
    def value(self, value):
        self.from_param(value)

    def __doc__(self):
        return f"{self.type}(KIND={self.kind})(*) :: {self.name}"

    def sizeof(self):
        return ctypes.sizeof(self.ctype)

    def len(self):
        return len(self._value)

    def ctype_len(self):
        return ctypes.c_int64(self.len())


class fStr(fVar_t):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self._len = None

    def ctype(self):
        return self._ctype_base * self.len()

    def from_param(self, value):
        if self.obj.is_deferred_len():
            self._len = len(value)

        if self._cvalue is None:
            self._cvalue = self.ctype()()

        self._value = value

        if hasattr(self._value, "encode"):
            self._value = self._value.encode()

        if len(self._value) > self.len():
            self._value = self._value[:self.len()]
        else:
            self._value = self._value + b" " * (self.len() - len(self._value))

        #self._buf = bytearray(self._value)  # Need to keep hold of the reference
        self._cvalue.value = self._value

        return self._cvalue

    @property
    def value(self):
        try:
            return self._cvalue.value.decode()
        except AttributeError:
            return str(self._cvalue) # Functions returning str's give us str not bytes

    @value.setter
    def value(self, value):
        self.from_param(value)

    def len(self):
        if self._len is None:
            if self.obj.is_deferred_len():
                self._len = len(self._cvalue)
            else:
                self._len = self.obj.strlen.value
        return self._len

    def ctype_len(self):
        return ctypes.c_int64(self.len())

    def __doc__(self):
        try:
            return f"{self.type}(LEN={self.obj.strlen}) :: {self.name}"
        except AttributeError:
            return f"{self.type}(LEN=:) :: {self.name}"

    def sizeof(self):
        return ctypes.sizeof(self.ctype)




def _copy_array(src, dst, length, size):
    ctypes.memmove(
        dst,
        src,
        length * size,
    )


def ctype_map(type, kind):
    if type == "INTEGER":
        if kind == 4:
            return ctypes.c_int32
        elif kind == 8:
            return ctypes.c_int64
        else:
            raise TypeError("Integer type of kind={kind} not supported")
    elif type == "REAL":
        if kind == 4:
            return ctypes.c_float
        elif kind == 8:
            return ctypes.c_double
        elif kind == 16:
            # Although we dont support quad yet we can keep things aligned
            return ctypes.c_ubyte * 16
        else:
            raise TypeError("Float type of kind={kind} not supported")
    elif type == "LOGICAL":
        return ctypes.c_int32
    elif type == "CHARACTER":
        return ctypes.c_char
    elif type == "COMPLEX":
        if kind == 4:
            ct = ctypes.c_float
        elif kind == 8:
            ct = ctypes.c_double
        elif kind == 16:
            ct = ctypes.c_ubyte * 16
        else:
            raise TypeError("Complex type of kind={kind} not supported")

        class complex(ctypes.Structure):
                _fields_ = [
                    ("real", ct),
                    ("imag", ct),
                ]
        return complex
    else:
        raise TypeError(f"Type={type} and kind={kind} not supported")

