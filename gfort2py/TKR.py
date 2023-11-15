from abc import ABC, abstractmethod
import ctypes
import numpy as np
import weakref
import functools

from gfort2py.module_parse import fTypeEnum, ArrayEnum
from gfort2py.utils import copy_array

try:
    import pyquadp as pyq

    PYQ_IMPORTED = True
except ImportError:
    PYQ_IMPORTED = False


##################################
# Base Classes
##################################


class fObject(ABC):
    def __init__(self, *args, **kwargs):
        pass

    @property
    @abstractmethod
    def ctype(self):
        """
        Returns the cvalue type
        """
        pass

    @abstractmethod
    def from_param(self, value):
        """
        Converts a python value into a ctype(value) making a new ctype object
        """
        pass

    @abstractmethod
    def set_ctype(self, ctype, value):
        """
        Sets a ctype with a Python value
        """
        pass

    @abstractmethod
    def from_ctype(self, cvalue):
        """
        Converts a cvalue object into a python object
        """
        pass

    @abstractmethod
    def from_maybe_ctype(self, cvalue):
        """
        Sometimes ctypes will *helpfully* convert the
        cvalue into a Python value. This makes converting
        things annoying as sometimes you have a Python thing
        and sometimes you have a ctype.

        So this is the version of from_ctype that also
        checks if cvalue is already a Python object
        and returns that.
        """
        pass

    def from_address(self, addr):
        """
        Gets cvalue from memory address
        """
        return self.ctype.from_address(addr)

    def in_dll(self, lib, name):
        """
        Look up variable name in library lib
        """
        return self.ctype.in_dll(lib, name)


class f_scalar(fObject):
    """
    Class for simply scalar variables
    """

    def from_param(self, value):
        ctype = self.ctype()
        return self.set_ctype(ctype, value)

    def set_ctype(self, ctype, value):
        """
        Sets a ctype with a given Python value
        """
        ctype.value = value
        return ctype


class f_array_explicit(fObject):
    """
    Class for explicit arrays (dimension(5) or dimension(n))

    Note Assumed size arrays can also be handled this way.

    Note this assumes python knows the shape
    """

    def __init__(self, shape, *args, **kwargs):
        self.shape = shape
        self.size = np.product(shape)

    def from_ctype(self, cvalue):
        return np.ctypeslib.as_array(cvalue).reshape(self.shape, order="F")

    def from_param(self, value):
        """
        Return a ctype pointer to the data in value
        """
        return value.ctypes.data_as(ctypes.c_void_p)

    def set_ctype(self, ctype, value):
        """
        Copies value into ctype

        Use when setting module variables, use from_param when passing to functions
        so that we avoid the copy.
        """

        if list(value.shape) != self.shape:
            raise ValueError(f"Wrong shape, got {value.shape} expected {self.shape}")

        value = value.astype(self.dtype)

        if not value.flags["F_CONTIGUOUS"]:
            value = np.asfortranarray(value)

        value = value.ravel(order="F")

        self._value = value

        # print(ctype)
        # print(self._value.size,ctypes.sizeof(self.ctype))

        copy_array(
            self._value.ctypes.data,
            ctypes.addressof(ctype),
            ctypes.sizeof(self.elem),
            self._value.size,
        )

        return ctype

    @property
    @abstractmethod
    def dtype(self):
        """
        This should be a Numpy dtype type.
        """
        pass

    @property
    @abstractmethod
    def elem(self):
        """
        Single element of array
        """
        pass

    @property
    def ctype(self):
        return self.elem * self.size

    def from_maybe_ctype(self, *args):
        return self.from_ctype(*args)


##################################
# Integers
##################################


class f_integer(f_scalar):
    def from_ctype(self, cvalue):
        return int(cvalue.value)

    @functools.cached_property
    def ctype(self):
        return ctypes.c_int32

    def from_maybe_ctype(self, cvalue):
        if isinstance(cvalue, int):
            return cvalue
        else:
            return self.from_ctype(cvalue)


class f_integer_1(f_integer):
    @functools.cached_property
    def ctype(self):
        return ctypes.c_int8


class f_integer_2(f_integer):
    @functools.cached_property
    def ctype(self):
        return ctypes.c_int16


f_integer_4 = f_integer


class f_integer_8(f_integer):
    @functools.cached_property
    def ctype(self):
        return ctypes.c_int64


##################################
# Reals
##################################


class f_real(f_scalar):
    def from_ctype(self, cvalue):
        return float(cvalue.value)

    @functools.cached_property
    def ctype(self):
        return ctypes.c_float

    def from_maybe_ctype(self, cvalue):
        if isinstance(cvalue, float):
            return cvalue
        else:
            return self.from_ctype(cvalue)


f_real_4 = f_real


class f_real_8(f_real):
    @functools.cached_property
    def ctype(self):
        return ctypes.c_double


class f_real_16(f_real):
    @functools.cached_property
    def ctype(self):
        return ctypes.c_char * 16

    def from_ctype(self, cvalue):
        if PYQ_IMPORTED:
            return pyq.qfloat.from_bytes(bytes(cvalue))
        else:
            raise TypeError(f"Quad precision floats requires pyQuadp to be installed")

    def from_param(self, value):
        if PYQ_IMPORTED:
            return pyq.qfloat(value).to_bytes()
        else:
            raise NotImplementedError(
                f"Quad precision floats requires pyQuadp to be installed"
            )

    def set_ctype(self, ctype, value):
        """
        Sets a ctype with a given Python value
        """
        # Don't use from_param().value
        ctype.value = self.from_param(value)
        return ctype


##################################
# Logicals
##################################


class f_logical(f_scalar):
    def from_ctype(self, cvalue):
        return cvalue.value == 1

    def from_param(self, value):
        if value:
            return self.ctype(1)
        else:
            return self.ctype(0)

    @functools.cached_property
    def ctype(self):
        return ctypes.c_int32

    def from_maybe_ctype(self, cvalue):
        if isinstance(cvalue, int):
            return cvalue == 1
        else:
            return self.from_ctype(cvalue)


f_logical_4 = f_logical


##################################
# Complex
##################################


class f_complex(f_scalar):
    """
    Base class for complex numbers

    Treats complex numbers as a pair of values (real,imag)
    """

    def from_ctype(self, cvalue):
        return complex(cvalue.real, cvalue.imag)

    def from_param(self, value):
        ctype = self.ctype()
        return self.set_ctype(ctype, value)

    @functools.cached_property
    def ctype(self):
        class _complex(ctypes.Structure):
            _fields_ = [
                ("real", ctypes.c_float),
                ("imag", ctypes.c_float),
            ]

        return _complex

    def set_ctype(self, ctype, value):
        """
        Sets a ctype with a given Python value
        """
        ctype.real = value.real
        ctype.imag = value.imag
        return ctype

    def from_maybe_ctype(self, cvalue):
        return self.from_ctype(cvalue)


f_complex_4 = f_complex


class f_complex_8(f_complex):
    @functools.cached_property
    def ctype(self):
        class _complex(ctypes.Structure):
            _fields_ = [
                ("real", ctypes.c_double),
                ("imag", ctypes.c_double),
            ]

        return _complex


class f_complex_16(f_complex):
    @functools.cached_property
    def ctype(self):
        return ctypes.c_char * 16

    def from_ctype(self, cvalue):
        if PYQ_IMPORTED:
            return pyq.qcmplx.from_bytes(bytes(cvalue))
        else:
            raise TypeError(f"Quad precision complex requires pyQuadp to be installed")

    def from_param(self, value):
        if PYQ_IMPORTED:
            return pyq.qcmplx(value).to_bytes()
        else:
            raise NotImplementedError(
                f"Quad precision complex requires pyQuadp to be installed"
            )


##################################
# Strings
##################################


class f_character(fObject):
    def __init__(self, len=None, *args, **kwargs):
        self.len = len

    @functools.cached_property
    def ctype(self):
        """
        We could do ctypes.c_char * self.len, but
        this makes allocatable characters harder
        as they just want a pointer
        """
        return ctypes.c_char_p

    def from_param(self, value):
        ctype = self.ctype()
        return self.set_ctype(ctype, value)

    def from_ctype(self, cvalue):
        if cvalue.value is None:
            return None

        v = cvalue.value[: self.len]

        return v.decode("ascii")

    def set_ctype(self, ctype, value):
        """
        Sets a ctype with a given Python value
        """
        if isinstance(value, str):
            value = value.encode("ascii")

        if len(value) > self.len:
            value = value[: self.len]
        elif len(value) < self.len:
            # Pad string to correct length
            value = value + b" " * (self.len - len(value))

        ctype.value = value
        # Must hold onto the memory
        self._ctype = ctype
        return ctype

    def from_maybe_ctype(self, cvalue):
        return self.from_ctype(cvalue)


##################################
# Arrays Integers
##################################


class f_integer_array_explicit(f_array_explicit):
    @functools.cached_property
    def dtype(self):
        return np.int32

    @functools.cached_property
    def elem(self):
        return ctypes.c_int32


class f_integer_1_array_explicit(f_integer_array_explicit):
    @functools.cached_property
    def elem(self):
        return ctypes.c_int8

    @functools.cached_property
    def dtype(self):
        return np.int8


class f_integer_2_array_explicit(f_integer_array_explicit):
    @functools.cached_property
    def elem(self):
        return ctypes.c_int16

    @functools.cached_property
    def dtype(self):
        return np.int16


f_integer_4_array_explicit = f_integer_array_explicit


class f_integer_8_array_explicit(f_integer_array_explicit):
    @functools.cached_property
    def elem(self):
        return ctypes.c_int64

    @functools.cached_property
    def dtype(self):
        return np.int64


##################################
# Arrays reals
##################################


class f_real_array_explicit(f_array_explicit):
    @functools.cached_property
    def dtype(self):
        return np.float32

    @functools.cached_property
    def elem(self):
        return ctypes.c_float


f_real_4_array_explicit = f_real_array_explicit


class f_real_8_array_explicit(f_real_array_explicit):
    @functools.cached_property
    def elem(self):
        return ctypes.c_double

    @functools.cached_property
    def dtype(self):
        return np.float64


##################################
# Arrays complex
##################################


class f_complex_array_explicit(f_array_explicit):
    @functools.cached_property
    def dtype(self):
        return np.complex64

    @functools.cached_property
    def elem(self):
        class _complex(ctypes.Structure):
            _fields_ = [
                ("real", ctypes.c_float),
                ("imag", ctypes.c_float),
            ]

        return _complex


f_complex_4_array_explicit = f_complex_array_explicit


class f_complex_8_array_explicit(f_complex_array_explicit):
    @functools.cached_property
    def elem(self):
        class _complex(ctypes.Structure):
            _fields_ = [
                ("real", ctypes.c_double),
                ("imag", ctypes.c_double),
            ]

        return _complex

    @functools.cached_property
    def dtype(self):
        return np.complex128


##################################
# Arrays Logicals
##################################


class f_logical_array_explicit(f_array_explicit):
    @functools.cached_property
    def dtype(self):
        return np.int32

    @functools.cached_property
    def elem(self):
        return ctypes.c_int32

    def from_ctype(self, cvalue):
        array = super().from_param(cvalue)
        return array.astype(bool)


f_logical_4_array_explicit = f_logical_array_explicit


##################################
# Arrays character
##################################
class f_character_array_explicit(f_array_explicit):
    def __init__(self, shape, len, ndim=None, *args, **kwargs):
        self.shape = shape
        self.len = len
        self.size = np.product(self.shape)

    @functools.cached_property
    def dtype(self):
        return f"S{self.len}"

    @functools.cached_property
    def elem(self):
        return ctypes.c_char

    # Needs convert str to bytes and back
    def from_ctype(self, cvalue):
        raise NotImplementedError

    def from_param(self, value):
        raise NotImplementedError

    def set_ctype(self, ctype, value):
        raise NotImplementedError


##################################
# Arrays Assumed shape
##################################


class f_array_assumed_shape(fObject):
    """
    Handles assumed shape arrays (dimension(:))

    """

    _index_t = ctypes.c_int64
    _size_t = ctypes.c_int64

    def __init__(self, shape=None, ndim=None, *args, **kwargs):
        self.shape = shape

        if ndim is None:
            self.ndim = len(self.shape)
        else:
            self.ndim = ndim

    def _make_fAlloc15(self, ndims):
        class _bounds14(ctypes.Structure):
            _fields_ = [
                ("stride", self._index_t),
                ("lbound", self._index_t),
                ("ubound", self._index_t),
            ]

        class _dtype_type(ctypes.Structure):
            _fields_ = [
                ("elem_len", self._size_t),
                ("version", ctypes.c_int32),
                ("rank", ctypes.c_byte),
                ("type", ctypes.c_byte),
                ("attribute", ctypes.c_ushort),
            ]

        class _fAllocArray(ctypes.Structure):
            _fields_ = [
                ("base_addr", ctypes.c_void_p),
                ("offset", self._size_t),
                ("dtype", _dtype_type),
                ("span", self._index_t),
                ("dims", _bounds14 * ndims),
            ]

        return _fAllocArray

    @property
    def ctype(self):
        return self._make_fAlloc15(self.ndim)

    @property
    @abstractmethod
    def elem(self):
        """
        cvalue of single elemement of the array
        """
        pass

    @property
    def sizeof(self):
        """
        Returns the size of a single element (ctypes.sizeof(ctypes.c_int32))
        """
        return ctypes.sizeof(self.elem)

    @property
    @abstractmethod
    def ftype(self):
        """
        Returns the fortran enum type (fTypeEnum)
        """
        pass

    @property
    @abstractmethod
    def dtype(self):
        pass

    def from_param(self, value):
        self.ndim = len(np.shape(value))

        cvalue = self.ctype()

        return self.set_ctype(cvalue, value)

    def set_ctype(self, ctype, value):
        if len(self.shape) == 0:
            self.shape = np.shape(value)

        self.ndim = len(self.shape)

        if value is None:
            # Passed an unallocated array
            ctype.base_addr = None
            return

        value = value.astype(self.dtype)

        if not value.flags["F_CONTIGUOUS"]:
            value = np.asfortranarray(value)

        self._value = value.ravel(order="F")

        ctype.base_addr = self._value.ctypes.data

        ctype.span = self.sizeof

        strides = []

        for i in range(self.ndim):
            ctype.dims[i].lbound = self._index_t(1)
            ctype.dims[i].ubound = self._index_t(self.shape[i])
            strides.append(ctype.dims[i].ubound - ctype.dims[i].lbound + 1)

        spans = []
        for i in range(self.ndim):
            spans.append(int(np.prod(strides[:i])))
            ctype.dims[i].stride = self._index_t(spans[-1])

        ctype.offset = -np.sum(spans)

        ctype.dtype.elem_len = ctype.span
        ctype.dtype.version = 0
        ctype.dtype.rank = self.ndim
        ctype.dtype.type = self.ftype.value
        ctype.dtype.attribute = 0

        return ctype

    def from_ctype(self, cvalue):
        if cvalue is None:
            return None

        if cvalue.base_addr is None:
            # un-allocated array
            return None

        ndim = cvalue.dtype.rank

        shape = []
        for i in range(ndim):
            shape.append(cvalue.dims[i].ubound - cvalue.dims[i].lbound + 1)

        self.shape = tuple(shape)

        size = np.prod(shape)

        new_array = np.zeros(shape, dtype=self.dtype, order="F")
        copy_array(cvalue.base_addr, new_array.ctypes.data, size, new_array.itemsize)

        return new_array

    def from_maybe_ctype(self, cvalue):
        return self.from_ctype(cvalue)

    @staticmethod
    def print(cvalue):
        if cvalue is None:
            return ""

        print(f"base_addr {cvalue.base_addr}")
        print(f"offset {cvalue.offset}")
        print(f"dtype")
        print(f"\t elem_len {cvalue.dtype.elem_len}")
        print(f"\t version {cvalue.dtype.version}")
        print(f"\t rank {cvalue.dtype.rank}")
        print(f"\t type {cvalue.dtype.type}")
        print(f"\t attribute {cvalue.dtype.attribute}")
        print(f"span {cvalue.span}")
        print(f"dims {cvalue.dtype.rank}")
        for i in range(cvalue.dtype.rank):
            print(f"\t lbound {cvalue.dims[i].lbound}")
            print(f"\t ubound {cvalue.dims[i].ubound}")
            print(f"\t stride {cvalue.dims[i].stride}")


#######################
# Assumed shape Ints
#######################


class f_integer_array_assumed_shape(f_array_assumed_shape):
    @functools.cached_property
    def dtype(self):
        return np.int32

    @property
    def ftype(self):
        return fTypeEnum.INTEGER

    @functools.cached_property
    def elem(self):
        return ctypes.c_int32


class f_integer_1_array_assumed_shape(f_integer_array_assumed_shape):
    @functools.cached_property
    def elem(self):
        return ctypes.c_int8

    @functools.cached_property
    def dtype(self):
        return np.int8


class f_integer_2_array_assumed_shape(f_integer_array_assumed_shape):
    @functools.cached_property
    def elem(self):
        return ctypes.c_int16

    @functools.cached_property
    def dtype(self):
        return np.int16


f_integer_4_array_assumed_shape = f_integer_array_assumed_shape


class f_integer_8_array_assumed_shape(f_integer_array_assumed_shape):
    @functools.cached_property
    def elem(self):
        return ctypes.c_int64

    @functools.cached_property
    def dtype(self):
        return np.int64


#######################
# Assumed shape floats
#######################


class f_real_array_assumed_shape(f_array_assumed_shape):
    @functools.cached_property
    def dtype(self):
        return np.float32

    @property
    def ftype(self):
        return fTypeEnum.REAL

    @functools.cached_property
    def elem(self):
        return ctypes.c_float


f_real_4_array_assumed_shape = f_real_array_assumed_shape


class f_real_8_array_assumed_shape(f_real_array_assumed_shape):
    @functools.cached_property
    def elem(self):
        return ctypes.c_double

    @functools.cached_property
    def dtype(self):
        return np.float64


#######################
# Assumed shape logicals
#######################


class f_logical_array_assumed_shape(f_array_assumed_shape):
    @functools.cached_property
    def dtype(self):
        return np.int32

    @property
    def ftype(self):
        return fTypeEnum.LOGICAL

    @functools.cached_property
    def elem(self):
        return ctypes.c_int32

    def from_ctype(self, cvalue):
        array = super().from_param(cvalue)
        return array.astype(bool)


f_logical_4_array_assumed_shape = f_logical_array_assumed_shape


#######################
# Assumed shape character
#######################


class f_character_array_assumed_shape(f_array_assumed_shape):
    def __init__(self, shape, len, ndim=None, *args, **kwargs):
        super().__init__(shape=shape, ndim=ndim)
        self.len = len

    @property
    def dtype(self):
        return f"S{self.len}"

    @property
    def ftype(self):
        return fTypeEnum.CHARACTER

    @property
    def elem(self):
        return ctypes.c_char * self.len


#######################
# Derived types
#######################


_all_dts = {}


class f_derived_type(fObject):
    def __init__(self, dt, *args, **kwargs):
        super().__init__()
        self.dt = dt
        self.allobjs = kwargs["allobjs"]

        self._fields = {}
        _ = self.ctype

    @property
    def ctype(self):
        if self.dt.name in _all_dts:
            return _all_dts[self.dt.name]

        fields = []
        for var in self.dt.dt_components():
            self._fields[var.name] = lookup(self.allobjs, var)

            if var.is_derived():
                print("*", var.dt_type(), self.allobjs[var.dt_type()].name)
                if self.allobjs[var.dt_type()].name in _all_dts:
                    ct = _all_dts[self.allobjs[var.dt_type()].name]
                else:
                    ct = f_derived_type(
                        self.allobjs[var.dt_type()], allobjs=self.allobjs
                    ).ctype
            else:
                ct = self._fields[var.name].ctype

            fields.append((var.name, ct))

        class _fDerivedType(ctypes.Structure):
            _fields_ = fields

        _all_dts[self.dt.name] = _fDerivedType

        return _fDerivedType

    def from_ctype(self, cvalue):
        return fDT(self.dt, self._fields, cvalue, self.allobjs)

    def from_param(self, value):
        ctype = self.ctype()
        return self.set_ctype(ctype, value)

    def set_ctype(self, ctype, value):
        for key, v in value.items():
            if key not in self._fields:
                raise KeyError(f"Derived type does not have element {key}")

            cvalue = self._fields[key].from_param(v)

            setattr(ctype, key, cvalue)

    def from_maybe_ctype(self, cvalue):
        return self.from_ctype(cvalue)


class fDT:
    def __init__(self, dt, fields, ctype, allobjs):
        self._dt = dt
        self._fields = fields
        self._ctype = ctype
        self._allobjs = allobjs

    def keys(self):
        return [i.name for i in self._dt.dt_components()]

    def __contains__(self, key):
        return key in self.keys()

    def values(self):
        return [self.__getitem__(key) for key in self.keys()]

    def items(self):
        return [(key, self.__getitem__(key)) for key in self.keys()]

    def __getitem__(self, key):
        if key not in self:
            raise KeyError(f"Derived type does not have element {key}")

        cvalue = getattr(self._ctype, key)

        return self._fields[key].from_maybe_ctype(cvalue)

    def __setitem__(self, key, value):
        if key not in self:
            raise KeyError(f"Derived type does not have element {key}")

        print(self._fields)
        cvalue = self._fields[key].from_param(value)
        setattr(self._ctype, key, cvalue)


class f_derived_type_explicit_array(f_derived_type):
    def __init__(self, dt, shape, *args, **kwargs):
        # These must before super().__init__ as that calls self.ctype()
        # and then calls the local ctype which needs self.size
        self.shape = shape
        self.ndim = len(shape)
        self.size = np.product(self.shape)

        super().__init__(dt, *args, **kwargs)

    @property
    def ctype(self):
        return super().ctype * self.size

    def from_ctype(self, cvalue):
        return fDT_array(self.dt, self._fields, cvalue, self.shape, self.allobjs)


class f_derived_type_assumed_shape(f_derived_type):
    pass


class fDT_array:
    def __init__(self, dt, fields, ctype, shape, allobjs):
        self._dt = dt
        self._fields = fields
        self._ctype = ctype
        self._allobjs = allobjs
        self.shape = shape
        self.size = np.product(self.shape)
        self.ndim = len(self.shape)

    def __getitem__(self, index):
        if isinstance(index, tuple):
            ind = np.ravel_multi_index(index, self.shape, order="F")
        else:
            ind = index

        if ind > self.size:
            raise IndexError("Out of bounds")

        return fDT(self._dt, self._fields, self._ctype[ind], self._allobjs)

    def __setitem__(self, index):
        if isinstance(index, tuple):
            ind = np.ravel_multi_index(index, self.shape, order="F")
        else:
            ind = index

        if ind > self.size:
            raise IndexError("Out of bounds")

        return fDT(self._dt, self._fields, self._ctype[ind], self._allobjs)


#######################
# Map function
#######################

ftype_mapper = {
    (fTypeEnum.INTEGER, 1, ArrayEnum.SCALAR): f_integer_1,
    (fTypeEnum.INTEGER, 2, ArrayEnum.SCALAR): f_integer_2,
    (fTypeEnum.INTEGER, 4, ArrayEnum.SCALAR): f_integer_4,
    (fTypeEnum.INTEGER, 8, ArrayEnum.SCALAR): f_integer_8,
    (fTypeEnum.REAL, 4, ArrayEnum.SCALAR): f_real_4,
    (fTypeEnum.REAL, 8, ArrayEnum.SCALAR): f_real_8,
    (fTypeEnum.REAL, 16, ArrayEnum.SCALAR): f_real_16,
    (fTypeEnum.COMPLEX, 4, ArrayEnum.SCALAR): f_complex_4,
    (fTypeEnum.COMPLEX, 8, ArrayEnum.SCALAR): f_complex_8,
    (fTypeEnum.COMPLEX, 16, ArrayEnum.SCALAR): f_complex_16,
    (fTypeEnum.LOGICAL, -1, ArrayEnum.SCALAR): f_logical,
    (fTypeEnum.CHARACTER, 1, ArrayEnum.SCALAR): f_character,
    # (fTypeEnum.CHARACTER,4,ArrayEnum.SCALAR): f_character, #TODO
    # Explicit arrays
    (fTypeEnum.INTEGER, 1, ArrayEnum.EXPLICIT): f_integer_1_array_explicit,
    (fTypeEnum.INTEGER, 2, ArrayEnum.EXPLICIT): f_integer_2_array_explicit,
    (fTypeEnum.INTEGER, 4, ArrayEnum.EXPLICIT): f_integer_4_array_explicit,
    (fTypeEnum.INTEGER, 8, ArrayEnum.EXPLICIT): f_integer_8_array_explicit,
    (fTypeEnum.REAL, 4, ArrayEnum.EXPLICIT): f_real_4_array_explicit,
    (fTypeEnum.REAL, 8, ArrayEnum.EXPLICIT): f_real_8_array_explicit,
    (fTypeEnum.COMPLEX, 4, ArrayEnum.EXPLICIT): f_complex_4_array_explicit,
    (fTypeEnum.COMPLEX, 8, ArrayEnum.EXPLICIT): f_complex_8_array_explicit,
    (fTypeEnum.LOGICAL, -1, ArrayEnum.EXPLICIT): f_logical_array_explicit,
    (fTypeEnum.CHARACTER, 1, ArrayEnum.EXPLICIT): f_character_array_explicit,
    # Assumed shape
    (fTypeEnum.INTEGER, 1, ArrayEnum.ASSUMED_SHAPE): f_integer_1_array_assumed_shape,
    (fTypeEnum.INTEGER, 2, ArrayEnum.ASSUMED_SHAPE): f_integer_2_array_assumed_shape,
    (fTypeEnum.INTEGER, 4, ArrayEnum.ASSUMED_SHAPE): f_integer_4_array_assumed_shape,
    (fTypeEnum.INTEGER, 8, ArrayEnum.ASSUMED_SHAPE): f_integer_8_array_assumed_shape,
    (fTypeEnum.REAL, 4, ArrayEnum.ASSUMED_SHAPE): f_real_4_array_assumed_shape,
    (fTypeEnum.REAL, 8, ArrayEnum.ASSUMED_SHAPE): f_real_8_array_assumed_shape,
    # (fTypeEnum.COMPLEX,4,ArrayEnum.ASSUMED_SHAPE): f_complex_4_array_assumed_shape, #TODO
    # (fTypeEnum.COMPLEX,8,ArrayEnum.ASSUMED_SHAPE): f_complex_8_array_assumed_shape, #TODO
    (fTypeEnum.LOGICAL, -1, ArrayEnum.ASSUMED_SHAPE): f_logical_array_assumed_shape,
    (fTypeEnum.CHARACTER, 1, ArrayEnum.ASSUMED_SHAPE): f_character_array_assumed_shape,
    # Derived types
    (fTypeEnum.DERIVED, -1, ArrayEnum.SCALAR): f_derived_type,
    (fTypeEnum.DERIVED, -1, ArrayEnum.EXPLICIT): f_derived_type_explicit_array,
    (fTypeEnum.DERIVED, -1, ArrayEnum.ASSUMED_SHAPE): f_derived_type_assumed_shape,
}


def lookup(allobjs, obj):
    tka = obj.type_kind_array()

    args = {"allobjs": allobjs}

    if obj.is_array():
        args["shape"] = obj.shape
        args["ndim"] = obj.ndim

    if obj.is_char():
        args["len"] = obj.strlen.value

    if obj.is_derived():
        args["dt"] = allobjs[obj.dt_type()]

    ftype = ftype_mapper[tka](**args)

    return ftype
