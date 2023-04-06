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
            if list(value.shape) != shape:
                raise ValueError(f"Wrong shape, got {value.shape} expected {shape}")

        value = value.ravel(order="F")
        return value

    def ctype_len(self):
        return None

class fScalar(fVar_t):
    def ctype(self):
        return self._ctype_base

    def from_param(self, param):
        self._cvalue.value = param
        return self._cvalue

    def from_address(self, addr):
        self._cvalue = self.ctype().from_address(addr)
        return self._cvalue

    def in_dll(self, lib):
        self._cvalue =  self.ctype().in_dll(lib, self.mangled_name)
        return  self._cvalue

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
        self._cvalue.real = param.real
        self._cvalue.imag = param.imag
        return self._cvalue

    def from_address(self, addr):
        self._cvalue = self.ctype().from_address(addr)
        return self._cvalue

    def in_dll(self, lib):
        self._cvalue =  self.ctype().in_dll(lib, self.mangled_name)
        return  self._cvalue

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
        return self._ctype_base * np.prod(self.obj.shape())

    def from_param(self, value):
        self._value = self._array_check(value)
        _copy_array(
            self._value.ctypes.data,
            ctypes.addressof(self._cvalue),
            ctypes.sizeof(self._ctype_base),
            self.obj.size,
        )
        return self._cvalue

    def from_address(self, addr):
        self._cvalue = self.ctype().from_address(addr)
        return self._cvalue

    def in_dll(self, lib):
        self._cvalue =  self.ctype().in_dll(lib, self.mangled_name)
        return  self._cvalue

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
    def ctype(self):
        return _make_fAlloc15(self.ndims)

    def from_param(self, value):
        self._value = self._array_check(value, False)
        self._cvalue = self.ctype()

        _copy_array(
            self._value.ctypes.data, self._cvalue.base_addr, self.sizeof, np.size(value)
        )
        return self._cvalue


    def from_address(self, addr):
        self._cvalue = self.ctype().from_address(addr)
        return self._cvalue

    def in_dll(self, lib):
        self._cvalue =  self.ctype().in_dll(lib, self.mangled_name)
        return  self._cvalue

    @property
    def value(self):
        if self._cvalue.base_addr is None:
            return None

        shape = []
        for i in range(self.obj.ndim):
            shape.append(self._cvalue.dims[i].ubound - self._cvalue.dims[i].lbound + 1)

        shape = tuple(shape)
        size = (np.prod(shape),)

        PTR = ctypes.POINTER(self.ctype_base())
        x_ptr = ctypes.cast(self._cvalue.base_addr, PTR)

        return np.ctypeslib.as_array(x_ptr,shape=size).reshape(shape,order='F')


    @value.setter
    def value(self, value):
        self.from_param(value)

    def __doc__(self):
        return f"{self.type}(KIND={self.kind})(:) :: {self.name}"

class fAssumedSize(fVar_t):
    def ctype(self):
        return self._ctype_base() * np.prod(self._value.shape())

    def from_param(self, value):
        self._value = self._array_check(value)
        self._cvalue = self.ctype()
        _copy_array(
            self._value.ctypes.data,
            ctypes.addressof(self._cvalue),
            self.sizeof,
            np.size(value),
        )
        return self._cvalue

    def from_address(self, addr):
        self._cvalue = self.ctype().from_address(addr)
        return self._cvalue

    def in_dll(self, lib):
        self._cvalue =  self.ctype().in_dll(lib, self.mangled_name)
        return  self._cvalue

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
    def ctype(self):
        return self._ctype_base * self.len()

    def from_param(self, value):
        self._value = value

        if hasattr(self._value, "encode"):
            self._value = self._value.encode()

        if len(self._value) > self.len():
            self._value = self._value[:self.len()]
        else:
            self._value = self._value + b" " * (self.len() - len(self._value))

        self._buf = bytearray(self._value)  # Need to keep hold of the reference
        for idx,i in enumerate(self._buf):
            self._cvalue[idx] = i

        return self._cvalue

    def from_address(self, addr):
        self._cvalue = self.ctype().from_address(addr)
        return self._cvalue

    def in_dll(self, lib):
        self._cvalue =  self.ctype().in_dll(lib, self.mangled_name)
        return self._cvalue

    @property
    def value(self):
        return "".join([i.decode() for i in self._cvalue])

    @value.setter
    def value(self, value):
        self.from_param(value)

    def len(self):
        if self.obj.is_deferred_len():
            return len(self._value)
        else:
            return self.obj.strlen.value

    def ctype_len(self):
        return ctypes.c_int64(self.len())

    def __doc__(self):
        try:
            strlen = (
                self.obj.sym.ts.charlen.value
            )  # We know the string length at compile time
            return f"{self.type}(LEN={strlen}) :: {self.name}"
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


def ptr_unpack(ptr):
    if hasattr(ptr, "contents"):
        if hasattr(ptr.contents, "contents"):
            x = ptr.contents.contents
        else:
            x = ptr.contents
    return x

def ctype_map(type, kind):
    if type == "INTEGER":
        if kind == 4:
            return ctypes.c_int32
        elif kind == 8:
            return ctypes.c_int64
        else:
            raise TypeError("Integer type of kind {kind} not supported")
    elif type == "REAL":
        if kind == 4:
            return ctypes.c_float
        elif kind == 8:
            return ctypes.c_double
        elif kind == 16:
            # Although we dont support quad yet we can keep things aligned
            return ctypes.c_ubyte * 16
        else:
            raise TypeError("Float type of kind {kind} not supported")
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
            raise TypeError("Complex type of kind {kind} not supported")

        class complex(ctypes.Structure):
                _fields_ = [
                    ("real", ct),
                    ("imag", ct),
                ]
        return complex
    else:
        raise TypeError(f"Type {type} and kind {kind} not supported")



# class fVar_t(ABC):
#     def __init__(self, obj):
#         self.obj = obj

#         self.type, self.kind = self.obj.type_kind()

#         self._ctype_base = ctype_map(self.type, self.kind)

#     def name(self):
#         return self.obj.name

#     def mangled_name(self):
#         return self.obj.mangled_name

#     def module(self):
#         return self.obj.module

#     def _array_check(self, value, know_shape=True):
#         value = value.astype(self.obj.dtype())
#         shape = self.obj.shape()
#         ndim = self.obj.ndim

#         if not value.flags["F_CONTIGUOUS"]:
#             value = np.asfortranarray(value)

#         if value.ndim != ndim:
#             raise ValueError(
#                 f"Wrong number of dimensions, got {value.ndim} expected {ndim}"
#             )

#         if know_shape:
#             if list(value.shape) != shape:
#                 raise ValueError(f"Wrong shape, got {value.shape} expected {shape}")

#         value = value.ravel(order="F")
#         return value

#     @abstractmethod
#     def from_param(self, value, ctype=None):

#         if self.obj.is_optional() and value is None:
#             return None

#         if self.obj.is_array():
#             if self.obj.is_explicit():
#                 value = self._array_check(value)
#                 if ctype is None:
#                     ctype = self.ctype(value)()
#                 self.copy_array(
#                     value.ctypes.data,
#                     ctypes.addressof(ctype),
#                     self.sizeof,
#                     self.obj.size,
#                 )
#                 return ctype
#             elif self.obj.is_assumed_size():
#                 value = self._array_check(value, know_shape=False)
#                 if ctype is None:
#                     ctype = self.ctype(value)()

#                 self.copy_array(
#                     value.ctypes.data,
#                     ctypes.addressof(ctype),
#                     self.sizeof,
#                     np.size(value),
#                 )

#                 return ctype

#             elif self.obj.needs_array_desc():
#                 shape = self.obj.shape
#                 ndim = self.obj.ndim

#                 if ctype is None:
#                     ctype = _make_fAlloc15(ndim)()

#                 if value is None:
#                     return ctype
#                 else:
#                     shape = value.shape
#                     value = self._array_check(value, False)

#                     self.copy_array(
#                         value.ctypes.data, ctype.base_addr, self.sizeof, np.size(value)
#                     )
#                     return ctype

#         if ctype is None:
#             ctype = self.ctype

#         if self.type == "INTEGER":
#             return ctype(value)
#         elif self.type == "REAL":
#             if self.kind == 16:
#                 print(
#                     f"Object of type {self.type} and kind {self.kind} not supported yet, passing None"
#                 )
#                 return ctype(None)

#             return ctype(value)
#         elif self.type == "LOGICAL":
#             if value:
#                 return ctype(1)
#             else:
#                 return ctype(0)
#         elif self.type == "CHARACTER":
#             strlen = self.len(value).value

#             if hasattr(value, "encode"):
#                 value = value.encode()

#             if len(value) > strlen:
#                 value = value[:strlen]
#             else:
#                 value = value + b" " * (strlen - len(value))

#             self._buf = bytearray(value)  # Need to keep hold of the reference

#             return ctype.from_buffer(self._buf)
#         elif self.type == "COMPLEX":
#             return ctype(value.real, value.imag)

#         raise NotImplementedError(
#             f"Object of type {self.type} and kind {self.kind} not supported yet"
#         )

#     def len(self, value=None):
#         if self.obj.is_char():
#             if self.obj.is_deferred_len():
#                 l = len(value)
#             else:
#                 l = self.obj.strlen.value

#         elif self.obj.is_array():
#             if self.obj.is_assumed_size():
#                 l = np.size(value)
#         else:
#             l = None

#         return ctypes.c_int64(l)

#     @abstractmethod
#     def ctype(self):
#         cb_arr = None

#         if self.type == "CHARACTER":
#             try:
#                 strlen = (
#                     self.obj.sym.ts.charlen.value
#                 )  # We know the string length at compile time
#                 self._ctype_base = ctypes.c_char * strlen                
#             except AttributeError:
#                 # We de not know the string length at compile time
#                 pass
            
#         if self.obj.is_array():
#             if self.obj.is_explicit():

#                 def callback(*args):
#                     return self._ctype_base() * np.prod(self.obj.shape())

#                 cb_arr = callback
#             elif self.obj.is_assumed_size():

#                 def callback(value, *args):
#                     return self._ctype_base() * np.size(value)

#                 cb_arr = callback

#             elif self.obj.needs_array_desc():

#                 def callback(*args):
#                     return _make_fAlloc15(self.obj.ndim)

#                 cb_arr = callback

#         else:
#             def callback(*args):
#                 return self._ctype_base
#             cb_arr = callback

#         if cb_arr is None:
#             raise NotImplementedError(
#                 f"Object of type {self.type} and kind {self.kind} not supported yet"
#             )
#         else:
#             return cb_arr

#     def from_ctype(self, value):
#         if value is None:
#             return None

#         x = value

#         if hasattr(value, "contents"):
#             if hasattr(value.contents, "contents"):
#                 x = value.contents.contents
#             else:
#                 x = value.contents

#         if self.obj.is_array():
#             if self.obj.is_explicit() or self.obj.is_assumed_size():
#                 # If x is a 1d array of prod(shape) then force a reshape
#                 return np.ctypeslib.as_array(x,shape=np.prod(self.obj.shape())).reshape(self.obj.shape(),order='F')
#             elif self.obj.needs_array_desc():
#                 if x.base_addr is None:
#                     return None

#                 shape = []
#                 for i in range(self.obj.ndim):
#                     shape.append(x.dims[i].ubound - x.dims[i].lbound + 1)

#                 shape = tuple(shape)
#                 size = (np.prod(shape),)

#                 PTR = ctypes.POINTER(self.ctype_map())
#                 x_ptr = ctypes.cast(x.base_addr, PTR)

#                 return np.ctypeslib.as_array(x_ptr,shape=size).reshape(shape,order='F')


#         if self.type == "COMPLEX":
#             return complex(x.real, x.imag)

#         if hasattr(x, "value") and not self.type == "CHARACTER":
#             x = x.value

#         if self.type == "INTEGER":
#             return int(x)
#         elif self.type == "REAL":
#             if self.kind == 16:
#                 raise NotImplementedError(
#                     f"Object of type {self.type} and kind {self.kind} not supported yet"
#                 )
#             return float(x)
#         elif self.type == "LOGICAL":
#             return x == 1
#         elif self.type == "CHARACTER":
#             return "".join([i.decode() for i in x])
#         else:
#             raise NotImplementedError(
#                 f"Object of type {self.type} and kind {self.kind} not supported yet"
#             )

#     # @property
#     # def __doc__(self):
#     #     return f"{self.obj.head.name}={self.typekind}"

#     # @property
#     # def typekind(self):
#     #     if self.type == "INTEGER" or self.type == "REAL":
#     #         return f"{self.type}(KIND={self.kind})"
#     #     elif self.type == "LOGICAL":
#     #         return f"{self.type}"
#     #     elif self.type == "CHARACTER":
#     #         try:
#     #             strlen = (
#     #                 self.obj.sym.ts.charlen.value
#     #             )  # We know the string length at compile time
#     #             return f"{self.type}(LEN={strlen})"
#     #         except AttributeError:
#     #             return f"{self.type}(LEN=:)"

#     @property
#     def sizeof(self):
#         return self.kind

#     def set_ctype(self, ctype, value):
#         if self.obj.is_array():
#             v = self.from_param(value, ctype)
#             return
#         elif isinstance(ctype, ctypes.Structure):
#             for k in ctype.__dir__():
#                 if not k.startswith("_") and hasattr(value, k):
#                     setattr(ctype, k, getattr(value, k))
#         else:
#             ctype.value = self.from_param(value).value
#             return

#     def copy_array(self, src, dst, length, size):
#         ctypes.memmove(
#             dst,
#             src,
#             length * size,
#         )

#     @abstractmethod
#     def in_dll(self):
#         pass

#     @abstractmethod
#     def value(self):
#         pass