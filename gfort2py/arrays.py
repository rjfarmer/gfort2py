from __future__ import print_function
import ctypes
from .var import fVar, fParam
import numpy as np


class fExplicitArray(fVar):

    def __init__(self, lib, obj,TEST_FLAG=False):
        self.__dict__.update(obj)
        self._lib = lib
        self._pytype = np.array
        self._ctype = self.ctype_def()
        #self._ctype_f = self.ctype_def_func()
        self.TEST_FLAG=TEST_FLAG
        self._dtype=self.pytype+str(8*ctypes.sizeof(self._ctype))

    def ctype_to_py(self, value):
        """
        Pass in a ctype value returns the python representation of it
        """
        return self._get_var_by_iter(value, self._array_size())
        
        
        
    def py_to_ctype_f(self, value):
        """
        Pass in a python value returns the ctype representation of it, 
        suitable for a function
        
        Second return value is anything that needs to go at the end of the
        arg list, like a string len
        """
        self._data = np.asfortranarray(value.T.astype(self._dtype))

        return self._data,None
        
    def ctype_to_py_f(self, value):
        """
        Pass in a ctype value returns the python representation of it,
        as returned by a function (may be a pointer)
        """
        return np.asfortranarray(value,dtype=self._dtype)

    def pytype_def(self):
        return self._pytype

    def ctype_def(self):
        """
        The ctype type of this object
        """
        return getattr(ctypes, self.ctype)

    def ctype_def_func(self):
        """
        The ctype type of a value suitable for use as an argument of a function

        May just call ctype_def
        
        Second return value is anythng that needs to go at the end of the
        arg list, like a string len
        """
        x=np.ctypeslib.ndpointer(dtype=self._dtype,ndim=self.array['ndims'],
                                flags='F_CONTIGUOUS')
        y=None
        return x,y        
        
        

    def set_mod(self, value):
        """
        Set a module level variable
        """
        r = self._get_from_lib()
        v = value.flatten(order='C')
        self._set_var_from_iter(r, v, self._array_size())
        
    def get(self):
        """
        Get a module level variable
        """
        r = self._get_from_lib()
        s = self.ctype_to_py(r)
        shape = self._make_array_shape()
        return np.reshape(s, shape)

    def _make_array_shape(self,bounds=None):
        if bounds is None:
            bounds = self.array['bounds']
        
        shape = []
        for i, j in zip(bounds[0::2], bounds[1::2]):
            shape.append(j - i + 1)
        return shape

    def _array_size(self,bounds=None):
        return np.product(self._make_array_shape(bounds))


class fDummyArray(fVar):
    _GFC_MAX_DIMENSIONS = 7

    _GFC_DTYPE_RANK_MASK = 0x07
    _GFC_DTYPE_TYPE_SHIFT = 3
    _GFC_DTYPE_TYPE_MASK = 0x38
    _GFC_DTYPE_SIZE_SHIFT = 6

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

    _index_t = ctypes.c_int64
    _size_t = ctypes.c_int64

    def __init__(self, lib, obj):
        self.__dict__.update(obj)
        self._lib = lib

        self.ndim = self.array['ndims']
        self._make_array_desc()

    def _make_array_desc(self):

        self._ctype = fDerivedTypeDesc()
        self._ctype.set_fields(['base_addr', 'offset', 'dtype'],
                               [ctypes.c_void_p, self._size_t, self._index_t])

        self._dims = fDerivedTypeDesc()
        self._dims.set_fields = (['stride', 'lbound', 'ubound'],
                                 [self._index_t, self._index_t, self._index_t])

        self._ctype.add_arg(['dims', self._dims * self.ndim])

    def _set_array(self, value):
        r = self._get_from_lib()

    def _get_array(self):
        r = self._get_from_lib()


class fParamArray(fParam):

    def get(self):
        """
        A parameters value is stored in the dict, as we cant access them
        from the shared lib.
        """
        return np.array(self.value, dtype=self.pytype)
