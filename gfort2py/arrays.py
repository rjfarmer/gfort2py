from __future__ import print_function
import ctypes
from .var import fVar, fParam
from .types import fDerivedType
import numpy as np
from .utils import *
from .fnumpy import *

class fExplicitArray(fVar):

    def __init__(self, lib, obj, TEST_FLAG=False):
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

    def __init__(self, lib, obj, TEST_FLAG=False):
        self.__dict__.update(obj)
        self._lib = lib

        self.ndim = self.array['ndims']
        self._lib = lib
        
        self._desc = self._setup_desc()
        self._ctype = getattr(ctypes,self.ctype)
        self._ctype_desc = ctypes.POINTER(self._desc)
        self.npdtype=self.pytype+str(8*ctypes.sizeof(self._ctype))
        

    def _setup_desc(self):
        class bounds(ctypes.Structure):
            _fields_=[("stride",self._index_t),
                      ("lbound",self._index_t),
                      ("ubound",self._index_t)]
        
        class fAllocArray(ctypes.Structure):
            _fields_=[('base_addr',ctypes.c_void_p), 
                      ('offset',self._size_t), 
                      ('dtype',self._index_t),
                      ('dims',bounds*self.ndim)
                      ]
                      
        return fAllocArray

    def _get_pointer(self):
        return self._ctype_desc.from_address(ctypes.addressof(getattr(self._lib,self.mangled_name)))


    def set_mod(self, value):
        """
        Set a module level variable
        """
        self._value = value.astype(self.npdtype)
        
        #Did we make a copy?
        if self._id(self._value)==self._id(value):
            remove_ownership(value)
        remove_ownership(self._value)
            
        p = self._get_pointer()
        self._set_to_pointer(self._value,p.contents)
        
        return 
        
    def _set_to_pointer(self,value,p):
    
        if value.ndim > self._GFC_MAX_DIMENSIONS:
            raise ValueError("Array too big")
        
        if not self.ndim == value.ndim:
            raise ValueError("Array size mismatch")
        
        p.base_addr = value.ctypes.get_data()
        p.offset = self._size_t(0)
        
        p.dtype = self._get_dtype()
        
        for i in range(self.ndim):
            p.dims[i].stride = self._index_t(value.strides[i]//ctypes.sizeof(self._ctype))
            p.dims[i].lbound = self._index_t(1)
            p.dims[i].ubound = self._index_t(value.shape[i])
            
        return

    def get(self,copy=False):
        """
        Get a module level variable
        """
        p = self._get_pointer()
        value = self._get_from_pointer(p.contents,copy)
        return value
        
    def _get_from_pointer(self,p,copy=False):
        base_addr = p.base_addr
        if not self._isallocated():
            raise ValueError("Array not allocated yet")
        
        offset = p.offset
        dtype = p.dtype
        
        dims=[]
        shape=[]
        for i in range(self.ndim):
            dims.append({})
            dims[i]['stride'] = p.dims[i].stride
            dims[i]['lbound'] = p.dims[i].lbound
            dims[i]['ubound'] = p.dims[i].ubound
            
        for i in range(self.ndim):
            shape.append(dims[i]['ubound']-dims[i]['lbound']+1)
            
        self._shape=tuple(shape)
        size=np.product(shape)
        
        if copy:
            # When we want a copy of the array not a pointer to the fortran memoray
            res = self._get_var_from_address(base_addr,size=size)
            res = np.asfortranarray(res)
            res = res.reshape(shape).astype(self.npdtype)
        else:
            # When we want to pointer to the underlaying fortran memoray
            # will leak as we dont have a deallocate call to call in a del func
            ptr = ctypes.cast(base_addr,ctypes.POINTER(self._ctype))
            res = np.ctypeslib.as_array(ptr,shape= self._shape)
        
        return res
        

    def py_to_ctype(self, value):
        """
        Pass in a python value returns the ctype representation of it
        """
        return self._set_to_pointer(value,self._ctype)
        
    def py_to_ctype_f(self, value):
        """
        Pass in a python value returns the ctype representation of it, 
        suitable for a function
        
        Second return value is anything that needs to go at the end of the
        arg list, like a string len
        """
        return self._set_to_pointer(value,self._ctype),None

    def ctype_to_py(self, value):
        """
        Pass in a ctype value returns the python representation of it
        """
        return self._get_from_pointer(value.contents)
        
    def ctype_to_py_f(self, value):
        """
        Pass in a ctype value returns the python representation of it,
        as returned by a function (may be a pointer)
        """
        return self._get_from_pointer(value.contents)

    def pytype_def(self):
        return np.array

    def ctype_def(self):
        """
        The ctype type of this object
        """
        return self._ctype_desc

    def ctype_def_func(self):
        """
        The ctype type of a value suitable for use as an argument of a function

        May just call ctype_def
        
        Second return value is anythng that needs to go at the end of the
        arg list, like a string len
        """

        return self.ctype_def(),None

    def _get_dtype(self):
        ftype=self._get_ftype()
        d=self.ndim
        d=d|(ftype<<self._GFC_DTYPE_TYPE_SHIFT)
        d=d|(ctypes.sizeof(self._ctype)<<self._GFC_DTYPE_SIZE_SHIFT)
        return d

    def _get_ftype(self):
        ftype=None
        dtype=self.ctype
        if 'c_int' in dtype:
            ftype=self._BT_INTEGER
        elif 'c_double' in dtype or 'c_real' in dtype:
            ftype=self._BT_REAL
        elif 'c_bool' in dtype:
            ftype=self._BT_LOGICAL
        elif 'c_char' in dtype:
            ftype=self._BT_CHARACTER
        else:
            raise ValueError("Cant match dtype, got "+dtype)
        return ftype

    def __str__(self):
        return str(self.get())
        
    def __repr__(self):
        return repr(self.get())

    def __getattr__(self, name): 
        if name in self.__dict__:
            return self.__dict__[name]

        return getattr(self.get(),name)
        
    def __del__(self):
        if '_value' in self.__dict__:
            #Problem occurs as both fortran and numpy are pointing to same memory address
            #Thus if fortran deallocates the array numpy will try to free the pointer
            #when del is called casuing a double free error
            
            #By calling remove_ownership we tell numpy it dosn't own the data
            #thus is shouldn't call free(ptr).
            remove_ownership(self._value)
            
            #Maybe leaks if fortran doesn't dealloc the array
                
                
    def _isallocated(self):
        p = self._get_pointer()
        if p.contents.base_addr:
            #Base addr is NULL if deallocated
            return True
        else:
            return False
        
    def _id(self,x):
        return x.ctypes.data
        
class fParamArray(fParam):

    def get(self):
        """
        A parameters value is stored in the dict, as we cant access them
        from the shared lib.
        """
        return np.array(self.value, dtype=self.pytype)


