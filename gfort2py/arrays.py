# SPDX-License-Identifier: GPL-2.0+
from __future__ import print_function
import ctypes
import sys
import numpy as np
import numbers

try:
    import __builtin__
except ImportError:
    import builtins as __builtin__

from .var import fVar, fParam
from .utils import *
from .fnumpy import *
from .errors import *

_index_t = ctypes.c_int64
_size_t = ctypes.c_int64
_mod_version = 14
_GFC_MAX_DIMENSIONS = -1

class _bounds14(ctypes.Structure):
    _fields_=[("stride",_index_t),
              ("lbound",_index_t),
              ("ubound",_index_t)]
              
class _dtype_type(ctypes.Structure):
    _fields_=[("elem_len",_size_t),
                ('version', ctypes.c_int),
                ('rank',ctypes.c_byte),
                ('type',ctypes.c_byte),
                ('attribute',ctypes.c_ushort)]
              
def _make_fAlloc15(ndims):
    class _fAllocArray(ctypes.Structure):
        _fields_=[('base_addr',ctypes.c_void_p), 
                ('offset',_size_t), 
                ('dtype',_dtype_type),
                ('span',_index_t),
                ('dims',_bounds14*ndims)
                ]
    return _fAllocArray
    

# gfortran 8 needs https://gcc.gnu.org/wiki/ArrayDescriptorUpdate
#https://docs.scipy.org/doc/numpy/reference/generated/numpy.lib.mixins.NDArrayOperatorsMixin.html#numpy.lib.mixins.NDArrayOperatorsMixin

#From gcc source code
#Parsed       Lower   Upper  Returned
#------------------------------------
#:           NULL    NULL   AS_DEFERRED (*)
#x            1       x     AS_EXPLICIT
#x:           x      NULL   AS_ASSUMED_SHAPE
#x:y          x       y     AS_EXPLICIT
#x:*          x      NULL   AS_ASSUMED_SIZE
#*            1      NULL   AS_ASSUMED_SIZE


if sys.byteorder is 'little':
    _byte_order=">"
else:
    _byte_order="<"
    
class BadFortranArray(Exception):
    pass
    

class fExplicitArray(fParentArray, np.lib.mixins.NDArrayOperatorsMixin):            
    def __init__(self, lib, obj):
        self.__dict__.update(obj)
        self._lib = lib
        self._array = True
        
        if 'array' in self.var:
          self.__dict__.update(obj['var'])
        
        self.ctype=self.var['ctype']
        self.pytype=self.var['pytype']

        if self.pytype == 'quad':
            self.pytype = np.longdouble
        elif self.pytype=='bool':
            self.pytype=int
            self.ctype='c_int32'
        else:
             self.pytype = getattr(__builtin__, self.pytype)

        self.ctype = getattr(ctypes, self.ctype)
        
        if self.pytype == int:
            self._dtype='int'+str(8*ctypes.sizeof(self.ctype))
        elif self.pytype == float:
            self._dtype='float'+str(8*ctypes.sizeof(self.ctype))
        else:
            raise NotImplementedError("Type not supported ",self.pytype)
        
        self._ndims = int(self.array['ndim'])
        
        size = self.size()
        if size > 0:
            self.ctype = self.ctype * self.size()
            
        self._shape = self.shape()

    def in_dll(self):
        if 'mangled_name' in self.__dict__ and '_lib' in self.__dict__:
            try:
                return self.ctype.in_dll(self._lib, self.mangled_name)
            except ValueError:
                raise NotInLib
        raise NotInLib 
        
    def from_address(self, addr):
        buff = {
                'data': (addr, True),
                'typestr': self._dtype,
                'shape': self._shape
                }

        class numpy_holder():
            pass

        holder = numpy_holder()
        holder.__array_interface__ = buff
        
        return np.asfortranarray(holder)
        
        
    def sizeof(self):
        return ctypes.sizeof(self.ctype)

    def shape(self):
        if len(self.array['shape'])/self._ndims != 2:
            return -1
        
        shape = []
        for l,u in zip(self.array['shape'][0::2],self.array['shape'][1::2]):
            shape.append(u-l+1)
        return tuple(shape)
        
    def size(self):
        return np.product(self.shape())

    def set_from_address(self, addr, value):
        ctype = self.from_address(addr)
        self._set(ctype, value)

    def _set(self, c, v):
        if v.ndim != self._ndims:
            raise AttributeError("Bad ndims for array")
        
        if v.shape != self._shape:
            raise AttributeError("Bad shape for array")
            
        v = np.asfortranarray(v.astype(self._dtype)).T
        v_addr = v.ctypes.data

        ctypes.memmove(ctypes.addressof(c), v_addr, self.sizeof())

    def get(self):
        return self.from_address(ctypes.addressof(self.in_dll())).T       

    def set(self, value):
        self._set(self.in_dll(), value)
        
    def from_param(self, value):
        size = np.size(value)
        self._shape = np.shape(value)
        self.ctype = self.ctype * size
        
        self._value = value # Keep hold of a reference to the array
        self._safe_ctype = self.ctype()
        self._set(self._safe_ctype, self._value)
        return self._safe_ctype
        
    def from_func(self, pointer):
        return self.from_address(ctypes.addressof(pointer))
        

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        out = kwargs.get('out', ())
        for x in inputs + out:
            # Only support operations with instances of _HANDLED_TYPES.
            # Use ArrayLike instead of type(self) for isinstance to
            # allow subclasses that don't override __array_ufunc__ to
            # handle ArrayLike objects.
            if not isinstance(x, self._HANDLED_TYPES + (fExplicitArray,)):
                return NotImplemented

        # Defer to the implementation of the ufunc on unwrapped values.
        inputs = tuple(x.value.get() if isinstance(x, fExplicitArray) else x
                       for x in inputs)
        if out:
            kwargs['out'] = tuple(
                x.get() if isinstance(x, fExplicitArray) else x
                for x in out)
        result = getattr(ufunc, method)(*inputs, **kwargs)

        if type(result) is tuple:
            # multiple return values
            return tuple(type(self)(x) for x in result)
        elif method == 'at':
            # no return value
            return None
        else:
            # one return value
            return type(self)(result)

    @property
    def __array_interface__(self):
        return self.get().__array_interface__
     


class fDummyArray(fParentArray):
    _GFC_MAX_DIMENSIONS = -1
    
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
    
    _BT_TYPESPEC = {_BT_UNKNOWN:'v',_BT_INTEGER:'i',_BT_LOGICAL:'b',
                _BT_REAL:'f',_BT_COMPLEX:'c',_BT_DERIVED:'v',
                _BT_CHARACTER:'v',_BT_CLASS:'v',_BT_PROCEDURE:'v',
                _BT_HOLLERITH:'v',_BT_VOID:'v',_BT_ASSUMED:'v'}
                
    _PY_TO_BT = {'int':_BT_INTEGER,'float':_BT_REAL,'bool':_BT_LOGICAL,
            'str':_BT_CHARACTER,'bytes':_BT_CHARACTER}

    def __init__(self, lib, obj,notinlib=False):
        self.__dict__.update(obj)
        self._lib = lib
        self._notinlib = notinlib
        self._array = self.var['array']

        self.ndim = int(self._array['ndim'])
        
        self.ctype_elem=self.var['ctype']
        self.pytype=self.var['pytype']

        if self.pytype == 'quad':
            self.pytype = np.longdouble
        elif self.pytype=='bool':
            self.pytype=int
            self.ctype_elem='c_int32'
        else:
             self.pytype = getattr(__builtin__, self.pytype)
             
        self.ctype_elem = getattr(ctypes, self.ctype_elem)
             
        if self.pytype == int:
            self._dtype='int'+str(8*ctypes.sizeof(self.ctype_elem))
        elif self.pytype == float:
            self._dtype='float'+str(8*ctypes.sizeof(self.ctype_elem))
        else:
            raise NotImplementedError("Type not supported yet ",self.pytype)
        
        self.ctype = ctypes.c_void_p
        
        self._array_desc = _make_fAlloc15(self.ndim)
        
    def _shape_from_bounds(self,bounds):
        shape = []
        for i in range(self.ndim):
            shape.append(bounds[i].ubound-bounds[i].lbound+1)
    
        return tuple(shape)       

    def _get_dtype15(self):
        ftype = self._get_ftype()
        x = _dtype_type()
        x.elem_len = ctypes.sizeof(self.ctype_elem)
        x.version = 0 
        x.rank = self.ndim
        x.type = ftype
        x.attribute = 0 
        
        return x

    def _get_ftype(self):
        ftype = None
        # Recover the orignal version
        ct = self.var['ctype']
        if 'c_int' == ct:
            ftype=self._BT_INTEGER
        elif 'c_double' == ct or 'c_real' == ct or 'c_float' == ct:
            ftype=self._BT_REAL
        elif 'c_bool' == ct:
            ftype=self._BT_LOGICAL
        elif 'c_char' == ct:
            ftype=self._BT_CHARACTER
        else:
            raise ValueError("Cant match dtype, got "+ctype)
        return ftype               
        
    
    def in_dll(self):
        if 'mangled_name' in self.__dict__ and '_lib' in self.__dict__:
            try:
                return self.ctype.in_dll(self._lib, self.mangled_name)
            except ValueError:
                raise NotInLib
        raise NotInLib 
        
    def from_address(self, addr):
        if self.ctype.from_address(addr).value is None:
            raise AllocationError("Array not allocated yet")
        
        
        v = self._array_desc.from_address(addr)
        buff = {
            'data': (v.base_addr,
                    True),
            'typestr': self._dtype,
            'shape': self._shape_from_bounds(v.dims)
            }
        
        class numpy_holder():
            pass
        
        holder = numpy_holder()
        holder.__array_interface__ = buff
        return np.array(holder)  

        
    def sizeof(self):
        return ctypes.sizeof(self.ctype)
            

    def set_from_address(self, addr, value):
        ctype = self._array_desc.from_address(addr)
        self._set(ctype, value)
        
    def _set(self, c, v):
        if v.ndim != self.ndim:
            raise AllocationError("Bad ndim for array")
        
        v_new = np.asfortranarray(v.astype(self._dtype))

        c.base_addr = v_new.ctypes.data
 
        strides = []
        for i in range(self.ndim):
            c.dims[i].lbound = _index_t(1)
            c.dims[i].ubound = _index_t(v_new.shape[i])
            strides.append(c.dims[i].ubound-c.dims[i].lbound+1)

        c.span = ctypes.sizeof(self.ctype_elem)
        c.dtype = self._get_dtype15()
        for i in range(self.ndim):
            c.dims[i].stride = _index_t(int(np.product(strides[:i])))  

        c.offset = -c.dims[-1].stride
 
    def get(self):
        return self.from_address(ctypes.addressof(self.in_dll()))   
        
    def set(self, value):
        self.set_from_address(ctypes.addressof(self.in_dll()), value)
 
    def from_param(self, value):
        self._value = value # Keep hold of a reference to the array
        self._safe_ctype =  self._array_desc()
        self.set_from_address(ctypes.addressof(self._safe_ctype), self._value)
        return self.ctype.from_address(ctypes.addressof(self._safe_ctype))  
           
    def from_func(self, pointer):
        return self.from_address(ctypes.addressof(pointer))          
           
           
           
# class fAssumedShape(fDummyArray):
    # def __init__(self, lib, obj):
        # super(fAssumedShape, self).__init__(lib,obj)
        # self._zero_offset = True
    
    
    # def _get_pointer(self):        
        # x = self._ctype_desc.from_address(ctypes.addressof(self._value_array))
        # return x
        
    # def _get_from_pointer(self,p,copy=False):
        # if not self._isallocated():
            # return np.zeros(1)
            # #raise ValueError("Array not allocated yet")
        # base_addr = p.base_addr
        # offset = p.offset
        # dtype = p.dtype
        # #print("h2")
        # if hasattr(p,'span'):
            # span=p.span
        # else:
            # span=-1
        
        # dims=[]
        # shape=[]
        # for i in range(self.ndim):
            # dims.append({})
            # dims[i]['stride'] = p.dims[i].stride
            # dims[i]['lbound'] = p.dims[i].lbound
            # dims[i]['ubound'] = p.dims[i].ubound
            
        # for i in range(self.ndim):
            # shape.append(dims[i]['ubound']-dims[i]['lbound']+1)
            
        # self._shape=tuple(shape)
        # size = np.product(shape)
        
        # #Counting starts at 1
        # # addr = base_addr + offset*span + ctypes.sizeof(self._ctype_single)
       # # if span > 0:
        # #    addr = base_addr + (p.dims[0].stride + offset) * span
        # #else:
            # # do this with 8 byte ints
            # #print(offset,p.dims[0].stride)
         # #   addr = base_addr + offset + p.dims[0].stride*ctypes.sizeof(self._ctype_single)
        # addr = base_addr
        
        # #print(base_addr,addr,span)
        # # print(base_addr,offset,span,ctypes.sizeof(self._ctype_single))
        # # print(addr)
        # # print(self._ctype_single.from_address(base_addr),self._ctype_single.from_address(addr))
        
        # # print(p.base_addr,p.offset,p.span,p.dims,p.dims[0].stride)
        # # print(p.dtype.elem_len,p.dtype.version,p.dtype.rank,p.dtype.type,p.dtype.attribute)

        # if copy:
            # # When we want a copy of the array not a pointer to the fortran memoray
            # res = self._get_var_from_address(addr,size=size)
            # res = np.asfortranarray(res)
            # res = res.reshape(shape).astype(self.npdtype)
        # else:
            # # When we want to pointer to the underlaying fortran memoray
            # # will leak as we dont have a deallocate call to call in a del func
            # ptr = ctypes.cast(addr,ctypes.POINTER(self._ctype_single))
            # res = np.ctypeslib.as_array(ptr,shape = self._shape)
        
        # remove_ownership(res)      

        # return res
        
# class fAssumedSize(fExplicitArray):
    # def ctype_def_func(self,pointer=False,intent=''):
        # """
        # The ctype type of a value suitable for use as an argument of a function

        # May just call ctype_def
        
        # Second return value is anythng that needs to go at the end of the
        # arg list, like a string len
        # """
        # #print("h1")
        # x=ctypes.POINTER(getattr(ctypes, self.ctype))
        # y=None
        # return x,y  
    
    # def _make_array_shape(self,bounds=None):
        
        # return [99]*self.ndims
    
    
    # def py_to_ctype_p(self,value):
        # """
        # The ctype representation suitable for function arguments wanting a pointer
        # """
        # self._data = value
        # ct = getattr(ctypes, self.ctype)
        # addr = self._data.ctypes.get_data()
        # t = ctypes.POINTER(ct)
        # return ctypes.cast(addr,t)
        
    # def py_to_ctype_f(self,value):
        # """
        # The ctype representation suitable for function arguments wanting a pointer
        # """
        # return self.py_to_ctype_p(value),None

# class fAllocatableArray(fDummyArray):
    # def py_to_ctype(self, value):
        # """
        # Pass in a python value returns the ctype representation of it
        # """
        # self.set_func_arg(value)
        
        # # self._value_array needs to be empty if the array is allocatable and not
        # # allready allocated
        # self._value_array.base_addr=ctypes.c_void_p(None)
        
        # return self._value_array
        
    # def py_to_ctype_f(self, value):
        # """
        # Pass in a python value returns the ctype representation of it, 
        # suitable for a function
        
        # Second return value is anything that needs to go at the end of the
        # arg list, like a string len
        # """
        # return self.py_to_ctype(value),None   
        
    # def ctype_to_py_f(self, value):
        # """
        # Pass in a ctype value returns the python representation of it,
        # as returned by a function (may be a pointer)
        # """
        # shape=[]
        # for i in value.dims:
            # shape.append(i.ubound-i.lbound+1)
        # shape=tuple(shape)
        # #print("here")
        # p=ctypes.POINTER(self._ctype_single)
        # res=ctypes.cast(value.base_addr,p)
        # z = np.ctypeslib.as_array(res,shape=shape)
        # remove_ownership(z)
        # return z
        

    
class fParamArray(fParent, np.lib.mixins.NDArrayOperatorsMixin):
    _HANDLED_TYPES = (np.ndarray, numbers.Number)
    
    def __init__(self, lib, obj):
        self.__dict__.update(obj)
        self._lib = lib
        self.pytype = self.param['pytype']
        self.pytype = getattr(__builtin__, self.pytype)
        self.value = np.array(self.param['value'], dtype=self.pytype, order='FORTRAN')
		
    def set(self, value):
        """
        Cant set a parameter
        """
        raise ValueError("Can't alter a parameter")
        
    def get(self):
        """
        A parameters value is stored in the dict, as we cant access them
        from the shared lib.
        """
        return self.value 


    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        out = kwargs.get('out', ())
        for x in inputs + out:
            # Only support operations with instances of _HANDLED_TYPES.
            # Use ArrayLike instead of type(self) for isinstance to
            # allow subclasses that don't override __array_ufunc__ to
            # handle ArrayLike objects.
            if not isinstance(x, self._HANDLED_TYPES + (fParamArray,)):
                return NotImplemented

        # Defer to the implementation of the ufunc on unwrapped values.
        inputs = tuple(x.value.get() if isinstance(x, fParamArray) else x
                       for x in inputs)
        if out:
            kwargs['out'] = tuple(
                x.get() if isinstance(x, fParamArray) else x
                for x in out)
        result = getattr(ufunc, method)(*inputs, **kwargs)

        if type(result) is tuple:
            # multiple return values
            return tuple(type(self)(x) for x in result)
        elif method == 'at':
            # no return value
            return None
        else:
            # one return value
            return type(self)(result)

    @property
    def __array_interface__(self):
        return self.get().__array_interface__

    @property
    def flags(self):
        return self.get().flags
