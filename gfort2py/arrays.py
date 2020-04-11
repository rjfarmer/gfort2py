# SPDX-License-Identifier: GPL-2.0+
from __future__ import print_function
import ctypes
import sys
import numpy as np

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


if sys.byteorder == 'little':
    _byte_order=">"
else:
    _byte_order="<"
    
class BadFortranArray(Exception):
    pass
    

class fExplicitArray(object):            
    def __init__(self, obj):
        self.__dict__.update(obj)
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
        
    def from_address(self, addr):
        self._base_holder = addr # Make sure to hold onto the object
        buff = {
            'data': (self._base_holder,
                    False),
                'typestr': self._dtype,
                'shape': self._shape,
                'version':3
                }

        class numpy_holder():
            pass

        holder = numpy_holder()
        holder.__array_interface__ = buff
        
        arr = np.asfortranarray(holder)
        remove_ownership(arr)
        
        return arr

    def shape(self):
        if 'shape' not in self.array or len(self.array['shape'])/self._ndims != 2:
            return -1
        
        shape = []
        for l,u in zip(self.array['shape'][0::2],self.array['shape'][1::2]):
            shape.append(u-l+1)
        return tuple(shape)
        
    def size(self):
        return np.product(self.shape())

    def set_from_address(self, addr, value):
        ctype = self.ctype.from_address(addr)
        self._set(ctype, value)

    def _set(self, c, v):
        if v.ndim != self._ndims:
            raise AttributeError("Bad ndims for array")
        
        if v.shape != self._shape:
            raise AttributeError("Bad shape for array")
            
        self._value = v
            
        self._value  = np.asfortranarray(self._value .astype(self._dtype)).T
        v_addr = self._value.ctypes.data

        ctypes.memmove(ctypes.addressof(c), v_addr, self.sizeof())
        remove_ownership(self._value)
 
    def set(self, value):
        self._set(self.in_dll(), value)
        
    def in_dll(self, lib):
        addr = ctypes.addressof(self.ctype.in_dll(lib, self.mangled_name))
        return self.from_address(addr).T
        
    def set_in_dll(self, lib, value):
        addr = ctypes.addressof(self.ctype.in_dll(lib, self.mangled_name))
        self.set_from_address(addr, value)
        
    def from_param(self, value):
        size = np.size(value)
        self._shape = np.shape(value)
        self.ctype = self.ctype * size
        
        self._safe_ctype = self.ctype()
        self._set(self._safe_ctype, value)
        return self._safe_ctype
        
    def from_func(self, pointer):
        return self.from_address(ctypes.addressof(pointer)).T
        
    def sizeof(self):
        return ctypes.sizeof(self.ctype)
        

class fDummyArray(object):
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

    def __init__(self, obj):
        self.__dict__.update(obj)
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
        
        self._array_desc = _make_fAlloc15(self.ndim)
        
        self.ctype = self._array_desc
        
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
        
    def from_address(self, addr):
        if self.ctype.from_address(addr).base_addr is None:
            raise AllocationError("Array not allocated yet")
        
        
        v = self._array_desc.from_address(addr)
        self._base_holder = v.base_addr # Make sure to hold onto the object
        buff = {
            'data': (self._base_holder,
                    False),
            'typestr': self._dtype,
            'shape': self._shape_from_bounds(v.dims),
            'version':3
            }
        
        class numpy_holder():
            pass
        
        holder = numpy_holder()
        holder.__array_interface__ = buff
        arr = np.asfortranarray(holder)
        remove_ownership(arr)
        
        return arr

    def set_from_address(self, addr, value):
        ctype = self._array_desc.from_address(addr)
        self._set(ctype, value)
        
    def _set(self, c, v):
        if v.ndim != self.ndim:
            raise AllocationError("Bad ndim for array")
        
        self._value = v
        
        self._value= np.asfortranarray(self._value.astype(self._dtype))

        c.base_addr = self._value.ctypes.data
 
        strides = []
        for i in range(self.ndim):
            c.dims[i].lbound = _index_t(1)
            c.dims[i].ubound = _index_t(self._value.shape[i])
            strides.append(c.dims[i].ubound-c.dims[i].lbound+1)

        c.span = ctypes.sizeof(self.ctype_elem)
        c.dtype = self._get_dtype15()
        for i in range(self.ndim):
            c.dims[i].stride = _index_t(int(np.product(strides[:i])))  

        c.offset = -c.dims[-1].stride
        remove_ownership(self._value)
 
    def in_dll(self, lib):
        addr = ctypes.addressof(self.ctype.in_dll(lib, self.mangled_name))
        return self.from_address(addr)
        
    def set_in_dll(self, lib, value):
        addr = ctypes.addressof(self.ctype.in_dll(lib, self.mangled_name))
        self.set_from_address(addr, value)
 
    def from_param(self, value):
        if self._array['atype'] == 'alloc' or self._array['atype'] == 'pointer' :
            
            self._safe_ctype =  self._array_desc()
            if value is not None:
                self.set_from_address(ctypes.addressof(self._safe_ctype), value)
            
            self._ptr_safe_ctype = ctypes.pointer(self._safe_ctype)
            return self._ptr_safe_ctype
        else:
            self._safe_ctype =  self._array_desc()
            self.set_from_address(ctypes.addressof(self._safe_ctype), value)
            return self.ctype.from_address(ctypes.addressof(self._safe_ctype))  
           
    def from_func(self, pointer):
        if self._array['atype'] == 'alloc' or self._array['atype'] == 'pointer' :
            return self.from_address(ctypes.addressof(pointer.contents))   
        else:
            return self.from_address(ctypes.addressof(pointer))          
        
        
class fAssumedShape(fDummyArray):
    
    def from_param(self, value):
        self._safe_ctype =  self._array_desc()
        if value is not None:
            self.set_from_address(ctypes.addressof(self._safe_ctype), value)
        
        self._ptr_safe_ctype = ctypes.POINTER(self._array_desc)(self._safe_ctype)
        
        return self._ptr_safe_ctype
           
    def from_func(self, pointer):
        return self.from_address(ctypes.addressof(pointer.contents))       
           
class fAssumedSize(fExplicitArray):
    # Only difference between this and an fExplicitArray is we don't know the shape.
    # We just pass the pointer to first element
    pass

    
class fParamArray(object):
    def __init__(self, obj):
        self.__dict__.update(obj)
        self.pytype = self.param['pytype']
        self.pytype = getattr(__builtin__, self.pytype)
        self.value = np.array(self.param['value'], dtype=self.pytype, order='F')
		
    def set_in_dll(self, lib, value):
        """
        Cant set a parameter
        """
        raise ValueError("Can't alter a parameter")
        
    def in_dll(self, lib):
        """
        A parameters value is stored in the dict, as we cant access them
        from the shared lib.
        """
        return self.value 

