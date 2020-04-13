# SPDX-License-Identifier: GPL-2.0+
from __future__ import print_function
import ctypes

try:
    import __builtin__
except ImportError:
    import builtins as __builtin__

from .var import fVar, fParam
from .errors import *
from .utils import *


class fComplex(object):
    def __init__(self, obj):
        self.__dict__.update(obj)
        self.ctype=self.var['ctype']
        self.pytype=self.var['pytype']

        if self.pytype == 'quad':
            self.pytype = np.longdouble
        elif self.pytype=='bool':
            self.pytype=int
            self.ctype='c_int32'
        else:
             self.pytype = getattr(__builtin__, self.pytype)

        self._single_ctype = getattr(ctypes, self.ctype)

        class _complex(ctypes.Structure):
            _fields_ = [('real', self._single_ctype),
                             ('imag', self._single_ctype)]

        self.ctype = _complex
        
    def from_address(self, addr):
        return self.ctype.from_address(addr)
        
    def set_from_address(self, addr, value):
        ctype = self.from_address(addr)
        self._set(ctype, value)
        
    def _set(self, c, v):
        c.real = v.real
        c.imag = v.imag 

    def from_param(self, value):
        ctype  = self.ctype()
        self._set(ctype, value)
        
        if 'optional' in self.var :
            if self.var['optional'] and value is None:
                return None
                
        if 'value' in self.var and self.var['value']:
            return ctype
        else:
            ctype = ctypes.pointer(ctype)
            if 'pointer' in self.var and self.var['pointer']:
                return ctypes.pointer(ctype)
            else:
                return ctype
        
        
        return ctype

    def in_dll(self, lib):
        addr = ctypes.addressof(self.ctype.in_dll(lib, self.mangled_name))
        c = self.from_address(addr)
        return complex(c.real, c.imag)
        
    def set_in_dll(self,lib, value):
        addr = ctypes.addressof(self.ctype.in_dll(lib, self.mangled_name))
        self.set_from_address(addr, value)
        
    def from_func(self, pointer):
        """
        Given the pointer to the variable return the python type
        
        We raise IgnoreReturnError when we get something that should not be returned,
        like the hidden string length that's at the end of the argument list.
        
        """       
        x = pointer
        if hasattr(pointer,'contents'):
            if hasattr(pointer.contents,'contents'):
                x = pointer.contents.contents
            else:
                x = pointer.contents
                
        return complex(x.real, x.imag)

    def sizeof(self):
        return ctypes.sizeof(self.ctype)

class fParamComplex(object):
    def __init__(self, obj):
        self.__dict__.update(obj)
        self.value = self.param['value']
        self.pytype = self.param['pytype']
        self.pytype = complex
		
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
        return self.pytype(self.value)
