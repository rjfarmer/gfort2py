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


class fComplex(fParent):
    def __init__(self, lib, obj):
        self.__dict__.update(obj)
        self._lib = lib
        self.ctype=self.var['ctype']
        self.pytype=self.var['pytype']

        if self.pytype == 'quad':
            self.pytype = np.longdouble
        elif self.pytype=='bool':
            self.pytype=int
            self.ctype='c_int32'
        else:
             self.pytype = getattr(__builtin__, self.pytype)

        self.ctype = getattr(ctypes, self.ctype)*2

    def in_dll(self):
        if 'mangled_name' in self.__dict__ and '_lib' in self.__dict__:
            try:
                return self.ctype.in_dll(self._lib, self.mangled_name)
            except ValueError:
                raise NotInLib
        raise NotInLib 
        
    def from_address(self, addr):
        return self.ctype.from_address(addr)
        
    def set_from_address(self, addr, value):
        ctype = self.from_address(addr)
        self._set(ctype, value)
        
    def _set(self, c, v):
        c[0] = v.real
        c[1] = v.imag 
        
    def sizeof(self):
        return ctypes.sizeof(self.ctype)

    def get(self):
        return complex(*self.from_address(ctypes.addressof(self.in_dll())))

    def set(self, value):      
        self.set_from_address(ctypes.addressof(self.in_dll()),value)
        
    def from_param(self, value):
        ctype  = self.ctype()
        self._set(ctype, value)
        return ctype


class fParamComplex(fParent):
    def __init__(self, lib, obj):
        self.__dict__.update(obj)
        self._lib = lib
        self.value = self.param['value']
        self.pytype = self.param['pytype']
        self.pytype = complex
		
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
        return self.pytype(self.value)
