# SPDX-License-Identifier: GPL-2.0+
from __future__ import print_function
try:
    import __builtin__
except ImportError:
    import builtins as __builtin__

import ctypes
import numpy as np

from .errors import *
from .utils import *

# Hacky, yes
__builtin__.quad = np.longdouble


class fVar(fParent):
    def __init__(self, lib, obj):
        self.__dict__.update(obj)
        self._lib = lib
        self.ctype=self.var['ctype']
        self.pytype=self.var['pytype']

        if self.pytype == 'quad':
            self.pytype = np.longdouble
        elif self.pytype=='bool':
            self.pytype=bool
            self.ctype='c_int32'
        else:
             self.pytype = getattr(__builtin__, self.pytype)

        self.ctype = getattr(ctypes, self.ctype)

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
        self.from_address(addr).value = self.pytype(value)
        
    def sizeof(self):
        return ctypes.sizeof(self.ctype)

    def get(self):
        return self.from_address(ctypes.addressof(self.in_dll())).value

    def set(self, value):
        self.set_from_address(ctypes.addressof(self.in_dll()),value)
        
    def from_param(self, value):
        if 'optional' in self.var and value is None:
            if self.var['optional']:
                return None
        
        return ctypes.pointer(self.ctype(value))
        
    def from_func(self, pointer):
        try:
            return pointer.contents.value
        except AttributeError:
            raise IgnoreReturnError

class fParam(fParent):
    def __init__(self, lib, obj):
        self.__dict__.update(obj)
        self._lib = lib
        self.value = self.param['value']
        self.pytype = self.param['pytype']
        self.pytype = getattr(__builtin__, self.pytype)
		
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
