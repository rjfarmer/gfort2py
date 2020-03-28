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
        
    def from_address(self, addr):
        """
        Given an address, return a ctype object from that address.
        
        addr -- integer address
        
        Returns:
        A ctype representation of the variable
        """
        return self.ctype.from_address(addr)
        
    def set_from_address(self, addr, value):
        """
        Given an address set the variable to value
        
        addr -- integer address
        value -- python object that can be coerced into self.pytype
        
        """
        self.from_address(addr).value = self.pytype(value)
        
    def get(self):
        """
        Get a fortran module variable from the library
        
        Returns:
        
        A python representation of the variable
        """
        return self.pytype(self.from_address(ctypes.addressof(self.in_dll())).value)

    def set(self, value):
        """
        Set a fortran module variable from the library to value.
        
        value -- python object that can be coerced into self.pytype
        """
        self.set_from_address(ctypes.addressof(self.in_dll()),value)
        
    def from_param(self, value):
        """
        Returns a ctype object needed by a function call
        
        This is a pointer to self.ctype(value).
        
        If the function argument is optional, and value==None we return None
        else we return the pointer.
        
        value -- python object that can be coerced into self.pytype
        
        """
        
        if 'optional' in self.var :
            if self.var['optional'] and value is None:
                return None
                
        if 'pointer' in self.var and self.var['pointer']:
            return ctypes.pointer(ctypes.pointer(self.ctype(self.pytype(value))))
        else:
            return ctypes.pointer(self.ctype(self.pytype(value)))
        
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
                
        try:
            return self.pytype(x.value)
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
        Can't set a parameter
        """
        raise ValueError("Can't alter a parameter")

    def get(self):
        """
        A parameters value is stored in the objects dict, as we can't access them
        from the shared lib.
        """
        return self.pytype(self.value)
