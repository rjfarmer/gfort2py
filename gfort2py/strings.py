# SPDX-License-Identifier: GPL-2.0+
from __future__ import print_function
import ctypes

from .errors import *
from .utils import *

class fStr(fParent):
    def __init__(self, lib, obj):
        self.__dict__.update(obj)
        self._lib = lib
        self.ctype=self.var['ctype']

        self.len = int(self.var['length'])
        self.pytype = str
        
        if self.len > 0:
            self.ctype = ctypes.c_char * self.len
        else:
            self.ctype = ctypes.c_char
        
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
        if hasattr(v,'encode'):
            v = v.encode()
        for i in range(self.len):
            if i < len(v):
                c[i] = v[i]
            else:
                c[i] = b' '
                
    def str_from_address(self, addr):
        return ''.join([i.decode() for i in self.from_address(addr)])
                        
    def sizeof(self):
        return ctypes.sizeof(self.ctype)

    def get(self):
        return self.str_from_address(ctypes.addressof(self.in_dll()))
        
    def set(self, value):            
       self.set_from_address(ctypes.addressof(self.in_dll()),value)
  
    def from_param(self, value):
        self.len = len(value)
        self.ctype = ctypes.c_char * self.len
        
        self._safe_ctype  = self.ctype()
        self._set(self._safe_ctype, value)
        return ctypes.pointer(self._safe_ctype)

    def from_func(self, pointer):
        return self.str_from_address(ctypes.addressof(pointer.contents))


class fStrLen(object):
    # Handles the hidden string length functions need
    def __init__(self):
        pass
        
    def from_param(self, value):
        return ctypes.c_int64(len(value))
        
    def from_func(self, pointer):
        raise IgnoreReturnError
