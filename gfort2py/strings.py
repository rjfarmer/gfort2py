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

    def get(self):
        return self.str_from_address(ctypes.addressof(self.in_dll()))
        
    def set(self, value):            
       self.set_from_address(ctypes.addressof(self.in_dll()),value)
  
    def from_param(self, value):        
        if 'optional' in self.var :
            if self.var['optional'] and value is None:
                return None
                
        self.len = len(value)
        self.ctype = ctypes.c_char * self.len
        
        self._safe_ctype  = self.ctype()
        self._set(self._safe_ctype, value)
                
        if 'pointer' in self.var and self.var['pointer']:
            return ctypes.pointer(ctypes.pointer(self._safe_ctype))
        else:
            return ctypes.pointer(self._safe_ctype)
        
        

    def from_func(self, pointer):
        
        x = pointer
        if hasattr(pointer,'contents'):
            if hasattr(pointer.contents,'contents'):
                x = pointer.contents.contents
            else:
                x = pointer.contents
        
        return self.str_from_address(ctypes.addressof(x))


class fStrLen(object):
    # Handles the hidden string length functions need
    def __init__(self):
        pass
        
    def from_param(self, value):
        return ctypes.c_int64(len(value))
        
    def from_func(self, pointer):
        raise IgnoreReturnError
