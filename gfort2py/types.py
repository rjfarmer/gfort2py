# SPDX-License-Identifier: GPL-2.0+
from __future__ import print_function
try:
    import __builtin__
except ImportError:
    import builtins as __builtin__

import ctypes
import os
import collections
import numpy as np


from .utils import *
from .errors import *
from .selector import _selectVar


_alldtdefs = {}



class fDerivedType(object):
    def __init__(self, obj):
        self.__dict__.update(obj)
        self._comp = collections.OrderedDict()
        self._dt_desc = self._init_keys()

        self.ctype  = self._dt_desc 
        self._safe_ctype = None
        self._addr = None

    def _init_keys(self):
        dtdef = _alldtdefs[self.var['dt']['name']]

        for i in  dtdef['dt_def']['arg']:
            self._comp[i['name']] = self._get_fvar(i)(i)

        class ctypesStruct(ctypes.Structure):
            _fields_ = [(key, value.ctype) for key,value in  self._comp.items()]
            
        return ctypesStruct

    def from_address(self, addr):
        self._addr = addr
        return self
        
    def sizeof(self):
        """ Gets the size in bytes of the ctype representation """
        return ctypes.sizeof(self.ctype)
    
    def get(self):
        return None
            
    def in_dll(self, lib):
        self._addr = ctypes.addressof(self.ctype.in_dll(lib, self.mangled_name))
        return self
    
    def set_in_dll(self, lib, value):
        self._addr = ctypes.addressof(self.ctype.in_dll(lib, self.mangled_name))
        if not (isinstance(value,dict) or isinstance(value,fDerivedType)):
            raise ValueError("Input must be a dict or an existing derived type")
        
        for k in value.keys():
            if k not in self.keys():
                raise KeyError("No key by name "+str(k))
        
        for key, items in value.items():
            self.__setattr__(key,items)         
            
    def set_all(self, value):
        for k in value.keys():
            if k not in self.keys():
                raise KeyError("No key by name "+str(k))
        
        for key, items in value.items():
            self.__setattr__(key,items)       

    def keys(self):
        return self._comp.keys()
        
        
    def __getattr__(self, key):
        if '_comp' in self.__dict__ and key in self.keys():
            if self._addr is None:
                raise ValueError("Must point to something first")
            
            addr = self._addr + getattr(self.ctype, key).offset
            obj = self._comp[key]
            res = obj.from_address(addr)
            
            if 'dt' in obj.var:
                dt_desc = self.var['dt']['name']
                dtdef = _alldtdefs[dt_desc]
                for i in dtdef['dt_def']['arg']:
                    if i['name'] == key:
                        x = fDerivedType(i)
                        break
                x._addr = addr
                return x
            else:
                if hasattr(res, 'value'):
                    return obj.pytype(res.value)
                else:
                    return res
        else:
            return self.__dict__[key]
        
    def __setattr__(self, key, value):
        if '_comp' in self.__dict__ and key in self.keys():
            if self._addr is None:
                raise ValueError("Must point to something first")
            
            addr = self._addr + getattr(self.ctype, key).offset
            obj = self._comp[key]
            
            if 'dt' in obj.var:
                dt_desc = self.var['dt']['name']
                dtdef = _alldtdefs[dt_desc]
                for i in dtdef['dt_def']['arg']:
                    if i['name'] == key:
                        x = fDerivedType(i)
                        break
                x._addr = addr
                x.set_all(value)
            else:
                obj.set_from_address(addr, value)
            
        else:
            self.__dict__[key] = value
        

    def from_param(self, value):
        if 'optional' in self.var :
            if self.var['optional'] and value is None:
                return None
                
        if not (isinstance(value,dict) or isinstance(value,fDerivedType)):
            raise ValueError("Input must be a dict or an existing derived type")

        # Hold a chunk of memory the size of the object
        self._safe_ctype = self._dt_desc()
        self._addr = ctypes.addressof(self._safe_ctype)
        self.set_all(value)
        
        ct = self._safe_ctype
                
        if 'value' in self.var and self.var['value']:
            return ct
        else:
            ct = ctypes.pointer(ct)
            if 'pointer' in self.var and self.var['pointer']:
                return ctypes.pointer(ct)
            else:
                return ct
        

    def from_func(self, pointer):
        x = pointer
        if hasattr(pointer,'contents'):
            if hasattr(pointer.contents,'contents'):
                x = pointer.contents.contents
            else:
                x = pointer.contents
                
        self._safe_ctype = x
        return self
        
    def _get_fvar(self,var):
        x = _selectVar(var)
        if x is None: # Handle derived types
            if 'dt' in var['var'] and var['var']['dt']:
                x = fDerivedType
            else:
                raise TypeError("Can't match ",var['name'])
        return x


    def __repr__(self):
        return str(self.var['dt']['name'])


    def __iter__(self):
        return self.keys()
        
    def __contains__(self, name):
        return name in self.keys()
