# SPDX-License-Identifier: GPL-2.0+
from __future__ import print_function

import ctypes
import os
import copy
import collections
import numpy as np

from .errors import *
from .selector import _selectVar


_alldtdefs = {}


class emptyDT(ctypes.Structure):
    def __init__(self, obj):
        pass


class fDerivedType(object):
    def __init__(self, obj):
        self._obj = obj
        self.__dict__.update(obj)
        self._comp = collections.OrderedDict()

        self._ndims = 1
        self._dt_desc = self._init_keys()

        self.ctype = self._dt_desc
        self._safe_ctype = None
        self._addr = None

    def _init_keys(self):
        dtdef = _alldtdefs[self.var['dt']['name'].lower().replace("'",'')]

        for i in dtdef['dt_def']['arg']:
            self._comp[i['name']] = self._get_fvar(i)(i)

        class ctypesStruct(ctypes.Structure):
            _fields_ = [(key, value.ctype)
                        for key, value in self._comp.items()]

        if self._isarray():
            self._ndims = int(self.var['array']['ndim'])
            ctypesStruct = ctypesStruct * self._size()

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
        self._addr = ctypes.addressof(
            self.ctype.in_dll(
                lib, self.mangled_name))
        return self

    def set_in_dll(self, lib, value):
        self._addr = ctypes.addressof(
            self.ctype.in_dll(
                lib, self.mangled_name))
        if not (isinstance(value, dict) or isinstance(value, fDerivedType)):
            raise ValueError(
                "Input must be a dict or an existing derived type")

        for k in value.keys():
            if k not in self.keys():
                raise KeyError("No key by name " + str(k))

        for key, items in value.items():
            self.__setattr__(key, items)

    def set_all(self, value):
        for k in value.keys():
            if k not in self.keys():
                raise KeyError("No key by name " + str(k))

        for key, items in value.items():
            self.__setattr__(key, items)

    def keys(self):
        return self._comp.keys()

    def __getitem__(self, key):
        ind = self._itemsetup(key)

        addr = ctypes.addressof(self.ctype.from_address(self._addr)[ind])

        return self._newdt_fromarray(addr)

    def __setitem__(self, key, value):
        ind = self._itemsetup(key)

        addr = ctypes.addressof(self.ctype[ind])

        x = self._newdt(_alldtdefs[self.var['dt']['name']], addr)
        x.set_all(value)

    def _itemsetup(self, key):
        # Only if arrays
        if not self._isarray():
            raise TypeError("Not an array")

        if isinstance(key, tuple):
            if len(key) != self._ndims:
                print(key, self._ndims)
                raise IndexError("Wrong number of dimensions")
            ind = np.ravel_multi_index(key, self._shape())
        else:
            ind = key

        if ind > self._size():
            raise ValueError("Out of bounds")

        return ind

    def __getattr__(self, key):
        if '_comp' in self.__dict__ and key in self.keys():
            if self._addr is None:
                raise ValueError("Must point to something first")

            addr = self._addr + getattr(self.ctype, key).offset
            obj = self._comp[key]
            res = obj.from_address(addr)

            if 'dt' in obj.var:
                return self._newdt_comp(key, addr)
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
                x = self._newdt_comp(key, addr)
                x.set_all(value)
            else:
                obj.set_from_address(addr, value)

        else:
            self.__dict__[key] = value

    def from_param(self, value):
        if 'optional' in self.var:
            if self.var['optional'] and value is None:
                return None

        if not (isinstance(value, dict) or isinstance(value, fDerivedType)):
            raise ValueError(
                "Input must be a dict or an existing derived type")

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
        if hasattr(pointer, 'contents'):
            if hasattr(pointer.contents, 'contents'):
                x = pointer.contents.contents
            else:
                x = pointer.contents

        self._safe_ctype = x
        return self

        
    def _get_fvar(self, var):
        if 'dt' in var['var'] and var['var']['dt']:
            if var['var']['dt']['name'] == self.var['dt']['name']: 
                raise TypeError(
                        "Cant not support recursive derived types yet")
                
        return _selectVar(var)

    def __repr__(self):
        return str(self.var['dt']['name'])

    def __iter__(self):
        return self.keys()

    def __contains__(self, name):
        return name in self.keys()

    def _isarray(self):
        return 'array' in self.var and self.var['array']

    def _shape(self):
        if self._isarray():
            if 'shape' not in self.var['array'] or len(
                    self.var['array']['shape']) / self._ndims != 2:
                return -1

            shape = []
            for l, u in zip(self.var['array']['shape']
                            [0::2], self.var['array']['shape'][1::2]):
                shape.append(u - l + 1)
            return tuple(shape)
        else:
            raise AttributeError

    def _size(self):
        return np.product(self._shape())

    def _newdt_comp(self, key, addr):
        dt_desc = self.var['dt']['name']
        dtdef = _alldtdefs[dt_desc]
        for i in dtdef['dt_def']['arg']:
            if i['name'] == key:
                x = fDerivedType(i)
                break
        x._addr = addr
        return x

    def _newdt_fromarray(self, addr):
        obj = copy.deepcopy(self._obj)
        del(obj['var']['array'])
        x = fDerivedType(obj)
        x._addr = addr
        return x
