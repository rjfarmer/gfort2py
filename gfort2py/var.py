# SPDX-License-Identifier: GPL-2.0+
from __future__ import print_function
try:
    import __builtin__
except ImportError:
    import builtins as __builtin__

import ctypes
import numpy as np

from .errors import *


class fVar(object):
    def __init__(self, obj):
        self.__dict__.update(obj)
        self.ctype = self.var['ctype']
        self.pytype = self.var['pytype']

        if self.pytype == 'quad':
            self.pytype = np.longdouble
        elif self.pytype == 'bool':
            self.pytype = bool
            self.ctype = 'c_int32'
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
        return self.pytype(self.ctype.from_address(addr).value)

    def set_from_address(self, addr, value):
        """
        Given an address set the variable to value

        addr -- integer address
        value -- python object that can be coerced into self.pytype

        """
        self.ctype.from_address(addr).value = self.pytype(value)

    def from_param(self, value):
        """
        Returns a ctype object needed by a function call

        This is a pointer to self.ctype(value).

        If the function argument is optional, and value==None we return None
        else we return the pointer.

        value -- python object that can be coerced into self.pytype

        """

        if 'optional' in self.var:
            if self.var['optional'] and value is None:
                return None

        ct = self.ctype(self.pytype(value))

        if 'value' in self.var and self.var['value']:
            return ct
        else:
            ct = ctypes.pointer(ct)
            if 'pointer' in self.var and self.var['pointer']:
                return ctypes.pointer(ct)
            else:
                return ct

    def from_func(self, pointer):
        """
        Given the pointer to the variable return the python type

        We raise IgnoreReturnError when we get something that should not be returned,
        like the hidden string length that's at the end of the argument list.

        """
        x = pointer
        if hasattr(pointer, 'contents'):
            if hasattr(pointer.contents, 'contents'):
                x = pointer.contents.contents
            else:
                x = pointer.contents

        if hasattr(x, 'value'):
            x = x.value

        if x is None:
            return None

        try:
            return self.pytype(x)
        except AttributeError:
            raise IgnoreReturnError

    def in_dll(self, lib):
        return self.ctype.in_dll(lib, self.mangled_name).value

    def set_in_dll(self, lib, value):
        self.ctype.in_dll(lib, self.mangled_name).value = value


class fParam(object):
    def __init__(self, obj):
        self.__dict__.update(obj)
        self.value = self.param['value']
        self.pytype = self.param['pytype']
        if self.pytype == 'quad':
            self.pytype = np.longdouble
        elif self.pytype == 'bool':
            self.pytype = bool
            self.ctype = 'c_int32'
        else:
            self.pytype = getattr(__builtin__, self.pytype)

    def set_in_dll(self, lib, value):
        """
        Can't set a parameter
        """
        raise ValueError("Can't alter a parameter")

    def in_dll(self, lib):
        """
        A parameters value is stored in the objects dict, as we can't access them
        from the shared lib.
        """
        return self.pytype(self.value)
