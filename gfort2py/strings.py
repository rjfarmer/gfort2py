# SPDX-License-Identifier: GPL-2.0+
from __future__ import print_function
import ctypes

from .errors import *
from .utils import *

_NULL_BYTE = ctypes.c_char(b'\0').value


class fStr(object):
    def __init__(self, obj):
        self.__dict__.update(obj)
        self.ctype = self.var['ctype']

        self.len = int(self.var['length'])
        self.pytype = str

        if self.len > 0:
            self._fixed = True
            self.ctype = ctypes.c_char * self.len
        else:
            self._fixed = False
            self.ctype = ctypes.c_char * 1

    def from_address(self, addr):
        start = self.ctype.from_address(addr)[0]
        if start == _NULL_BYTE:
            return ''
        else:
            if self.len > 0:
                r = self.ctype.from_address(addr)
            else:
                r = self._get_var_from_address(addr)

            return ''.join([i.decode() for i in r])

    def set_from_address(self, addr, value):
        ctype = self.ctype.from_address(addr)
        self._set(ctype, value)

    def _set(self, c, v):
        if hasattr(v, 'encode'):
            v = v.encode()
        for i in range(self.len):
            if i < len(v):
                c[i] = v[i]
            else:
                c[i] = b' '

    def in_dll(self, lib):
        addr = ctypes.addressof(self.ctype.in_dll(lib, self.mangled_name))
        # Can't use self._fixed here as we make a new fStr everytime we access
        # it
        return self.from_address(addr)

    def set_in_dll(self, lib, value):
        if not self._fixed:
            self.len = len(value)
            self.ctype = ctypes.c_char * self.len

        addr = ctypes.addressof(self.ctype.in_dll(lib, self.mangled_name))
        self.set_from_address(addr, value)

    def from_param(self, value):
        if 'optional' in self.var:
            if self.var['optional'] and value is None:
                return None

        if len(value):
            self.len = len(value)
        else:
            self.len = 1

        self.ctype = ctypes.c_char * self.len

        self._safe_ctype = self.ctype()
        self._set(self._safe_ctype, value)

        if 'pointer' in self.var and self.var['pointer']:
            return ctypes.pointer(ctypes.pointer(self._safe_ctype))
        else:
            return ctypes.pointer(self._safe_ctype)

    def from_func(self, pointer):

        x = pointer
        if hasattr(pointer, 'contents'):
            if hasattr(pointer.contents, 'contents'):
                x = pointer.contents.contents
            else:
                x = pointer.contents

        return self.from_address(ctypes.addressof(x))

    def from_len(self, pointer, length):
        x = pointer
        if hasattr(pointer, 'contents'):
            if hasattr(pointer.contents, 'contents'):
                x = pointer.contents.contents
            else:
                x = pointer.contents

        addr = ctypes.addressof(x)
        if hasattr(length, 'value'):
            length = length.value

        if length == 0:
            c = ctypes.c_char * self.len
        else:
            c = ctypes.c_char * length

        return ''.join([i.decode() for i in c.from_address(addr)])

    def _get_var_from_address(self, ctype_address, size=-1):
        out = []
        i = 0
        sof = ctypes.sizeof(ctypes.c_char)
        while True:
            if i == size:
                break
            x = ctypes.c_char.from_address(ctype_address + i * sof)
            # Null or padding (\x08 occurs length is  a multiple of 16)
            if x.value == b'\x00' or x.value == b'\x08':
                break
            else:
                out.append(x.value)
            i = i + 1
        return out


class fStrLen(object):
    # Handles the hidden string length functions need
    def __init__(self):
        self.ctype = ctypes.c_int64

    def from_param(self, value):
        return self.ctype(len(value))

    def from_func(self, pointer):
        raise IgnoreReturnError
