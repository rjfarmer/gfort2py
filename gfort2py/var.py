try:
    import __builtin__
except ImportError:
    import builtins as __builtin__

import ctypes
import numpy as np

# Hacky, yes
__builtin__.quad = np.longdouble


class fVar(object):

    def __init__(self, lib, obj,TEST_FLAG=False):
        self.__dict__.update(obj)
        self._lib = lib
        self._ctype = self.ctype_def()
        #self._ctype_f = self.ctype_def_func()
        self._pytype = self.pytype_def()
        if self.pytype == 'quad':
            self.pytype = np.longdouble

        # if true for things that are fortran things
        self._fortran = True
        
        # True if its a function argument
        self._func_arg=False
        
        self.TEST_FLAG=TEST_FLAG

    def py_to_ctype(self, value):
        """
        Pass in a python value returns the ctype representation of it
        """
        return self._cytype(value)

    def ctype_to_py(self, value):
        """
        Pass in a ctype value returns the python representation of it
        """
        return self._pytype(value.value)

    def pytype_def(self):
        return getattr(__builtin__, self.pytype)

    def ctype_def(self):
        """
        The ctype type of this object
        """
        return getattr(ctypes, self.ctype)

    def ctype_def_func(self):
        """
        The ctype type of a value suitable for use as an argument of a function

        May just call ctype_def
        """
        c = None
        #if 'intent' not in self.__dict__.keys():
            #c = self.ctype_def()
        #elif self.intent == "out" or self.intent == "inout" or self.pointer:
            #c = ctypes.POINTER(self.ctype_def())
        #else:
            #c = self.ctype_def()
        #return c
        return ctypes.POINTER(self.ctype_def())

    def set_mod(self, value):
        """
        Set a module level variable
        """
        r = self._get_from_lib()
        r.value = self._pytype(value)

    def get(self):
        """
        Get a module level variable
        """
        r = self._get_from_lib()
        return self.ctype_to_py(r)

    def _get_from_lib(self):
        res = None
        try:
            res = self._ctype.in_dll(self._lib, self.mangled_name)
        except AttributeError:
            raise
        return res

    def _get_var_by_iter(self, value, size=-1):
        """ Gets a variable where we have to iterate to get multiple elements"""
        base_address = ctypes.addressof(value)
        return self._get_var_from_address(base_address, size=size)

    def _get_var_from_address(self, ctype_address, size=-1):
        out = []
        i = 0
        sof = ctypes.sizeof(self._ctype)
        while True:
            if i == size:
                break
            x = self._ctype.from_address(ctype_address + i * sof)
            if x.value == b'\x00':
                break
            else:
                out.append(x.value)
            i = i + 1
        return out

    def _set_var_from_iter(self, res, value, size=99999):
        base_address = ctypes.addressof(res)
        self._set_var_from_address(base_address, value, size)

    def _set_var_from_address(self, ctype_address, value, size=99999):
        for j in range(min(len(value), size)):
            offset = ctype_address + j * ctypes.sizeof(self._ctype)
            self._ctype.from_address(offset).value = value[j]

    def _pname(self):
        return str(self.name) + " <" + str(self.pytype) + ">"

    def __str__(self):
        return str(self.get())

    def __repr__(self):
        s=''
        try:
            s=str(self.get()) + " <" + str(self.pytype) + ">"
        except ValueError:
            # Skip for things that aren't in the module (function arg)
            s=" <" + str(self.pytype) + ">"
        return s
    
    def __getattr__(self, name): 
        return getattr(self.get(), name)

    def __add__(self, other):
        return self.get() + other

    def __sub__(self, other):
        return self.get() - other

    def __mul__(self, other):
        return self.get() * other

    def __truediv__(self, other):
        return self.get() / other

    def __pow__(self, other, modulo=None):
        x = self.get()**other
        if modulo:
            x = x % modulo
        return x

    def __iadd__(self, other):
        self.set_mod(self.get() + other)
        return self.get()

    def __isub__(self, other):
        self.set_mod(self.get() - other)
        return self.get()

    def __imul__(self, other):
        self.set_mod(self.get() * other)
        return self.get()

    def __itruediv__(self, other):
        self.set_mod(self.get() / other)
        return self.get()

    def __ipow__(self, other, modulo=None):
        x = self.get()**other
        if modulo:
            x = x % modulo
        self.set_mod(x)
        return self.get()

    def __eq__(self, other):
        return self.get() == other

    def __neq__(self, other):
        return self.get() != other

    def __lt__(self, other):
        return self.get() < other

    def __le__(self, other):
        return self.get() <= other

    def __gt__(self, other):
        return self.get() > other

    def __ge__(self, other):
        return self.get() >= other
        
    def __dir__(self):
        return list(self.__dict__.keys()) + list(dir(self.get()))


class fParam(fVar):

    def set_mod(self, value):
        """
        Cant set a parameter
        """
        raise ValueError("Can't alter a parameter")

    def get(self):
        """
        A parameters value is stored in the dict, as we cant access them 
        from the shared lib.
        """
        return self._pytype(self.value)
