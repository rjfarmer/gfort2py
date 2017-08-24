from __future__ import print_function
import ctypes
from .var import fVar, fParam
from .errors import *

class fComplex(fVar):

    def __init__(self, lib, obj):
        self.__dict__.update(obj)
        self._lib = lib
        
        self.ctype=self.var['ctype']
        self.pytype=self.var['pytype']
        
        self._ctype = self.ctype_def()
        #self._ctype_f = self.ctype_def_func()
        self._pytype = self.pytype_def()
        
        #Store the ref to the lib object
        try:   
            self._ref = self._get_from_lib()
        except NotInLib:
            self._ref = None

    def py_to_ctype(self, value):
        """
        Pass in a python value returns the ctype representation of it
        """
        x = [value.real, value.imag]
        return self._set_var_from_iter(self._ref, x, 2)

    def ctype_to_py(self, value):
        """
        Pass in a ctype value returns the python representation of it
        """
        x = self._get_var_by_iter(value, 2)
        return self._pytype(x[0], x[1])

    def pytype_def(self):
        return complex

    def ctype_def(self):
        """
        The ctype type of this object
        """
        if '_cached_ctype' not in self.__dict__:
            self._cached_ctype = getattr(ctypes, self.ctype)
        
        return self._cached_ctype

    def set_mod(self, value):
        if isinstance(value, complex):
            self.py_to_ctype(value)
        else:
            raise ValueError("Not complex")

    def get(self,copy=True):
        s = self.ctype_to_py(self._ref)
        if not copy:
            raise ValueError("Must copy complex number")
        
        return s

    def __repr__(self):
        return str(self.get()) + " <complex>"


class fParamComplex(fParam):

    def get(self):
        """
        A parameters value is stored in the dict, as we cant access them 
        from the shared lib.
        """
        return self.value
