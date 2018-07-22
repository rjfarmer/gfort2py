from __future__ import print_function
import ctypes
from .var import fVar
from .errors import *

class fStr(fVar):

    def __init__(self, lib, obj):
        self.__dict__.update(obj)
        self._lib = lib
        self._ctype = self.ctype_def()
       # self._ctype_f = self.ctype_def_func()
        self._pytype = str
        self._dt_arg = False
        
        self.char_len = self.var['length']

        #Store the ref to the lib object
        try:   
            self._ref = self._get_from_lib()
        except NotInLib:
            self._ref = None

    def py_to_ctype(self, value):
        """
        Pass in a python value returns the ctype representation of it
        """
        return ctypes.c_char(value.encode())
        
    def py_to_ctype_f(self, value):
        """
        Pass in a python value returns the ctype representation of it, 
        suitable for a function
        
        Second return value is anythng that needs to go at the end of the
        arg list, like a string len
        """
        x,y=self.ctype_def_func()
        
        return x(value.encode()),y(len(value))

    def ctype_to_py(self, value):
        """
        Pass in a ctype value returns the python representation of it
        """
        return self._get_var_by_iter(value, self.char_len)
        
    def ctype_def(self):
        if self._dt_arg:
            return ctypes.c_char_p
        else:
            return ctypes.c_char

    def ctype_def_func(self,pointer=False,intent=''):
        """
        The ctype type of a value suitable for use as an argument of a function

        May just call ctype_def
        
        Second return value is anything that needs to go at the end of the
        arg list, like a string len
        """
        f = ctypes.c_char_p
        if pointer:
            f = ctypes.POINTER(f)
        
        return f,ctypes.c_int
        
    def py_to_ctype_p(self,value):
        """
        The ctype represnation suitable for function arguments wanting a pointer
        """
        return ctypes.c_char_p(value.encode())
        
    def ctype_to_py_f(self, value):
        """
        Pass in a ctype value returns the python representation of it,
        as returned by a function (may be a pointer)
        """
        if hasattr(value,'contents'):
            r = value.contents.value
        elif hasattr(value,'value'):
            r = value.value
        else:
            r = value
            
        try:
            r = r.decode()
        except AttributeError:
            pass
        except UnicodeDecodeError:
            r = ''
            
        return r
            

    def set_mod(self, value):
        """
        Set a module level variable
        """
        self._set_var_from_iter(self._ref, value.encode(), self.char_len)

    def get(self,copy=True):
        """
        Get a module level variable
        """
        s = self.ctype_to_py(self._ref)
        if not copy:
            raise ValueError("Must copy a string")
        
        return ''.join([i.decode() for i in s])
