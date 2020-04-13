# SPDX-License-Identifier: GPL-2.0+
from __future__ import print_function
try:
    import __builtin__
except ImportError:
    import builtins as __builtin__

import ctypes
import os
import six
import select
import collections

from .strings import fStr, fStrLen
from .types import fDerivedType

from .utils import *
from .errors import *

from .selector import _selectVar

_TEST_FLAG = os.environ.get("_GFORT2PY_TEST_FLAG") is not None

_allFuncs = {}

Result = collections.namedtuple('Result', 'result args')

class captureStdOut():
    def read_pipe(self,pipe_out):
        def more_data():
            r, _, _ = select.select([pipe_out], [], [], 0)
            return bool(r)
        out = b''
        while more_data():
            out += os.read(pipe_out, 1024)
        return out.decode()
    
    def __enter__(self):
        if _TEST_FLAG:
            self.pipe_out, self.pipe_in = os.pipe()
            self.stdout = os.dup(1)
            os.dup2(self.pipe_in, 1)
    
    def __exit__(self,*args,**kwargs):
        if _TEST_FLAG:
            os.dup2(self.stdout, 1)
            print(self.read_pipe(self.pipe_out))
            os.close(self.pipe_in)
            os.close(self.pipe_out)
            os.close(self.stdout)




class fFunc(object):
    def __init__(self, obj):
        self.__dict__.update(obj)
        self.ctype = ctypes.c_void_p
        self._func = None
            
    def sizeof(self):
        return ctypes.sizeof(self.ctype)
                    
    def in_dll(self, lib):
        self._sub = self.proc['sub']
        
        self._init_args()
        self._init_return()
        self._func = getattr(lib, self.mangled_name)  
        self._func.restype = self._return.ctype
    

    def _set_func(self, func):
        self.proc = func.proc
        self.arg = func.arg
        self._init_args()
        self._init_return() 
        
        self._func = func._func


    def __call__(self, *args):
        
        if self._func is None:
            raise AttributeError("Must point to something first")
        
        if len(args) != len(self.arg) :
            raise TypeError(str(self.name)+" takes "+str(len(self.arg)) + " arguments got "+str(len(args)))
            
            
        args_in = []
        for value, ctype in zip(args, self._args):
            args_in.append(ctype.from_param(value))
        
        # Now handle adding string lengths to end of argument list
        for a in self._args[len(self.arg):]: #self.arg is the list of normal arguments
            for v in args:
                if type(v) is str or type(v) is bytes:
                    args_in.append(a.from_param(v))
        
        # Capture stdout messages
        with captureStdOut() as cs:        
            if len(args_in) > 0:
                ret = self._func(*args_in)
            else:
                ret = self._func()
        dummy_args = {}
        count = 0
        for value,obj in zip(args_in, self._args):
            try:
                dummy_args[obj.name] = obj.from_func(value)
                #dummy_args[obj.name] = value
            except IgnoreReturnError:
                pass
                
        if self._sub:
            ret = 0

        return Result(ret, dummy_args)

    def _init_return(self):
        # make a object usable by _selectVar()
        ret = {}
        ret['var'] = self.proc['ret']
        self._return = self._get_fvar(ret)(ret)
        
    def _init_args(self):
        self._args = []
        extras = []
        for i in self.arg:
            x = self._get_fvar(i)(i)
            
            if isinstance(x,fStr): # Need a string length at the end of the argument list
                extras.append(fStrLen())
            self._args.append(x)
            
        self._args.extend(extras)
            
        
    def _get_fvar(self,var):
        x = _selectVar(var)
        if x is None: # Handle derived types
            if 'dt' in var['var'] and var['var']['dt']:
                x = fDerivedType
            else:
                raise TypeError("Can't match ",var['name'])
        return x



    def from_param(self, value):
        pass






# class fFuncPtr(fFunc):
    # def __init__(self,*args,**kwargs):
        # super(fFuncPtr,self).__init__(*args,**kwargs)
        # self._cfunc = ctypes.CFUNCTYPE(self._res_type,*self._arg_ctypes)
        # # self._ref = self._cfunc.in_dll(self._lib,self.mangled_name)
    
    # # def set_mod(self,func_obj):
        # # if isinstance(func_obj, six.string_types):
            # # # String name of the function
            # # self._ref.contents = self._ctype(self._get_ptr_func(self._mangle_name(self.module,func_obj))).contents
        # # elif isinstance(func_obj,fFunc):
            # # # Passed a fortran function 
            # # addr = ctypes.addressof(self._get_ptr_func(func_obj.mangled_name))
            # # print(addr)
            # # ctypes.memmove(self._ref,addr,ctypes.sizeof(ctypes.c_void_p))
            # # print(self._ref)
        # # elif callable(func_obj):
            # # # Passed a python function
            # # self._ref.contents  =  self._cfunc(func_obj)
        # # else:
            # # raise TypeError("Expecting either a name of function (str), a fFort function, or a python callable")      
        
    # @property
    # def _call(self):
        # # Use cvoidp to check as that is None if we have =>Null()
        # xx = self._cfunc.in_dll(self._lib,self.mangled_name)
        # print("*",ctypes.addressof(xx),xx)
        # c = ctypes.c_void_p.in_dll(self._lib,self.mangled_name)
        # if c.value is None:
            # raise ValueError("Must set pointer first")
        # self._ref = self._cfunc.in_dll(self._lib,self.mangled_name)
        # return self._ref
        
        
        # # if self._func_ptr is not None:
            # # return self._func_ptr
        # # else:
            # # raise AttributeError("Must call set a function to be pointed to")
    
