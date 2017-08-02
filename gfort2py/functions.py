from __future__ import print_function
try:
    import __builtin__
except ImportError:
    import builtins as __builtin__

import ctypes
import os
from .var import fVar
from .cmplx import fComplex
from .arrays import fExplicitArray, fDummyArray, fAssumedShape, fAssumedSize, fAllocatableArray
from .strings import fStr
from .types import fDerivedType

from .utils import *


class fFunc(fVar):

    def __init__(self, lib, obj,TEST_FLAG=False):
        self.__dict__.update(obj)
        self._lib = lib
        self._sub = self.proc['sub']
        self._call = getattr(self._lib, self.mangled_name)
        self._set_return()
        self._set_arg_ctypes()
        self.TEST_FLAG=TEST_FLAG

    def _set_arg_ctypes(self):
        self._arg_ctypes = []
        self._arg_vars = []
        
        tmp=[]
        if len(self.proc['arg_nums'])>0:
            for i in self.arg:
                self._arg_vars.append(self._init_var(i))
                self._arg_vars[-1]._func_arg=True
                
                x,y=self._arg_vars[-1].ctype_def_func()
                self._arg_ctypes.append(x)
                if y is not None:
                    tmp.append(y)
            self._call.argtypes = self._arg_ctypes+tmp

    def _init_var(self, obj):
        if obj['var']['pytype'] == 'str':
            x = fStr(self._lib, obj)
        elif obj['var']['pytype'] == 'complex':
            x = fComplex(self._lib, obj)
        elif 'dt' in obj['var']:
            x = fDerivedType(self._lib, obj)
        elif 'array' in obj['var']:
            if obj['var']['array']['atype'] == 'explicit':
                x = fExplicitArray(self._lib, obj,self.TEST_FLAG)
            elif obj['var']['array']['atype'] == 'alloc':
                x = fAllocatableArray(self._lib, obj, self.TEST_FLAG)
            elif obj['var']['array']['atype'] == 'assumed_shape':
                x = fAssumedShape(self._lib, obj, self.TEST_FLAG)
            elif obj['var']['array']['atype'] == 'assumed_size':
                x = fAssumedSize(self._lib, obj, self.TEST_FLAG)
            else:
                print("Unknown: "+str(obj))
                raise ValueError
        else:
            x = fVar(self._lib, obj)

        x._func_arg=True

        return x

    def _set_return(self):
        if not self._sub:
            self._restype = self.ctype_def()
            self._call.restype = self._restype
            
    def _args_to_ctypes(self,args):
        tmp = []
        args_in = []
        #if type(args) is not list: args = [ args ]
        #print("args ",args)
        #print("argtypes",self._call.argtypes)
        for i, j in  zip(self._arg_vars, args):
            #print("j ",type(j),j)
            x,y=i.py_to_ctype_f(j)
            args_in.append(x)
            if y is not None:
                tmp.append(y)
        return args_in + tmp
        
    def ctype_def(self):
        """
        The ctype type of this object
        """
        return getattr(ctypes, self.proc['ret']['ctype'])
    
    def _ctypes_to_return(self,args_out):
        r={}
        for i,j in zip(self._arg_vars,args_out):
            if 'out' in i.var['intent'] or i.var['intent']=='':
                r[i.name]=i.ctype_to_py_f(j)
        return r
    
    def __call__(self, *args):
        #print("call ",args)
        args_in = self._args_to_ctypes(args)
        # Capture stdout messages
        # Cant call python print() untill after the read_pipe call
        if self.TEST_FLAG:
            pipe_out, pipe_in = os.pipe()
            stdout = os.dup(1)
            os.dup2(pipe_in, 1)
        if len(args_in) > 0:
            res = self._call(*args_in)
        else:
            res = self._call()
        if self.TEST_FLAG:
            # Print stdout
            os.dup2(stdout, 1)
            print(read_pipe(pipe_out))
        # Python print available now
        
        if self._sub:
            return self._ctypes_to_return(args_in)
        else:
            return self.returnPytype()(res)
            
    def returnPytype(self):
        return getattr(__builtin__, self.proc['ret']['pytype'])
            
    def __str__(self):
        return str("Function: " + self.name)

    def __repr__(self):
        return self.__str__()

    def __doc__(self):
        s = "Function: " + self.name + "("
        if len(self.args) > 0:
            s = s + ",".join([i._pname() for i in self._arg_vars])
        else:
            s = s + "None"
        s = s + ")" + os.linesep
        s = s + "Args In: " + \
            ",".join([i._pname()
                      for i in self._arg_vars if 'in' in i.intent]) + os.linesep
        s = s + "Args Out: " + \
            ",".join([i._pname()
                      for i in self._arg_vars if 'out' in i.intent]) + os.linesep
        s = s + "Returns: "
        if self.sub:
            s = s + "None"
        else:
            s = s + str(self.pytype)
        s = s + os.linesep
        return s
