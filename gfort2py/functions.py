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

from .var import fVar
from .cmplx import fComplex
from .arrays import fExplicitArray, fDummyArray, fAssumedShape, fAssumedSize, fAllocatableArray
from .strings import fStr
from .types import fDerivedType
from .utils import *
from .errors import *


_TEST_FLAG = os.environ.get("_GFORT2PY_TEST_FLAG") is not None

_allFuncs = {}

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

class fFunc(fVar):

    def __init__(self, lib, obj):
        self.__dict__.update(obj)
        self._lib = lib
        self._sub = self.proc['sub']
        try:
            self._call = self._get_ptr_func(self.mangled_name)
        except AttributeError:
            print("Skipping "+self.mangled_name)
            return
            
        self._set_return()
        self._set_arg_ctypes()
        self.save_args=False
        self.args_out = None
        
        self._num_opt = self._count_optionals()
        self._num_args = len(self._arg_vars) - self._num_opt

    def _get_ptr_func(self,name):
        return getattr(self._lib, name)
        

    def _set_arg_ctypes(self):
        self._arg_ctypes = []
        self._arg_vars = []
        
        tmp=[]
        set_args = True
        if len(self.proc['arg_nums'])>0:
            for i in self.arg:
                self._arg_vars.append(self._init_var(i))
                self._arg_vars[-1]._func_arg=True
                
                if 'is_func' in i['var']:
                    # See https://stackoverflow.com/questions/25014191/python-ctypes-function-pointer
                    # Dont set argtypes if we want function ponters
                    set_args = False
                    self._arg_ctypes.append(None)
                else:
                    if 'pointer' in i['var']:
                        pointer=True
                    else:
                        pointer=False                    
                    x,y=self._arg_vars[-1].ctype_def_func(pointer=pointer,intent=i['var']['intent'])
                    self._arg_ctypes.append(x)
                    if y is not None:
                        tmp.append(y)
                    
            if set_args:
                self._call.argtypes = self._arg_ctypes+tmp
            

    def _init_var(self, obj):
        array = None
        if 'array' in obj['var']:
            array = obj['var']['array']
        
        if obj['var']['pytype'] == 'str':
            x = fStr(self._lib, obj)
        elif obj['var']['pytype'] == 'complex':
            x = fComplex(self._lib, obj)
        elif 'dt' in obj['var']:
            x = fDerivedType(self._lib, obj)
        elif array is not None:
            #print(self.name,array)
            if array['atype'] == 'explicit':
                x = fExplicitArray(self._lib, obj)
            elif array['atype'] == 'alloc':
                x = fAllocatableArray(self._lib, obj)
            elif array['atype'] == 'assumed_shape' or array['atype'] == 'pointer':
                x = fAssumedShape(self._lib, obj)
            elif array['atype'] == 'assumed_size':
                x = fAssumedSize(self._lib, obj)
            else:
                raise ValueError("Unknown array: "+str(obj))
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
        for idx, x in enumerate(six.moves.zip_longest(self._arg_vars, args, self._arg_ctypes, self.arg)):
            vout, vin, fctype, a = x
            if a['var']['optional'] and vin is None:
                    #Missing optional arguments 
                    args_in.append(None)            
            elif 'is_func' in a['var']:
                if isinstance(vin, six.string_types):
                    # String name of the function
                    # f_addr = ctypes.addressof(self._get_ptr_func(self._mangle_name(self.module,vin)))
                    args_in.append(self._get_ptr_func(self._mangle_name(self.module,vin)))
                elif isinstance(vin,fFunc):
                    # Passed a fortran function 
                    # f_addr = ctypes.addressof(self._get_ptr_func(vin.mangled_name))
                    # ptr = ctypes.pointer(ctypes.c_int(f_addr))
                    # print(f_addr,ptr)
                    args_in.append(self._get_ptr_func(vin.mangled_name))
                elif isinstance(vin, list):
                    func = vin[0]
                    func_name = vin[1]
                    # Passed a python function
                    print(func_name)
                    self._arg_ctypes[idx] = ctypes.CFUNCTYPE(_allFuncs[func_name]._restype,*_allFuncs[func_name]._arg_ctypes)
                    args_in.append(self._arg_ctypes[idx](func))
                else:
                    raise TypeError("Expecting either a name of function (str) or a fFort function")
            else:
                x,y=vout.py_to_ctype_f(vin)
                if 'pointer' in a['var']:
                    args_in.append(vout.py_to_ctype_p(vin))
                else:
                    args_in.append(x)
                if y is not None:
                    tmp.append(y)

        return args_in + tmp
        
    def ctype_def(self):
        """
        The ctype type of this object
        """
        if '_cached_ctype' not in self.__dict__:
            self._cached_ctype = getattr(ctypes, self.proc['ret']['ctype'])
        
        return self._cached_ctype
    
    def _ctypes_to_return(self,args_out):
    
        r = {}
        self.args_out = {}
        
        if self.save_args:
            # Save arguments inside this object
            for i,j in zip(self._arg_vars,args_out):
                if 'out' in i.var['intent'] or i.var['intent']=='na': 
                    r[i.name]=''
                    if hasattr(j,'contents'):
                        self.args_out[i.name]=j.contents
                    else:
                        self.args_out[i.name]=j
        else:
            # Copy arguments into a dict for returning
            for i,j in zip(self._arg_vars,args_out):
                if 'out' in i.var['intent'] or i.var['intent']=='na':
                    if 'is_func' in i.var:
                        r[i.name] = None
                    elif hasattr(j,'contents'):
                        r[i.name]=i.ctype_to_py_f(j.contents)
                    else:
                        r[i.name]=i.ctype_to_py_f(j)

        return r
    
    def __call__(self, *args):
        if len(args) < self._num_args:
            raise TypeError(str(self.name)+" takes atleast "+str(self._num_args) + " arguments got "+str(len(args)))
        if len(args) > len(self.arg):
            raise TypeError(str(self.name)+" takes atmost "+str(len(self.arg)) + " arguments got "+str(len(args)))
            
        args_in = self._args_to_ctypes(args)
        
        # Capture stdout messages
        with captureStdOut() as cs:        
            if len(args_in) > 0:
                res = self._call(*args_in)
            else:
                res = self._call()

        if self._sub:
            return self._ctypes_to_return(args_in)
        else:
            return self.returnPytype()(res)
            
    def saveArgs(self,v=False):
        """ 
        Instead of copying arguments back we save them
        inside the func object so we dont need to copy them
        """
        self.save_args=False
            
    def returnPytype(self):
        if '_cached_pytype' not in self.__dict__:
            self._cached_pytype = getattr(__builtin__, self.proc['ret']['pytype'])
        
        return self._cached_pytype
            
    def __str__(self):
        return str("Function: " + self.name)

    def __repr__(self):
        return self.__str__()

    @property
    def __doc__(self):
        s = "Function: " + self.name + "("
        if len(self._arg_vars) > 0:
            s = s + ",".join([i._pname() for i in self._arg_vars])
        else:
            s = s + "None"
        s = s + ")" + os.linesep+' '
        s = s + "Args In: " + \
            ", ".join([i._pname()
                      for i in self._arg_vars if 'in' in i.var['intent']]) + ' '+os.linesep+' '
        s = s + "Args Out: " + \
            ", ".join([i._pname()
                      for i in self._arg_vars if 'out' in i.var['intent']]) + ' '+os.linesep+' '
        s = s + "Returns: "
        if self.sub:
            s = s + "None"
        else:
            s = s + str(self.pytype)
        s = s + os.linesep+' '
        return s

    def __bool__(self):
        return True
        
    def __len__(self):
        return 1
        
    def _count_optionals(self):
        count = 0
        for a in self.arg:
            if 'optional' in a['var']:
                if a['var']['optional']:
                    count = count + 1 
        return count
        
    def __dir__(self):
        return ['saveArgs']
        
    def load(self):
        # empty function call it to trigger the load in gfort2py without do anything
        pass
    
