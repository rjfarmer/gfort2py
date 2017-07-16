import ctypes
import os
from .var import fVar
from .cmplx import fComplex
from .arrays import fExplicitArray
from .strings import fStr
from gfort2py.types import fDerivedType, fDerivedTypeDesc

from .utils import *


class fFunc(fVar):

    def __init__(self, lib, obj,TEST_FLAG=False):
        self.__dict__.update(obj)
        self._lib = lib
        self._call = getattr(self._lib, self.mangled_name)
        self._set_return()
        self._set_arg_ctypes()
        self.TEST_FLAG=TEST_FLAG

    def _set_arg_ctypes(self):
        self._arg_ctypes = []
        self._arg_vars = []
        for i in self.args:
            self._arg_vars.append(self._init_var(i))
            self._arg_ctypes.append(self._arg_vars[-1].ctype_def_func())
        self._call.argtypes = self._arg_ctypes

    def _init_var(self, obj):
        if obj['pytype'] == 'str':
            x = fStr(self._lib, obj)
        elif obj['cmplx']:
            x = fComplex(self._lib, obj)
        elif obj['dt']:
            x = fDerivedType(self._lib, obj)
        elif obj['array']:
            x = fExplicitArray(self._lib, obj)
        else:
            x = fVar(self._lib, obj)

        return x

    def _set_return(self):
        self.sub = False
        if self.pytype == 'void':
            self.sub = True

        if not self.sub:
            self._restype = self.ctype_def()
            self._call.restype = self._restype

    def __call__(self, *args):
        args_in = [i.ctype_def()(j) for i, j in zip(self._arg_vars, args)]
        
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

        return res

    def __str__(self):
        return str("Function: " + self.name)

    def __repr__(self):
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
