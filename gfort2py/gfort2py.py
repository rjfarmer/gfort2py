# SPDX-License-Identifier: GPL-2.0+
import ctypes
import numpy as np
import collections
import select
import os

from . import parseMod as pm


_TEST_FLAG = os.environ.get("_GFORT2PY_TEST_FLAG") is not None


class _captureStdOut:
    def read_pipe(self, pipe_out):
        def more_data():
            r, _, _ = select.select([pipe_out], [], [], 0)
            return bool(r)

        out = b""
        while more_data():
            out += os.read(pipe_out, 1024)
        return out.decode()

    def __enter__(self):
        if _TEST_FLAG:
            self.pipe_out, self.pipe_in = os.pipe()
            self.stdout = os.dup(1)
            os.dup2(self.pipe_in, 1)

    def __exit__(self, *args, **kwargs):
        if _TEST_FLAG:
            os.dup2(self.stdout, 1)
            print(self.read_pipe(self.pipe_out))
            os.close(self.pipe_in)
            os.close(self.pipe_out)
            os.close(self.stdout)

class fObject:
    def __eq__(self, other):
        return self.value.__eq__(other)

    def __ne__(self, other):
        return self.value.__neq__(other)

    def __lt__(self, other):
        return self.value.__lt__(other)

    def __le__(self, other):
        return self.value.__le__(other)

    def __gt__(self, other):
        return self.value.__gt__(other)

    def __ge__(self, other):
        return self.value.__ge__(other)

    def __add__(self, other):
        return self.value.__add__(other)

    def __sub__(self, other):
        return self.value.__sub__(other)

    def __matmul__(self, other):
        return self.value.__matmul__(other)

    def __truediv__(self, other):
        return self.value.__truediv__(other)

    def __floordiv__(self, other):
        return self.value.__floordiv__(other)

    def __mod__(self, other):
        return self.value.__mod__(other)

    def __divmod__(self, other):
        return self.value.__divmod__(other)

    def __pow__(self, other, modulo=None):
        return self.value.__pow__(other, modulo)

    def __lshift__(self, other):
        return self.value.__lshift__(other)

    def __rshift__(self, other):
        return self.value.__rshift__(other)

    def __and__(self, other):
        return self.value.__and__(other)

    def __xor__(self, other):
        return self.value.__xor__(other)

    def __or__(self, other):
        return self.value.__or__(other)

    def __radd__(self, other):
        return self.value.__radd__(other)

    def __rsub__(self, other):
        return self.value.__rsub__(other)

    def __rmatmul__(self, other):
        return self.value.__rmatmul__(other)

    def __rtruediv__(self, other):
        return self.value.__rtruediv__(other)

    def __rfloordiv__(self, other):
        return self.value.__rfloordiv__(other)

    def __rmod__(self, other):
        return self.value.__rmod__(other)

    def __rdivmod__(self, other):
        return self.value.__rdivmod__(other)

    def __rlshift__(self, other):
        return self.value.__rlshift__(other)

    def __rrshift__(self, other):
        return self.value.__rrshift__(other)

    def __rand__(self, other):
        return self.value.__rand__(other)

    def __rxor__(self, other):
        return self.value.__rxor__(other)

    def __ror__(self, other):
        return self.value.__ror__(other)

    def __neg__(self):
        return self.value.__neg__()

    def __pos__(self):
        return self.value.__pos__()

    def __abs__(self):
        return self.value.__abs__()

    def __invert__(self):
        return self.value.__invert__()

    def __complex__(self):
        return self.value.__complex__()

    def __int__(self):
        return self.value.__int__()

    def __float__(self):
        return self.value.__float__()

    def __round__(self, ndigits=None):
        return self.value.__round__(ndigits)

    def __trunc__(self):
        return self.value.__trunc__()

    def __floor__(self):
        return self.value.__floor__()

    def __ceil__(self):
        return self.value.__ceil__()

    def __len__(self):
        return self.value.__len__()

    def __index__(self):
        return int(self.value)

class fParam(fObject):
    def __init__(self, lib, allobjs, key):
        self._allobjs = allobjs
        self._object = self._allobjs[key]
        self._lib = lib

    @property
    def value(self):
        return self._object.sym.parameter.value

    @value.setter
    def value(self, value):
        raise AttributeError("Parameter can't be altered")



class fVar_t:
    def __init__(self, obj):
        self._object = obj

    def type(self):
        return self._object.sym.ts.type

    def flavor(self):
        return self._object.sym.flavor

    def kind(self):
        return self._object.sym.ts.kind

    def from_param(self, value):
        t = self.type()
        k = int(self.kind())

        if self.is_optional and value is None:
            return None

        if t == 'INTEGER':
            return self.ctype(value)(value)
        elif t == 'REAL':
            return self.ctype(value)(value)
        elif t == 'LOGICAL':
            if value:
                return self.ctype(value)(1)
            else:
                return self.ctype(value)(0)
        elif t == 'CHARACTER':
            try:
                strlen = self._object.sym.ts.charlen.value # We know the string length at compile time
            except AttributeError:
                strlen = len(value) # We do not know the length of the string at compile time
            if hasattr(value, "encode"):
                value = value.encode()

            if len(value) > strlen:
                value = value[:strlen] 
            else:
                value = value + b' '*(strlen-len(value))

            self._buf = bytearray(value) # Need to keep hold of the reference
            
            return self.ctype(value).from_buffer(self._buf)
        elif t == 'COMPLEX':
            return self.ctype()(value.real, value.imag)


        raise NotImplementedError(f'Object of type {t} and kind {k} not supported yet')

    def is_pointer(self):
        return 'POINTER' in self._object.sym.attr.attributes

    def is_value(self):
        return 'VALUE' in self._object.sym.attr.attributes

    def is_optional(self):
        return 'OPTIONAL' in self._object.sym.attr.attributes

    def is_char(self):
        return self.type() == 'CHARACTER'

    def needs_len(self, *args):
        # Only needed for things that need an extra function argument for thier length 
        if self.is_char():
            try:
                self._object.sym.ts.charlen.value # We know the string length at compile time
                return False
            except AttributeError:
                return True # We do not know the length of the string at compile time
        return False

    def clen(self, *args):
        if self.is_char():
            try:
                return ctypes.c_int64(self._object.sym.ts.charlen.value) # We know the string length at compile time
            except AttributeError:
                return ctypes.c_int64(len(args[0])) # We do not know the length of the string at compile time



    @property
    def ctype(self):
        t = self.type()
        k = int(self.kind())

        if t == 'INTEGER':
            if k == 4:
                def callback(*args):
                    return ctypes.c_int32
                return callback
            elif k == 8:
                def callback(*args):
                    return ctypes.c_int64
                return callback
        elif t == 'REAL':
            if k == 4:
                def callback(*args):
                    return ctypes.c_float
                return callback
            elif k == 8:
                def callback(*args):
                    return ctypes.c_double
                return callback
        elif t == 'LOGICAL':
            def callback(*args):
                return ctypes.c_int32
            return callback
        elif t == 'CHARACTER':
            try:
                strlen = self._object.sym.ts.charlen.value # We know the string length at compile time
                def callback(*args):
                    return ctypes.c_char * strlen
                return callback
            except AttributeError:
                def callback(value, *args): # We de not know the string length at compile time
                    return ctypes.c_char * len(value)
                return callback
        elif t == 'COMPLEX':
            if k == 4:
                def callback(*args):
                    class complex(ctypes.Structure):
                        _fields_ = [("real", ctypes.c_float), ("imag", ctypes.c_float)]
                    return complex
                return callback
            elif k == 8:
                def callback(*args):
                    class complex(ctypes.Structure):
                        _fields_ = [("real", ctypes.c_double), ("imag", ctypes.c_double)]
                    return complex
                return callback
            elif k == 16:
                def callback(*args):
                    class complex(ctypes.Structure):
                        _fields_ = [("real", ctypes.c_longdouble), ("imag", ctypes.c_longdouble)]
                    return complex
                return callback

        raise NotImplementedError(f'Object of type {t} and kind {k} not supported yet')


    def from_ctype(self, value):
        t = self.type()
        k = int(self.kind())

        if value is None:
            return None

        x = value
        if hasattr(value, "contents"):
            if hasattr(value.contents, "contents"):
                x = value.contents.contents
            else:
                x = value.contents

        if t == 'COMPLEX':
            return complex(x.real, x.imag)

        if hasattr(x,'value'):
            if t == 'INTEGER':
                return x.value
            elif t == 'REAL':
                return x.value
            elif t == 'LOGICAL':
                return x.value == 1
            elif t == 'CHARACTER':
                return "".join([i.decode() for i in x])
            raise NotImplementedError(f'Object of type {t} and kind {k} not supported yet')
        else:
            return x


    @property
    def name(self):
        return self._object.head.name

    @property
    def __doc__(self):
        t = self.type()
        k = self.kind()
        return f"{self._object.head.name}={self.typekind}"

    @property
    def typekind(self):
        t = self.type()
        k = self.kind()
        if t == 'INTEGER' or t == 'REAL':
            return f"{t}(KIND={k})"
        elif t == 'LOGICAL':
            return f"{t}"
        elif t == 'CHARACTER':
            try:
                strlen = self._object.sym.ts.charlen.value # We know the string length at compile time
                return f"{t}(LEN={strlen})"
            except AttributeError:
                return f"{t}(LEN=:)"


class fVar(fObject):
    def __init__(self, lib, allobjs, key):
        self._allobjs = allobjs
        self._object = self._allobjs[key]
        self._lib = lib

        self._value = fVar_t(self._object)

    def from_param(self, value):
        return self._value.from_param(value)

    @property
    def value(self):
        return self._value.from_ctype(self.in_dll(self._lib))

    @value.setter
    def value(self, value):
        ct = self.in_dll(self._lib)

        if isinstance(ct, ctypes.Structure):
            for k in ct.__dir__():
                if not k.startswith('_') and hasattr(value,k):
                    setattr(ct,k,getattr(value,k))
        else:
            ct.value = self.from_param(value).value

    @property
    def mangled_name(self):
        return self._object.head.mn_name

    def __repr__(self):
        return str(self.value)

    def __str__(self):
        return str(self.value)

    def in_dll(self, lib):
        return self._value.ctype().in_dll(lib, self.mangled_name)

    @property
    def module(self):
        return self._object.head.module

    @property
    def __doc__(self):
        return f"{self._value.type()}(KIND={self._value.kind()}) " \
               f"MODULE={self.module}.mod"

class fProc:
    Result = collections.namedtuple('Result', ["res", "args"])


    def __init__(self, lib, allobjs, key):
        self._allobjs = allobjs
        self._object = self._allobjs[key]
        self._lib = lib

        self._func = getattr(lib, self.mangled_name)

        self.fargs = self._object.sym.formal_arg
        self.symref = self._object.sym.sym_ref.ref

    @property
    def mangled_name(self):
        return self._object.head.mn_name

    def __repr__(self):
        return str(self.value)

    def __str__(self):
        return str(self.value)

    def in_dll(self, lib):
        return self._func

    @property
    def module(self):
        return self._object.head.module


    @property
    def name(self):
        return self._object.head.name

    @property
    def __doc__(self):
        return f"Procedure"

    def __call__(self, *args, **kwargs):

        self._set_return()        

        func_args = self._convert_args(*args, **kwargs)

        with _captureStdOut() as cs:
            if func_args is not None:
                res = self._func(*func_args)
            else:
                res = self._func()

        return self._convert_result(res, func_args)

    def _set_return(self):
        if self.symref == 0:
            self._func.restype = None # Subroutine
        else:
            fvar = fVar_t(self._allobjs[self.symref])

            if fvar.is_char(): #Return a character is done as a character + len at start of arg list
                self._func.restype = None
            else:
                self._func.restype = fvar.ctype()


    def _convert_args(self, *args, **kwargs):

        res_start = []
        res = []
        res_end = []


        if self.symref != 0:
            fvar = fVar_t(self._allobjs[self.symref])
            if fvar.is_char():
                l = fvar.clen()
                res_start.append(fvar.from_param(' '*l.value))
                res_start.append(l)

        count = 0
        for fval in self.fargs:
            var = fVar_t(self._allobjs[fval.ref])

            try:
                x = kwargs[var.name]
            except KeyError:
                if count <= len(args):
                    x = args[count]
                    count = count+1
                else:
                    raise TypeError('Not enough arguments passed')

            if x is None and not var.is_optional():
                raise ValueError(f'Got None for {var.name}')

            if x is not None:
                z = var.from_param(x)

                if var.is_value():
                    res.append(z)
                elif var.is_pointer():
                    res.append(ctypes.pointer(ctypes.pointer(z)))
                else:
                    res.append(ctypes.pointer(z))

                if var.needs_len(x):
                    res_end.append(var.clen(x))

            else:
                res.append(None)
                if var.needs_len(x):
                    res_end.append(None)

        return res_start + res + res_end

    def _convert_result(self, result, args):
        res = {}

        if self.symref != 0:
            fvar = fVar_t(self._allobjs[self.symref])
            if fvar.is_char():
                result = args[0]
                _ = args.pop(0)
                _ = args.pop(0) # Twice to pop first and second value

        if len(self.fargs):
            for ptr,fval in zip(args,self.fargs):
                res[self._allobjs[fval.ref].head.name] = fVar_t(self._allobjs[fval.ref]).from_ctype(ptr)


        if self.symref != 0:
            result = fVar_t(self._allobjs[self.symref]).from_ctype(result)

        return self.Result(result, res)

    def __repr__(self):
        return self.__doc__

    @property
    def __doc__(self):

        if self.symref == 0:
            ftype = f"subroutine {self.name}"
        else:
            fv = fVar_t(self._allobjs[self.symref]).typekind
            ftype = f"{fv} function {self.name}"

        args = []
        for fval in self.fargs:
            args.append(fVar_t(self._allobjs[fval.ref]).__doc__)

        return ftype + '(' + ', '.join(args) + ')'

class fFort:
    _initialised = False

    def __init__(self, libname, mod_file):
        self._lib = ctypes.CDLL(libname)
        self._mod_file = mod_file 
        self._module = pm.module(self._mod_file)

        self._initialised = True

    def keys(self):
        return self._module.keys()

    def __contains__(self, key):
        return key in self._module.keys()

    def __dir__(self):
        return self.keys()

    def __getattr__(self, key):
        if key in self.__dict__:
            return self.__dict__[key]

        if '_initialised' in self.__dict__:
            if self._initialised:
                if key not in self.keys():
                    raise AttributeError(f"{self._mod_file}  has no attribute {key}")

            flavor = self._module[key].sym.attr.flavor
            if flavor == 'VARIABLE':
                return fVar(self._lib, self._module, key)
            elif flavor == 'PROCEDURE':
                return fProc(self._lib, self._module, key)
            elif flavor == 'PARAMETER':
                return fParam(self._lib, self._module, key)
            else:
                raise NotImplementedError(f"Object type {flavor} not implemented yet")


    def __setattr__(self, key, value):
        if '_initialised' in self.__dict__:
            if self._initialised:
                if key not in self:
                    raise AttributeError(f"{self._mod_file}  has no attribute {key}")

                flavor = self._module[key].sym.attr.flavor
                if flavor == 'VARIABLE':
                    f = fVar(self._lib, self._module, key)
                    f.value = value
                    return
                elif flavor == 'PARAMETER':
                    raise AttributeError('Can not alter a parameter')

                else:
                    raise NotImplementedError(f"Object type {flavor} not implemented yet")

        self.__dict__[key] = value

    @property
    def __doc__(self):
        return f"MODULE={self._module.filename}"

    def __str__(self):
        return f"{self._module.filename}"
