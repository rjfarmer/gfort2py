# SPDX-License-Identifier: GPL-2.0+
from __future__ import print_function

def find_key_val(list_dicts, key, value):
    v = value.lower()
    for idx, i in enumerate(list_dicts):
        if i[key].lower() == v:
            return idx


def _makefuncs(listfuncs):
    for i in listfuncs: 
        print('    def '+i+'(self, *args, **kwargs):') 
        print('        return getattr(self.get(),"'+i+'")(*args,**kwargs)')
        print()


class fParent(object):
    def get(self):
        return None

    def set(self, value):
        pass

    def __add__(self, other):
        return getattr(self.get(), '__add__')(other)

    def __sub__(self, other):
        return getattr(self.get(), '__sub__')(other)

    def __mul__(self, other):
        return getattr(self.get(), '__mul__')(other)

    def __matmul__(self,other):
        return getattr(self.get(), '__matmul__')(other)

    def __truediv__(self, other):
        return getattr(self.get(), '__truediv__')(other)

    def __floordiv__(self,other):
        return getattr(self.get(), '__floordiv__')(other)

    def __pow__(self, other, modulo=None):
        return getattr(self.get(), '__pow__')(other,modulo)

    def __mod__(self,other):
        return getattr(self.get(), '__mod__')(other)

    def __lshift__(self,other):
        return getattr(self.get(), '__lshift__')(other)

    def __rshift__(self,other):
        return getattr(self.get(), '__rshift__')(other)

    def __and__(self,other):
        return getattr(self.get(), '__and__')(other)

    def __xor__(self,other):
        return getattr(self.get(), '__xor__')(other)

    def __or__(self,other):
        return getattr(self.get(), '__or__')(other)

    def __radd__(self, other):
        return getattr(self.get(), '__radd__')(other)

    def __rsub__(self, other):
        return getattr(self.get(), '__rsub__')(other)

    def __rmul__(self, other):
        return getattr(self.get(), '__rmul__')(other)

    def __rmatmul__(self,other):
        return getattr(self.get(), '__rmatmul__')(other)

    def __rtruediv__(self, other):
        return getattr(self.get(), '__rtruediv__')(other)

    def __rfloordiv__(self,other):
        return getattr(self.get(), '__rfloordiv__')(other)

    def __rpow__(self, other):
        return getattr(self.get(), '__rpow__')(other)

    def __rmod__(self,other):
        return getattr(self.get(), '__rmod__')(other)

    def __rlshift__(self,other):
        return getattr(self.get(), '__rlshift__')(other)

    def __rrshift__(self,other):
        return getattr(self.get(), '__rrshift__')(other)

    def __rand__(self,other):
        return getattr(self.get(), '__rand__')(other)

    def __rxor__(self,other):
        return getattr(self.get(), '__rxor__')(other)

    def __ror__(self,other):
        return getattr(self.get(), '__ror__')(other)

    def __iadd__(self, other):
        self.set_mod(self.get() + other)
        return self.get()

    def __isub__(self, other):
        self.set(self.get() - other)
        return self.get()

    def __imul__(self, other):
        self.set(self.get() * other)
        return self.get()

    def __itruediv__(self, other):
        self.set(self.get() / other)
        return self.get()

    def __ipow__(self, other, modulo=None):
        x = self.get()**other
        if modulo:
            x = x % modulo
        self.set(x)
        return self.get()

    def __eq__(self, other):
        return getattr(self.get(), '__eq__')(other)

    def __neq__(self, other):
        return getattr(self.get(), '__new__')(other)

    def __lt__(self, other):
        return getattr(self.get(), '__lt__')(other)

    def __le__(self, other):
        return getattr(self.get(), '__le__')(other)

    def __gt__(self, other):
        return getattr(self.get(), '__gt__')(other)

    def __ge__(self, other):
        return getattr(self.get(), '__ge__')(other)

    def __format__(self, other):
        return getattr(self.get(), '__format__')(other)

    def __bytes__(self):
        return getattr(self.get(), '__bytes__')()

    def __bool__(self):
        return getattr(self.get(), '__bool__')()

    def __len__(self):
        return getattr(self.get(), '__len__')()

    def __length_hint__(self):
        return getattr(self.get(), '__length_hint__')()

    def __dir__(self):
        return list(self.__dict__.keys()) + list(dir(self.get()))

    def __int__(self):
        return getattr(self.get(), '__int__')()

    def __float__(self):
        return getattr(self.get(), '__float__')()

    def __complex__(self):
        return getattr(self.get(), '__complex__')()

    def __neg__(self):
        return getattr(self.get(), '__neg__')()

    def __index__(self):
        return getattr(self.get(), '__index__')()

    def __round__(self, ndigits=None):
        return round(self.get(),ndigits)

    def __trunc__(self):
        return getattr(self.get(), '__trunc__')()

    def __ceil__(self):
        return getattr(self.get(), '__ceil__')()

    def __floor__(self):
        return getattr(self.get(), '__floor__')()

    def __neg__(self):
        return getattr(self.get(), '__neg__')()

    def __pos__(self):
        return getattr(self.get(), '__pos__')()

    def __abs__(self):
        return getattr(self.get(), '__abs__')()
 
    def __invert__(self):
        return getattr(self.get(), '__invert__')()
        
    def __str__(self):
        return getattr(self.get(), '__str__')()
        
    def __repr__(self):
        return getattr(self.get(), '__repr__')()
        
        

class fParentArray(fParent):
    def all(self, *args, **kwargs):
        return getattr(self.get(),"all")(*args,**kwargs)

    def any(self, *args, **kwargs):
        return getattr(self.get(),"any")(*args,**kwargs)

    def argmax(self, *args, **kwargs):
        return getattr(self.get(),"argmax")(*args,**kwargs)

    def argmin(self, *args, **kwargs):
        return getattr(self.get(),"argmin")(*args,**kwargs)

    def argpartition(self, *args, **kwargs):
        return getattr(self.get(),"argpartition")(*args,**kwargs)

    def argsort(self, *args, **kwargs):
        return getattr(self.get(),"argsort")(*args,**kwargs)

    def astype(self, *args, **kwargs):
        return getattr(self.get(),"astype")(*args,**kwargs)

    def base(self, *args, **kwargs):
        return getattr(self.get(),"base")(*args,**kwargs)

    def byteswap(self, *args, **kwargs):
        return getattr(self.get(),"byteswap")(*args,**kwargs)

    def choose(self, *args, **kwargs):
        return getattr(self.get(),"choose")(*args,**kwargs)

    def clip(self, *args, **kwargs):
        return getattr(self.get(),"clip")(*args,**kwargs)

    def compress(self, *args, **kwargs):
        return getattr(self.get(),"compress")(*args,**kwargs)

    def conj(self, *args, **kwargs):
        return getattr(self.get(),"conj")(*args,**kwargs)

    def conjugate(self, *args, **kwargs):
        return getattr(self.get(),"conjugate")(*args,**kwargs)

    def copy(self, *args, **kwargs):
        return getattr(self.get(),"copy")(*args,**kwargs)

    def ctypes(self, *args, **kwargs):
        return getattr(self.get(),"ctypes")(*args,**kwargs)

    def cumprod(self, *args, **kwargs):
        return getattr(self.get(),"cumprod")(*args,**kwargs)

    def cumsum(self, *args, **kwargs):
        return getattr(self.get(),"cumsum")(*args,**kwargs)

    def data(self, *args, **kwargs):
        return getattr(self.get(),"data")(*args,**kwargs)

    def diagonal(self, *args, **kwargs):
        return getattr(self.get(),"diagonal")(*args,**kwargs)

    def dot(self, *args, **kwargs):
        return getattr(self.get(),"dot")(*args,**kwargs)

    def dtype(self, *args, **kwargs):
        return getattr(self.get(),"dtype")(*args,**kwargs)

    def dump(self, *args, **kwargs):
        return getattr(self.get(),"dump")(*args,**kwargs)

    def dumps(self, *args, **kwargs):
        return getattr(self.get(),"dumps")(*args,**kwargs)

    def fill(self, *args, **kwargs):
        return getattr(self.get(),"fill")(*args,**kwargs)

    def flags(self, *args, **kwargs):
        return getattr(self.get(),"flags")(*args,**kwargs)

    def flat(self, *args, **kwargs):
        return getattr(self.get(),"flat")(*args,**kwargs)

    def flatten(self, *args, **kwargs):
        return getattr(self.get(),"flatten")(*args,**kwargs)

    def getfield(self, *args, **kwargs):
        return getattr(self.get(),"getfield")(*args,**kwargs)

    def imag(self, *args, **kwargs):
        return getattr(self.get(),"imag")(*args,**kwargs)

    def item(self, *args, **kwargs):
        return getattr(self.get(),"item")(*args,**kwargs)

    def itemset(self, *args, **kwargs):
        return getattr(self.get(),"itemset")(*args,**kwargs)

    def itemsize(self, *args, **kwargs):
        return getattr(self.get(),"itemsize")(*args,**kwargs)

    def max(self, *args, **kwargs):
        return getattr(self.get(),"max")(*args,**kwargs)

    def mean(self, *args, **kwargs):
        return getattr(self.get(),"mean")(*args,**kwargs)

    def min(self, *args, **kwargs):
        return getattr(self.get(),"min")(*args,**kwargs)

    def nbytes(self, *args, **kwargs):
        return getattr(self.get(),"nbytes")(*args,**kwargs)

    def newbyteorder(self, *args, **kwargs):
        return getattr(self.get(),"newbyteorder")(*args,**kwargs)

    def nonzero(self, *args, **kwargs):
        return getattr(self.get(),"nonzero")(*args,**kwargs)

    def partition(self, *args, **kwargs):
        return getattr(self.get(),"partition")(*args,**kwargs)

    def prod(self, *args, **kwargs):
        return getattr(self.get(),"prod")(*args,**kwargs)

    def ptp(self, *args, **kwargs):
        return getattr(self.get(),"ptp")(*args,**kwargs)

    def put(self, *args, **kwargs):
        return getattr(self.get(),"put")(*args,**kwargs)

    def ravel(self, *args, **kwargs):
        return getattr(self.get(),"ravel")(*args,**kwargs)

    def real(self, *args, **kwargs):
        return getattr(self.get(),"real")(*args,**kwargs)

    def repeat(self, *args, **kwargs):
        return getattr(self.get(),"repeat")(*args,**kwargs)

    def reshape(self, *args, **kwargs):
        return getattr(self.get(),"reshape")(*args,**kwargs)

    def resize(self, *args, **kwargs):
        return getattr(self.get(),"resize")(*args,**kwargs)

    def round(self, *args, **kwargs):
        return getattr(self.get(),"round")(*args,**kwargs)

    def searchsorted(self, *args, **kwargs):
        return getattr(self.get(),"searchsorted")(*args,**kwargs)

    def setfield(self, *args, **kwargs):
        return getattr(self.get(),"setfield")(*args,**kwargs)

    def setflags(self, *args, **kwargs):
        return getattr(self.get(),"setflags")(*args,**kwargs)

    def shape(self, *args, **kwargs):
        return getattr(self.get(),"shape")(*args,**kwargs)

    def size(self, *args, **kwargs):
        return getattr(self.get(),"size")(*args,**kwargs)

    def sort(self, *args, **kwargs):
        return getattr(self.get(),"sort")(*args,**kwargs)

    def squeeze(self, *args, **kwargs):
        return getattr(self.get(),"squeeze")(*args,**kwargs)

    def std(self, *args, **kwargs):
        return getattr(self.get(),"std")(*args,**kwargs)

    def strides(self, *args, **kwargs):
        return getattr(self.get(),"strides")(*args,**kwargs)

    def sum(self, *args, **kwargs):
        return getattr(self.get(),"sum")(*args,**kwargs)

    def swapaxes(self, *args, **kwargs):
        return getattr(self.get(),"swapaxes")(*args,**kwargs)

    def take(self, *args, **kwargs):
        return getattr(self.get(),"take")(*args,**kwargs)

    def tobytes(self, *args, **kwargs):
        return getattr(self.get(),"tobytes")(*args,**kwargs)

    def tofile(self, *args, **kwargs):
        return getattr(self.get(),"tofile")(*args,**kwargs)

    def tolist(self, *args, **kwargs):
        return getattr(self.get(),"tolist")(*args,**kwargs)

    def tostring(self, *args, **kwargs):
        return getattr(self.get(),"tostring")(*args,**kwargs)

    def trace(self, *args, **kwargs):
        return getattr(self.get(),"trace")(*args,**kwargs)

    def transpose(self, *args, **kwargs):
        return getattr(self.get(),"transpose")(*args,**kwargs)

    def var(self, *args, **kwargs):
        return getattr(self.get(),"var")(*args,**kwargs)

    def view(self, *args, **kwargs):
        return getattr(self.get(),"view")(*args,**kwargs)
    
 
