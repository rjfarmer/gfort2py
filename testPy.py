import ctypes
import pickle
import parseMod
import numpy as np

libname='./test_mod.so'
lib=ctypes.CDLL(libname)

func=getattr(lib,'__tester_MOD_array_in_fixed')

x=np.arange(1,10,dtype=np.int32)

func.argtypes=[ctypes.c_void_p]
func(x.ctypes.data_as(ctypes.c_void_p))

