# cython: language_level=3

cimport numpy as np
import numpy as np

np.import_array()

cdef extern from "numpy/ndarraytypes.h":
    void PyArray_CLEARFLAGS(np.ndarray arr, int flags)
    void PyArray_ENABLEFLAGS(np.ndarray arr, int flags)

cpdef remove_ownership(np.ndarray arr):   
    PyArray_CLEARFLAGS(arr, np.NPY_ARRAY_OWNDATA)

cpdef give_ownership(np.ndarray arr):    
    PyArray_ENABLEFLAGS(arr, np.NPY_ARRAY_OWNDATA)
