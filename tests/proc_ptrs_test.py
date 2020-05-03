# SPDX-License-Identifier: GPL-2.0+

import os, sys

os.environ["_GFORT2PY_TEST_FLAG"] = "1"

import numpy as np
import gfort2py as gf

import unittest as unittest
    
import subprocess
import numpy.testing as np_test

from contextlib import contextmanager
from io import StringIO
from io import BytesIO

#Decreases recursion depth to make debugging easier
# sys.setrecursionlimit(10)

SO = __file__.replace('_test.py','')+'.so'
MOD =__file__.replace('_test.py','')+'.mod'

x=gf.fFort(SO,MOD,rerun=True)



@contextmanager
def captured_output():
    """
    For use when we need to grab the stdout/stderr from fortran (but only in testing)
    Use as:
    with captured_output() as (out,err):
        func()
    output=out.getvalue().strip()
    error=err.getvalue().strip()
    """
    new_out, new_err = StringIO(),StringIO()
    old_out,old_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = new_out, new_err
        yield sys.stdout, sys.stderr
    finally:
        sys.stdout, sys.stderr = old_out, old_err

class TestProcPtrsMethods(unittest.TestCase):
    def test_proc_ptr_ffunc(self):
        x.sub_null_proc_ptr()
        with self.assertRaises(AttributeError) as cm:
            y = x.p_func_func_run_ptr(1)
        
        x.p_func_func_run_ptr = x.func_func_run
        y = x.p_func_func_run_ptr(1)
        self.assertEqual(y.result,10)
        y = x.p_func_func_run_ptr(2)
        self.assertEqual(y.result,20)
        
        
    def test_proc_ptr_ffunc2(self):
        x.sub_null_proc_ptr()
        with self.assertRaises(AttributeError) as cm:
            y = x.p_func_func_run_ptr2(1) # Allready set
        
        x.p_func_func_run_ptr2 = x.func_func_run
        y = x.p_func_func_run_ptr2(10)
        self.assertEqual(y.result,100)        

    def test_proc_update(self):
        x.sub_null_proc_ptr()
        x.p_func_func_run_ptr = x.func_func_run
        y = x.p_func_func_run_ptr(1)
        self.assertEqual(y.result,10)
        
        x.p_func_func_run_ptr = x.func_func_run2
        y = x.p_func_func_run_ptr(1)
        self.assertEqual(y.result,2)
    
    
if __name__ == '__main__':
    unittest.main() 
