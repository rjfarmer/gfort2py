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
    @unittest.skip("Skipping as we cant set the ptr yet")
    def test_func_func_str(self):
        y = x.func_func_arg(x.func_func_run)
        self.assertEqual(y,10)

    @unittest.skip("Skipping as we cant set the ptr yet")		
    def test_func_func_ffunc(self):
        y = x.func_func_arg(x.func_func_run)
        self.assertEqual(y,10)
        
    @unittest.skip("Skipping as we cant set the ptr yet")	
    def test_func_func_py(self):
        def my_py_func(x):
            xv=x.contents.value
            return 10*xv
        
        x.func_func_run.load()
        y = x.func_func_arg([my_py_func,'func_func_run'])
        self.assertEqual(y,10)
        
    @unittest.skip("Skipping as we cant set the ptr yet")
    def test_proc_ptr_str(self):
        x.sub_null_proc_ptr()
        x.p_func_func_run_ptr = 'func_func_run'
        y = x.p_func_func_run_ptr(1)
        self.assertEqual(y,10)
        
    @unittest.skip("Skipping as we cant set the ptr yet")
    def test_proc_ptr_ffunc(self):
        x.p_func_func_run_ptr = x.func_func_run
        y = x.p_func_func_run_ptr(1)
        self.assertEqual(y,10)
        
    @unittest.skip("Skipping as we cant set the ptr yet")
    def test_proc_ptr_py(self):
        def my_py_func(x):
            return 10*x
        
        x.p_func_func_run_ptr = my_py_func
        y = x.p_func_func_run_ptr(1)
        self.assertEqual(y,10)

    @unittest.skip("Skipping as we cant set the ptr yet")		
    def test_call_set_proc_ptr(self):
        x.sub_null_proc_ptr()
        x.sub_proc_ptr2()
        y = x.p_func_func_run_ptr(1)
        self.assertEqual(y,2)
        
    @unittest.skip("Skipping as we cant set the ptr yet")
    def test_call_null_proc_ptr(self):
        x.sub_null_proc_ptr()
        with self.assertRaises(ValueError) as cm:
            y = x.p_func_func_run_ptr(1)

        x.sub_null_proc_ptr()
        x.sub_proc_ptr2()
        x.sub_null_proc_ptr()
        with self.assertRaises(ValueError) as cm:
            y = x.p_func_func_run_ptr(1)
    
    
if __name__ == '__main__':
    unittest.main() 
