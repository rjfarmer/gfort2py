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

SO = './pointers.so'
MOD ='./ptrs.mod'

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

class TestPtrsMethods(unittest.TestCase):
    
    def test_a_int_point(self):
        v=1
        x.a_int_point=v
        self.assertEqual(x.a_int_point,v)
    
    def test_a_int_lp_point(self):
        v=1
        x.a_int_lp_point=v
        self.assertEqual(x.a_int_lp_point,v)
    
    def test_a_real_point(self):
        v=1.0
        x.a_real_point=v
        self.assertEqual(x.a_real_point,v)
        
    def test_a_real_dp_point(self):
        v=1.0
        x.a_real_dp_point=v
        self.assertEqual(x.a_real_dp_point,v)
        
    def test_a_real_qp_point(self):
        v=1.0
        x.a_real_qp_point=v
        self.assertEqual(x.a_real_qp_point,v)
        
    def test_a_str_point(self):
        v='abcdefghij'
        x.a_str_point=v
        self.assertEqual(x.a_str_point,v)
    
    def test_a_int_target(self):
        v=1
        x.a_int_target=v
        self.assertEqual(x.a_int_target,v)
    
    def test_a_int_lp_target(self):
        v=1
        x.a_int_lp_target=v
        self.assertEqual(x.a_int_lp_target,v)
    
    def test_a_real_target(self):
        v=1.0
        x.a_real_target=v
        self.assertEqual(x.a_real_target,v)
        
    def test_a_real_dp_target(self):
        v=1.0
        x.a_real_dp_target=v
        self.assertEqual(x.a_real_dp_target,v)
        
    def test_a_real_qp_target(self):
        v=1.0
        x.a_real_qp_target=v
        self.assertEqual(x.a_real_qp_target,v)
        
    def test_a_str_target(self):
        v='abcdefghij'
        x.a_str_target=v
        self.assertEqual(x.a_str_target,v)
    
if __name__ == '__main__':
    unittest.main() 
