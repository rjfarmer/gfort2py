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

class TestStringMethods(unittest.TestCase):
    def test_a_str(self):
        v='123456798 '
        x.a_str=v
        self.assertEqual(x.a_str,v)
        
    def test_a_str_bad_length(self):
        v='132456789kjhgjhf'
        x.a_str=v
        self.assertEqual(x.a_str,v[0:10])
        
    def test_sub_str_in_explicit(self):
        v='1324567980'
        with captured_output() as (out,err):
            y=x.sub_str_in_explicit(v)
        output=out.getvalue().strip()
        self.assertEqual(output,v)	
        
    def test_sub_str_in_implicit(self):
        v='123456789'
        with captured_output() as (out,err):
            y=x.sub_str_in_implicit(v)
        output=out.getvalue().strip()	
        self.assertEqual(output,v)	
    
    def test_sub_str_multi(self):
        v=5
        u='123456789'
        w=4
        with captured_output() as (out,err):
            y=x.sub_str_multi(v,u,w)
        output=out.getvalue().strip()	
        self.assertEqual(output,str(v+w)+' '+u)	
        
    def test_sub_str_p(self):
        with captured_output() as (out,err):
            y=x.sub_str_p('abcdef')
        output=out.getvalue().strip()
        self.assertEqual(output,'abcdef')
        self.assertEqual(y.args['zzz'],'xyzxyz')
    
    
if __name__ == '__main__':
    unittest.main() 
