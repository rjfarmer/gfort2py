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

class TestBasicMethods(unittest.TestCase):

    def test_mising_var(self):	
        with self.assertRaises(AttributeError) as cm:
            a=x.invalid_var
            
    def test_a_int(self):
        v=1
        x.a_int=v
        self.assertEqual(x.a_int,v)
    
    def test_a_int_str(self):
        with self.assertRaises(TypeError) as cm:
            x.a_int='abc'
            
    def test_a_real(self):
        v=1.0
        x.a_real=v
        self.assertEqual(x.a_real,v)
    
    def test_a_real_str(self):	
        with self.assertRaises(TypeError) as cm:
            x.a_real='abc'
        
    def test_const_int_set(self):	
        with self.assertRaises(ValueError) as cm:
            x.const_int=2
        
    def test_const_int(self):	
        self.assertEqual(x.const_int,1)	
    
    def test_const_int_p1(self):	
        self.assertEqual(x.const_int_p1,2)	
    
    def test_const_int_long(self):	
        self.assertEqual(x.const_int_lp,1)	
    
    def test_const_real_dp(self):	
        self.assertEqual(x.const_real_dp,1.0)
    
    def test_const_real_pi_dp(self):	
        self.assertEqual(x.const_real_pi_dp,3.14)
    
    def test_const_real_qp(self):	
        self.assertEqual(x.const_real_qp,1.0)
        
    def test_sub_no_args(self):
        with captured_output() as (out,err):
            x.sub_no_args()
        output=out.getvalue().strip()
        self.assertEqual(output,"1")
        
    def test_sub_alter_mod(self):
        y=x.sub_alter_mod()
        self.assertEqual(x.a_int,99)
        self.assertEqual(x.a_int_lp,99)
        self.assertEqual(x.a_real,99.0)
        self.assertEqual(x.a_real_dp,99.0)
        
    @unittest.skip("Skipping due to quad support")	
    def test_sub_alter_mod_qp(self):
        y=x.sub_alter_mod()
        self.assertEqual(x.a_real_qp,99.0)

    def test_func_int_in(self):
        v=5
        y=x.func_int_in(v)
        self.assertEqual(int(y.result),2*v)
        
    def test_func_int_in_multi(self):
        v=5
        w=3
        u=4
        y=x.func_int_in_multi(v,w,u)
        self.assertEqual(y.result,v+w+u)
        
    def test_sub_int_in(self):
        v=5
        with captured_output() as (out,err):
            y=x.sub_int_in(v)
        output=out.getvalue().strip()
        self.assertEqual(int(output),2*v)	

    def test_func_int_no_args(self):
        y=x.func_int_no_args()
        self.assertEqual(y.result,2)
        
    def test_func_real_no_args(self):
        y=x.func_real_no_args()
        self.assertEqual(y.result,3.0)

    def test_func_real_dp_no_args(self):
        y=x.func_real_dp_no_args()
        self.assertEqual(y.result,4.0)
  
  
    def test_sub_int_out(self):
        v=5
        with captured_output() as (out,err):
            y=x.sub_int_out(v)
        output=out.getvalue().strip()
        self.assertEqual(y.args,{'x':1})		

    def test_sub_int_inout(self):
        v=5
        with captured_output() as (out,err):
            y=x.sub_int_inout(v)
        output=out.getvalue().strip()
        self.assertEqual(y.args,{'x':2*v})
        
    def test_sub_int_no_intent(self):
        v=5
        with captured_output() as (out,err):
            y=x.sub_int_no_intent(v)
        output=out.getvalue().strip()
        self.assertEqual(y.args,{'x':2*v})
        
    def test_sub_real_inout(self):
        v=5.0
        with captured_output() as (out,err):
            y=x.sub_real_inout(v)
        output=out.getvalue().strip()
        self.assertEqual(y.args,{'x':2*v})
  
    def test_func_return_res(self):
        y=x.func_return_res(2)
        self.assertEqual(y.result,True)
        y=x.func_return_res(10)
        self.assertEqual(y.result,False)	
   
    def test_sub_int_p(self):
        with captured_output() as (out,err):
            y=x.sub_int_p(1)
        output=out.getvalue().strip()
        self.assertEqual(output,'1')
        self.assertEqual(y.args['zzz'],5)

    def test_sub_real_p(self):
        with captured_output() as (out,err):
            y=x.sub_real_p(1.0)
        output=out.getvalue().strip()
        self.assertEqual(output,'1.00')
        self.assertEqual(y.args['zzz'],5.0)
  
    def test_sub_opt(self):
        with captured_output() as (out,err):
            y=x.sub_int_opt(1)
        output=out.getvalue().strip()
        self.assertEqual(output,'100')
        with captured_output() as (out,err):
            y=x.sub_int_opt(None)
        output=out.getvalue().strip()
        self.assertEqual(output,'200')
        
    def test_second_mod(self):
        x.f_struct_simple2.x=99
        y=x.sub_use_mod()
        self.assertEqual(x.test2_x,1)
        self.assertEqual(x.f_struct_simple2.x,5)
        self.assertEqual(x.f_struct_simple2.y,6)
        self.assertEqual(x.f_struct_simple2.z,0)
    
    
    def test_func_value(self):
        y = x.func_int_value(5)
        self.assertEqual(y.result,10)
        
        
    def test_func_pass_mod_var(self):
        x.a_int = 5
        z = x.func_int_in(x.a_int)
        self.assertEqual(z.result,10)
        
        
    def test_sub_man_args(self):
        # if this doesn't seg fault we are good
        x.sub_many_args(1,2,3,4,True,False,True,
                'abc','def','ghj','qwerty','zxcvb')
                
    def test_func_intent_out(self):
        y = x.func_intent_out(9,0)
        self.assertEqual(y.result,9)
        self.assertEqual(y.args['x'],9) 
        
    def test_func_result(self):
        y = x.func_result(9,0)
        self.assertEqual(y.result,18)
        self.assertEqual(y.args['y'],9)      

    
    
if __name__ == '__main__':
    unittest.main() 
