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

class TestExplicitArrayMethods(unittest.TestCase):

    def test_const_int_arr_error(self):	
        with self.assertRaises(ValueError) as cm:
            x.const_int_arr='abc'
    
    def test_const_int_arr(self):	
        np_test.assert_array_equal(x.const_int_arr,np.array([1,2,3,4,5,6,7,8,9,0],dtype='int'))
    
    def test_const_real_arr(self):	
        np_test.assert_array_equal(x.const_real_arr,np.array([1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,0.0],dtype='float'))
    
    def test_const_dp_arr(self):	
        np_test.assert_array_equal(x.const_real_dp_arr,np.array([1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,0.0],dtype='float'))
    
    def test_b_int_exp_1d(self):
        v=np.random.randint(0,100,size=(5))
        x.b_int_exp_1d=v
        np_test.assert_array_equal(x.b_int_exp_1d,v)
        
    def test_b_int_exp_2d(self):
        v=np.random.randint(0,100,size=(5,5))
        x.b_int_exp_2d=v
        np_test.assert_array_equal(x.b_int_exp_2d,v)
        
    def test_b_int_exp_3d(self):
        v=np.random.randint(0,100,size=(5,5,5))
        x.b_int_exp_3d=v
        np_test.assert_array_equal(x.b_int_exp_3d,v)
        
    def test_b_int_exp_4d(self):
        v=np.random.randint(0,100,size=(5,5,5,5))
        x.b_int_exp_4d=v
        np_test.assert_array_equal(x.b_int_exp_4d,v)
        
    def test_b_int_exp_5d(self):
        v=np.random.randint(0,100,size=(5,5,5,5,5))
        x.b_int_exp_5d=v
        np_test.assert_array_equal(x.b_int_exp_5d,v)
        
    def test_b_real_exp_1d(self):
        v=np.random.random(size=(5))
        x.b_real_exp_1d=v
        np_test.assert_allclose(x.b_real_exp_1d,v)
        
    def test_b_real_exp_2d(self):
        v=np.random.random(size=(5,5))
        x.b_real_exp_2d=v
        np_test.assert_allclose(x.b_real_exp_2d,v)
        
    def test_b_real_exp_3d(self):
        v=np.random.random(size=(5,5,5))
        x.b_real_exp_3d=v
        np_test.assert_allclose(x.b_real_exp_3d,v)
        
    def test_b_real_exp_4d(self):
        v=np.random.random(size=(5,5,5,5))
        x.b_real_exp_4d=v
        np_test.assert_allclose(x.b_real_exp_4d,v)
        
    def test_b_real_exp_5d(self):
        v=np.random.random(size=(5,5,5,5,5))
        x.b_real_exp_5d=v
        np_test.assert_allclose(x.b_real_exp_5d,v)
        
    def test_b_real_exp_5d_2(self):
        v=np.random.random(size=(2,3,4,5,6))
        x.b_real_exp_5d_2=v
        np_test.assert_allclose(x.b_real_exp_5d_2,v)
        
    def test_b_real_dp_exp_1d(self):
        v=np.random.random(size=(5))
        x.b_real_dp_exp_1d=v
        np_test.assert_allclose(x.b_real_dp_exp_1d,v)
        
    def test_b_real_dp_exp_2d(self):
        v=np.random.random(size=(5,5))
        x.b_real_dp_exp_2d=v
        np_test.assert_allclose(x.b_real_dp_exp_2d,v)
        
    def test_b_real_dp_exp_3d(self):
        v=np.random.random(size=(5,5,5))
        x.b_real_dp_exp_3d=v
        np_test.assert_allclose(x.b_real_dp_exp_3d,v)
       
    def test_b_real_dp_exp_4d(self):
        v=np.random.random(size=(5,5,5,5))
        x.b_real_dp_exp_4d=v
        np_test.assert_allclose(x.b_real_dp_exp_4d,v)
        
    def test_b_real_dp_exp_5d(self):
        v=np.random.random(size=(5,5,5,5,5))
        x.b_real_dp_exp_5d=v
        np_test.assert_allclose(x.b_real_dp_exp_5d,v)
    
    
    def test_sub_array_n_int_1d(self):
        v=np.arange(0,5)
        o=' '.join([str(i) for i in v.flatten()])
        with captured_output() as (out,err):
            y=x.sub_array_n_int_1d(np.size(v),v)
        output=out.getvalue().strip()	
        self.assertEqual(output,o.strip())
    
    def test_sub_array_n_int_2d(self):
        v=[0,1,2,3,4]*5
        v=np.array(v).reshape(5,5)
        o=' '.join([str(i) for i in v.T.flatten()])
        with captured_output() as (out,err):
            y=x.sub_array_n_int_2d(5,5,v)
        output=out.getvalue().strip()	
        self.assertEqual(output,o.strip())
        
    def test_sub_exp_array_int_1d(self):
        v=np.arange(0,5)
        o=' '.join([str(i) for i in v.flatten()])
        with captured_output() as (out,err):
            y=x.sub_exp_array_int_1d(v)
        output=out.getvalue().strip()	
        self.assertEqual(output,o.strip())	
        
    def test_sub_exp_array_int_2d(self):
        v=np.arange(0,5*5).reshape((5,5))
        o=''.join([str(i).zfill(2).ljust(3) for i in v.T.flatten()])
        with captured_output() as (out,err):
            y=x.sub_exp_array_int_2d(v)
        output=out.getvalue().strip()	
        self.assertEqual(output,o.strip())	
    
    def test_sub_exp_array_int_3d(self):
        v=np.arange(0,5*5*5).reshape((5,5,5))
        o=''.join([str(i).zfill(3).ljust(4) for i in v.T.flatten()])
        with captured_output() as (out,err):
            y=x.sub_exp_array_int_3d(v)
        output=out.getvalue().strip()	
        self.assertEqual(output,o.strip())			
    
    def test_sub_exp_array_real_1d(self):
        v=np.arange(0,5.0).reshape((5))
        o='  '.join(["{:>4.1f}".format(i) for i in v.T.flatten()])
        with captured_output() as (out,err):
            y=x.sub_exp_array_real_1d(v)
        output=out.getvalue().strip()	
        self.assertEqual(output,o.strip())	
    
    def test_sub_exp_array_real_2d(self):
        v=np.arange(0,5.0*5.0).reshape((5,5))
        o='  '.join(["{:>4.1f}".format(i) for i in v.T.flatten()])
        with captured_output() as (out,err):
            y=x.sub_exp_array_real_2d(v)
        output=out.getvalue().strip()	
        self.assertEqual(output,o.strip())	
    
    def test_sub_exp_array_real_3d(self):
        v=np.arange(0,5.0*5.0*5.0).reshape((5,5,5))
        o=' '.join(["{:>5.1f}".format(i) for i in v.T.flatten()])
        with captured_output() as (out,err):
            y=x.sub_exp_array_real_3d(v)
        output=out.getvalue().strip()	
        self.assertEqual(output,o.strip())	 
    
    def test_sub_exp_array_int_1d_multi(self):
        u=19
        w=20
        v=np.arange(0,5)
        o=' '.join([str(i) for i in v.T.flatten()])
        with captured_output() as (out,err):
            y=x.sub_exp_array_int_1d_multi(u,v,w)
        output=out.getvalue().strip()	
        self.assertEqual(output,str(u)+' '+o.strip()+' '+str(w)) 
    
    def test_sub_exp_array_real_dp_1d(self):
        v=np.arange(0,5.0).reshape((5))
        o='  '.join(["{:>4.1f}".format(i) for i in v.T.flatten()])
        with captured_output() as (out,err):
            y=x.sub_exp_array_real_dp_1d(v)
        output=out.getvalue().strip()	
        self.assertEqual(output,o.strip())	
    
    def test_sub_exp_array_real_dp_2d(self):
        v=np.arange(0,5.0*5.0).reshape((5,5))
        o='  '.join(["{:>4.1f}".format(i) for i in v.T.flatten()])
        with captured_output() as (out,err):
            y=x.sub_exp_array_real_dp_2d(v)
        output=out.getvalue().strip()	
        self.assertEqual(output,o.strip())	
    
    def test_sub_exp_array_real_dp_3d(self):
        v=np.arange(0,5.0*5.0*5.0).reshape((5,5,5))
        o=' '.join(["{:>5.1f}".format(i) for i in v.T.flatten()])
        with captured_output() as (out,err):
            y=x.sub_exp_array_real_dp_3d(v)
        output=out.getvalue().strip()	
        self.assertEqual(output,o.strip())	   


    def test_sub_exp_inout(self):
        v=np.array([1,2,3,4,5])
        with captured_output() as (out,err):
            y=x.sub_exp_inout(v)
        output=out.getvalue().strip()

        np_test.assert_array_equal(y.args['x'],2*v)
        
    def test_sub_arr_exp_p(self):
        v=np.arange(0,5)
        o=' '.join([str(i) for i in v.flatten()])
        with captured_output() as (out,err):
            y=x.sub_exp_array_int_1d(v)
        output=out.getvalue().strip()	
        self.assertEqual(output,o.strip())
        
    def test_logical_arr_multi(self):
        xarr=np.zeros(5)
        xarr[:]=True
        
        y=x.func_logical_multi(1.0,2.0,xarr,3.0,4.0)
        self.assertEqual(y.result,True)
        
    @unittest.skip("Skipping as we seg fault")	
    def test_mesh_exp(self):
        # Github issue #13
        i=5
        y = x.func_mesh_exp(i)
        self.assertEqual(y.result,np.arrange(0,i))
        
    def test_check_exp_2d_2m3(self):
        # Github issue #19
        arr_test = np.zeros((3,4), dtype=np.int, order='F')

        arr_test[0,1] = 1
        arr_test[1,0] = 2
        arr_test[1,2] = 3
        arr_test[-2,-1] = 4
     
        y = x.check_exp_2d_2m3_nt(arr_test, 4, 0)
        
        self.assertEqual(y.args['success'],True)
        
        arr_test[0,3] = 5
        
        np_test.assert_array_equal(y.args['arr'],arr_test)
        
    
if __name__ == '__main__':
    unittest.main() 
