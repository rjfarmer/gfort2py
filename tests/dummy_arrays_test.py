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

class TestDummyArrayMethods(unittest.TestCase):
    def test_sub_alloc_1d_arrs(self):
        y=x.sub_alloc_int_1d_arrs()
        
    def test_c_int_alloc_1d_non_alloc(self):
        y=x.sub_alloc_int_1d_cleanup()
        with self.assertRaises(gf.errors.AllocationError) as cm:
            y = x.c_int_alloc_1d
            
    def test_c_int_alloc_1d(self):
        y=x.sub_alloc_int_1d_cleanup()
        y=x.sub_alloc_int_1d_arrs()
        v=np.zeros([5])
        v[:]=1
        np_test.assert_array_equal(x.c_int_alloc_1d,v)
        y=x.sub_alloc_int_1d_cleanup()
    
    def test_c_int_alloc_2d(self):
        y=x.sub_alloc_int_1d_cleanup()
        y=x.sub_alloc_int_1d_arrs()
        v=np.zeros([5,5])
        v[:]=1
        np_test.assert_array_equal(x.c_int_alloc_2d,v)
        y=x.sub_alloc_int_1d_cleanup()
    
    def test_c_int_alloc_3d(self):
        y=x.sub_alloc_int_1d_cleanup()
        y=x.sub_alloc_int_1d_arrs()
        v=np.zeros([5,5,5])
        v[:]=1
        np_test.assert_array_equal(x.c_int_alloc_3d,v)
        y=x.sub_alloc_int_1d_cleanup()
    
    def test_c_int_alloc_4d(self):
        y=x.sub_alloc_int_1d_cleanup()
        y=x.sub_alloc_int_1d_arrs()
        v=np.zeros([5,5,5,5])
        v[:]=1
        np_test.assert_array_equal(x.c_int_alloc_4d,v)
        y=x.sub_alloc_int_1d_cleanup()
    
    def test_c_int_alloc_5d(self):
        y=x.sub_alloc_int_1d_cleanup()
        y=x.sub_alloc_int_1d_arrs()
        v=np.zeros([5,5,5,5,5])
        v[:]=1
        np_test.assert_array_equal(x.c_int_alloc_5d,v)
        y=x.sub_alloc_int_1d_cleanup()
        
    def test_c_int_alloc_1d_set(self):
        y=x.sub_alloc_int_1d_cleanup()
        y=x.sub_alloc_int_1d_arrs()
        v=np.zeros([5])
        v[:]=5
        x.c_int_alloc_1d = v
        np_test.assert_array_equal(x.c_int_alloc_1d,v)
        y=x.sub_alloc_int_1d_cleanup()
    
    def test_c_int_alloc_2d_set(self):
        y=x.sub_alloc_int_1d_cleanup()
        y=x.sub_alloc_int_1d_arrs()
        v=np.zeros([5,5])
        v[:]=5
        x.c_int_alloc_2d = v
        np_test.assert_array_equal(x.c_int_alloc_2d,v)
        y=x.sub_alloc_int_1d_cleanup()
    
    def test_c_int_alloc_3d_set(self):
        y=x.sub_alloc_int_1d_cleanup()
        y=x.sub_alloc_int_1d_arrs()
        v=np.zeros([5,5,5])
        v[:]=5
        x.c_int_alloc_3d = v
        np_test.assert_array_equal(x.c_int_alloc_3d,v)
        y=x.sub_alloc_int_1d_cleanup()
    
    def test_c_int_alloc_4d_set(self):
        y=x.sub_alloc_int_1d_cleanup()
        y=x.sub_alloc_int_1d_arrs()
        v=np.zeros([5,5,5,5])
        v[:]=5
        x.c_int_alloc_4d = v
        np_test.assert_array_equal(x.c_int_alloc_4d,v)
        y=x.sub_alloc_int_1d_cleanup()
    
    def test_c_int_alloc_5d_set(self):
        y=x.sub_alloc_int_1d_cleanup()
        y=x.sub_alloc_int_1d_arrs()
        v=np.zeros([5,5,5,5,5])
        v[:]=5
        x.c_int_alloc_5d = v
        np_test.assert_array_equal(x.c_int_alloc_5d,v)
        y=x.sub_alloc_int_1d_cleanup()
    
    def test_c_int_alloc_1d_large(self):
        # Can have issues exiting when using large (>255) arrays
        y=x.sub_alloc_int_1d_cleanup()
        y=x.sub_alloc_int_1d_arrs()
        v=np.zeros([256],dtype='int32')
        v[:]=5
        x.c_int_alloc_1d = v
        np_test.assert_array_equal(x.c_int_alloc_1d,v)
        y=x.sub_alloc_int_1d_cleanup()

    def test_c_real_alloc_1d(self):
        y=x.sub_alloc_real_1d_cleanup()
        y=x.sub_alloc_real_1d_arrs()
        v=np.zeros([5])
        v[:]=1
        np_test.assert_array_equal(x.c_real_alloc_1d,v)
        y=x.sub_alloc_real_1d_cleanup()
    
    def test_c_real_alloc_2d(self):
        y=x.sub_alloc_real_1d_cleanup()
        y=x.sub_alloc_real_1d_arrs()
        v=np.zeros([5,5])
        v[:]=1
        np_test.assert_array_equal(x.c_real_alloc_2d,v)
        y=x.sub_alloc_real_1d_cleanup()
    
    def test_c_real_alloc_3d(self):
        y=x.sub_alloc_real_1d_cleanup()
        y=x.sub_alloc_real_1d_arrs()
        v=np.zeros([5,5,5])
        v[:]=1
        np_test.assert_array_equal(x.c_real_alloc_3d,v)
        y=x.sub_alloc_real_1d_cleanup()
    
    def test_c_real_alloc_4d(self):
        y=x.sub_alloc_real_1d_cleanup()
        y=x.sub_alloc_real_1d_arrs()
        v=np.zeros([5,5,5,5])
        v[:]=1
        np_test.assert_array_equal(x.c_real_alloc_4d,v)
        y=x.sub_alloc_real_1d_cleanup()
    
    def test_c_real_alloc_5d(self):
        y=x.sub_alloc_real_1d_cleanup()
        y=x.sub_alloc_real_1d_arrs()
        v=np.zeros([5,5,5,5,5])
        v[:]=1
        np_test.assert_array_equal(x.c_real_alloc_5d,v)
        y=x.sub_alloc_real_1d_cleanup()
        
    def test_c_real_dp_alloc_1d_set(self):
        y=x.sub_alloc_real_dp_1d_cleanup()
        y=x.sub_alloc_real_dp_1d_arrs()
        v=np.zeros([5])
        v[:]=2.0
        x.c_real_dp_alloc_1d = v
        np_test.assert_array_equal(x.c_real_dp_alloc_1d,v)
        y=x.sub_alloc_real_dp_1d_cleanup()
    
    def test_c_real_dp_alloc_2d_set(self):
        y=x.sub_alloc_real_dp_1d_cleanup()
        y=x.sub_alloc_real_dp_1d_arrs()
        v=np.zeros([5,5])
        v[:]=2.0
        x.c_real_dp_alloc_2d = v
        np_test.assert_array_equal(x.c_real_dp_alloc_2d,v)
        y=x.sub_alloc_real_dp_1d_cleanup()
 
    def test_c_real_dp_alloc_3d_set(self):
        y=x.sub_alloc_real_dp_1d_cleanup()
        y=x.sub_alloc_real_dp_1d_arrs()
        v=np.zeros([5,5,5])
        v[:]=2.0
        x.c_real_dp_alloc_3d = v
        np_test.assert_array_equal(x.c_real_dp_alloc_3d,v)
        y=x.sub_alloc_real_dp_1d_cleanup()
    
    def test_c_real_dp_alloc_4d_set(self):
        y=x.sub_alloc_real_dp_1d_cleanup()
        y=x.sub_alloc_real_dp_1d_arrs()
        v=np.zeros([5,5,5,5])
        v[:]=2.0
        x.c_real_dp_alloc_4d = v
        np_test.assert_array_equal(x.c_real_dp_alloc_4d,v)
        y=x.sub_alloc_real_dp_1d_cleanup()
    
    def test_c_real_dp_alloc_5d_set(self):
        y=x.sub_alloc_real_dp_1d_cleanup()
        y=x.sub_alloc_real_dp_1d_arrs()
        v=np.zeros([5,5,5,5,5])
        v[:]=2.0
        x.c_real_dp_alloc_5d = v
        np_test.assert_array_equal(x.c_real_dp_alloc_5d,v)
        y=x.sub_alloc_real_dp_1d_cleanup()
    
    def test_c_real_dp_alloc_1d(self):
        y=x.sub_alloc_real_dp_1d_cleanup()
        y=x.sub_alloc_real_dp_1d_arrs()
        v=np.zeros([5])
        v[:]=1
        np_test.assert_array_equal(x.c_real_dp_alloc_1d,v)
        y=x.sub_alloc_real_dp_1d_cleanup()
 
    def test_c_real_dp_alloc_2d(self):
        y=x.sub_alloc_real_dp_1d_cleanup()
        y=x.sub_alloc_real_dp_1d_arrs()
        v=np.zeros([5,5])
        v[:]=1
        np_test.assert_array_equal(x.c_real_dp_alloc_2d,v)
        y=x.sub_alloc_real_dp_1d_cleanup()
    
    def test_c_real_dp_alloc_3d(self):
        y=x.sub_alloc_real_dp_1d_cleanup()
        y=x.sub_alloc_real_dp_1d_arrs()
        v=np.zeros([5,5,5])
        v[:]=1
        np_test.assert_array_equal(x.c_real_dp_alloc_3d,v)
        y=x.sub_alloc_real_dp_1d_cleanup()
    
    def test_c_real_dp_alloc_4d(self):
        y=x.sub_alloc_real_dp_1d_cleanup()
        y=x.sub_alloc_real_dp_1d_arrs()
        v=np.zeros([5,5,5,5])
        v[:]=1
        np_test.assert_array_equal(x.c_real_dp_alloc_4d,v)
        y=x.sub_alloc_real_dp_1d_cleanup()
    
    def test_c_real_dp_alloc_5d(self):
        y=x.sub_alloc_real_dp_1d_cleanup()
        y=x.sub_alloc_real_dp_1d_arrs()
        v=np.zeros([5,5,5,5,5])
        v[:]=1
        np_test.assert_array_equal(x.c_real_dp_alloc_5d,v)
        y=x.sub_alloc_real_dp_1d_cleanup()
        
    def test_c_real_dp_alloc_1d_set(self):
        y=x.sub_alloc_real_dp_1d_cleanup()
        y=x.sub_alloc_real_dp_1d_arrs()
        v=np.zeros([5])
        v[:]=2.0
        x.c_real_dp_alloc_1d = v
        np_test.assert_array_equal(x.c_real_dp_alloc_1d,v)
        y=x.sub_alloc_real_dp_1d_cleanup()
    
    def test_c_real_dp_alloc_2d_set(self):
        y=x.sub_alloc_real_dp_1d_cleanup()
        y=x.sub_alloc_real_dp_1d_arrs()
        v=np.zeros([5,5])
        v[:]=2.0
        x.c_real_dp_alloc_2d = v
        np_test.assert_array_equal(x.c_real_dp_alloc_2d,v)
        y=x.sub_alloc_real_dp_1d_cleanup()
    
    def test_c_real_dp_alloc_3d_set(self):
        y=x.sub_alloc_real_dp_1d_cleanup()
        y=x.sub_alloc_real_dp_1d_arrs()
        v=np.zeros([5,5,5])
        v[:]=2.0
        x.c_real_dp_alloc_3d = v
        np_test.assert_array_equal(x.c_real_dp_alloc_3d,v)
        y=x.sub_alloc_real_dp_1d_cleanup()
  
    def test_c_real_dp_alloc_4d_set(self):
        y=x.sub_alloc_real_dp_1d_cleanup()
        y=x.sub_alloc_real_dp_1d_arrs()
        v=np.zeros([5,5,5,5])
        v[:]=2.0
        x.c_real_dp_alloc_4d = v
        np_test.assert_array_equal(x.c_real_dp_alloc_4d,v)
        y=x.sub_alloc_real_dp_1d_cleanup()
    
    def test_c_real_dp_alloc_5d_set(self):
        y=x.sub_alloc_real_dp_1d_cleanup()
        y=x.sub_alloc_real_dp_1d_arrs()
        v=np.zeros([5,5,5,5,5])
        v[:]=2.0
        x.c_real_dp_alloc_5d = v
        np_test.assert_array_equal(x.c_real_dp_alloc_5d,v)
        y=x.sub_alloc_real_dp_1d_cleanup()
        
    
    def test_func_assumed_shape_arr_1d(self):
        v=np.zeros([5],dtype='int32')
        v[0]=2.0
        y=x.func_assumed_shape_arr_1d(v)
        self.assertEqual(y.result,True)
        np_test.assert_array_equal(y.args['x'],np.array([9,9,9,9,9]))
        
    def test_func_assumed_shape_arr_2d(self):
        v=np.zeros([5,5],dtype='int32')
        v[1,0]=2.0
        y=x.func_assumed_shape_arr_2d(v)
        self.assertEqual(y.result,True)
        
    def test_func_assumed_shape_arr_3d(self):
        v=np.zeros([5,5,5],dtype='int32')
        v[2,1,0]=2.0
        y=x.func_assumed_shape_arr_3d(v)
        self.assertEqual(y.result,True)
        
    def test_func_assumed_shape_arr_4d(self):
        v=np.zeros([5,5,5,5],dtype='int32')
        v[3,2,1,0]=2.0
        y=x.func_assumed_shape_arr_4d(v)
        self.assertEqual(y.result,True)
        
    def test_func_assumed_shape_arr_5d(self):
        v=np.zeros([5,5,5,5,5],dtype='int32')
        v[4,3,2,1,0]=2.0
        y=x.func_assumed_shape_arr_5d(v)
        self.assertEqual(y.result,True)
        
    def test_func_assumed_size_arr_1d(self):
        v=np.zeros([5],dtype='int32')
        v[1]=2
        y=x.func_assumed_size_arr_1d(v)
        self.assertEqual(y.result,True)
        
    def test_func_assumed_size_arr_real_1d(self):
        v=np.zeros([5],dtype='float32')
        v[1]=2.0
        y=x.func_assumed_size_arr_real_1d(v)
        self.assertEqual(y.result,True)
        
    def test_func_assumed_size_arr_real_dp_1d(self):
        v=np.zeros([5],dtype='float64')
        v[1]=2.0
        y=x.func_assumed_size_arr_real_dp_1d(v)
        self.assertEqual(y.result,True)
            
    def test_sub_alloc_arr_1d(self):
        y=x.sub_alloc_arr_1d(None)
        vTest=np.zeros(10)
        vTest[:]=10
        np_test.assert_array_equal(y.args['x'],vTest)
        
    def test_logical_arr(self):
        xarr=np.zeros(10)
        x2arr=np.zeros(10)
        x2arr[:]=False
        xarr[:]=True
        
        y=x.func_alltrue_arr_1d(xarr)
        y2=x.func_allfalse_arr_1d(x2arr)
        self.assertEqual(y.result,True)
        self.assertEqual(y2.result,True)

        
    def test_sub_arr_assumed_rank_int_1d(self):
        v=np.arange(10,15)
        o=' '.join([str(i) for i in v.flatten()])
        with captured_output() as (out,err):
            y=x.sub_arr_assumed_rank_int_1d(v)
        output=out.getvalue().strip()	
        np_test.assert_array_equal(y.args['zzz'],np.array([100]*5))
        
    def test_sub_arr_assumed_rank_real_1d(self):
        v=np.arange(10.0,15.0)
        o=' '.join([str(i) for i in v.flatten()])
        with captured_output() as (out,err):
            y=x.sub_arr_assumed_rank_real_1d(v)
        output=out.getvalue().strip()	
        np_test.assert_array_equal(y.args['zzz'],np.array([100.0]*5))
        
    def test_sub_arr_assumed_rank_dp_1d(self):
        v=np.arange(10.0,15.0)
        o=' '.join([str(i) for i in v.flatten()])
        with captured_output() as (out,err):
            y=x.sub_arr_assumed_rank_dp_1d(v)
        output=out.getvalue().strip()	
        np_test.assert_array_equal(y.args['zzz'],np.array([100.0]*5))
        
    
    
    
if __name__ == '__main__':
    unittest.main() 
