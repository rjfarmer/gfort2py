import os, sys
import numpy as np
import gfort2py as gf
import unittest2 as unittest
import subprocess
import numpy.testing as np_test

from contextlib import contextmanager
try:
	from StringIO import StringIO
	from BytesIO import BytesIO
except ImportError:
	from io import StringIO
	from io import BytesIO

os.chdir('tests')
subprocess.check_output(["make"])
x=gf.fFort('./tester.so','tester.mod',reload=True,TEST_FLAG=True)

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
	
	def test_mising_var(self):	
		with self.assertRaises(AttributeError) as cm:
			a=x.invalid_var.get()
	
	def test_a_str(self):
		v='123456798'
		x.a_str=v
		self.assertEqual(x.a_str.get(),v)
		
	def test_a_str_bad_length(self):
		v='132456789kjhgjhf'
		x.a_str=v
		self.assertEqual(x.a_str.get(),v[0:10])
		
	def test_a_int(self):
		v=1
		x.a_int=v
		self.assertEqual(x.a_int.get(),v)
		
	def test_a_int_str(self):
		with self.assertRaises(ValueError) as cm:
			x.a_int='abc'
			
	def test_a_real(self):
		v=1.0
		x.a_real=v
		self.assertEqual(x.a_real.get(),v)
	
	def test_a_real_str(self):	
		with self.assertRaises(ValueError) as cm:
			x.a_real='abc'
			
	def test_const_int_set(self):	
		with self.assertRaises(ValueError) as cm:
			x.const_int=2
			
	def test_const_int(self):	
		self.assertEqual(x.const_int.get(),1)	

	def test_const_int_p1(self):	
		self.assertEqual(x.const_int_p1.get(),2)	

	def test_const_int_long(self):	
		self.assertEqual(x.const_int_lp.get(),1)	

	def test_const_real_dp(self):	
		self.assertEqual(x.const_real_dp.get(),1.0)
		
	def test_const_real_qp(self):	
		self.assertEqual(x.const_real_qp.get(),1.0)

	def test_const_int_arr_error(self):	
		with self.assertRaises(ValueError) as cm:
			x.const_int_arr='abc'
		
	def test_const_int_arr(self):	
		np_test.assert_array_equal(x.const_int_arr.get(),np.array([1,2,3,4,5,6,7,8,9,0],dtype='int'))

	def test_const_real_arr(self):	
		np_test.assert_array_equal(x.const_real_arr.get(),np.array([1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,0.0],dtype='float'))

	def test_const_dp_arr(self):	
		np_test.assert_array_equal(x.const_real_dp_arr.get(),np.array([1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,0.0],dtype='float'))

	def test_b_int_exp_1d(self):
		v=np.random.randint(0,100,size=(5))
		x.b_int_exp_1d=v
		np_test.assert_array_equal(x.b_int_exp_1d.get(),v)
		
	def test_b_int_exp_2d(self):
		v=np.random.randint(0,100,size=(5,5))
		x.b_int_exp_2d=v
		np_test.assert_array_equal(x.b_int_exp_2d.get(),v)
		
	def test_b_int_exp_3d(self):
		v=np.random.randint(0,100,size=(5,5,5))
		x.b_int_exp_3d=v
		np_test.assert_array_equal(x.b_int_exp_3d.get(),v)
		
	def test_b_int_exp_4d(self):
		v=np.random.randint(0,100,size=(5,5,5,5))
		x.b_int_exp_4d=v
		np_test.assert_array_equal(x.b_int_exp_4d.get(),v)
		
	def test_b_int_exp_5d(self):
		v=np.random.randint(0,100,size=(5,5,5,5,5))
		x.b_int_exp_5d=v
		np_test.assert_array_equal(x.b_int_exp_5d.get(),v)
		
	def test_b_real_exp_1d(self):
		v=np.random.random(size=(5))
		x.b_real_exp_1d=v
		np_test.assert_allclose(x.b_real_exp_1d.get(),v)
		
	def test_b_real_exp_2d(self):
		v=np.random.random(size=(5,5))
		x.b_real_exp_2d=v
		np_test.assert_allclose(x.b_real_exp_2d.get(),v)
		
	def test_b_real_exp_3d(self):
		v=np.random.random(size=(5,5,5))
		x.b_real_exp_3d=v
		np_test.assert_allclose(x.b_real_exp_3d.get(),v)
		
	def test_b_real_exp_4d(self):
		v=np.random.random(size=(5,5,5,5))
		x.b_real_exp_4d=v
		np_test.assert_allclose(x.b_real_exp_4d.get(),v)
		
	def test_b_real_exp_5d(self):
		v=np.random.random(size=(5,5,5,5,5))
		x.b_real_exp_5d=v
		np_test.assert_allclose(x.b_real_exp_5d.get(),v)
		
	def test_b_real_dp_exp_1d(self):
		v=np.random.random(size=(5))
		x.b_real_dp_exp_1d=v
		np_test.assert_allclose(x.b_real_dp_exp_1d.get(),v)
		
	def test_b_real_dp_exp_2d(self):
		v=np.random.random(size=(5,5))
		x.b_real_dp_exp_2d=v
		np_test.assert_allclose(x.b_real_dp_exp_2d.get(),v)
		
	def test_b_real_dp_exp_3d(self):
		v=np.random.random(size=(5,5,5))
		x.b_real_dp_exp_3d=v
		np_test.assert_allclose(x.b_real_dp_exp_3d.get(),v)
		
	def test_b_real_dp_exp_4d(self):
		v=np.random.random(size=(5,5,5,5))
		x.b_real_dp_exp_4d=v
		np_test.assert_allclose(x.b_real_dp_exp_4d.get(),v)
		
	def test_b_real_dp_exp_5d(self):
		v=np.random.random(size=(5,5,5,5,5))
		x.b_real_dp_exp_5d=v
		np_test.assert_allclose(x.b_real_dp_exp_5d.get(),v)

	def test_a_int_point(self):
		v=1
		x.a_int_point=v
		self.assertEqual(x.a_int_point.get(),v)

	def test_a_int_lp_point(self):
		v=1
		x.a_int_lp_point=v
		self.assertEqual(x.a_int_lp_point.get(),v)

	def test_a_real_point(self):
		v=1.0
		x.a_real_point=v
		self.assertEqual(x.a_real_point.get(),v)
		
	def test_a_real_dp_point(self):
		v=1.0
		x.a_real_dp_point=v
		self.assertEqual(x.a_real_dp_point.get(),v)
		
	def test_a_real_qp_point(self):
		v=1.0
		x.a_real_qp_point=v
		self.assertEqual(x.a_real_qp_point.get(),v)
		
	def test_a_str_point(self):
		v='abcdefghij'
		x.a_str_point=v
		self.assertEqual(x.a_str_point.get(),v)

	def test_a_int_target(self):
		v=1
		x.a_int_target=v
		self.assertEqual(x.a_int_target.get(),v)

	def test_a_int_lp_target(self):
		v=1
		x.a_int_lp_target=v
		self.assertEqual(x.a_int_lp_target.get(),v)

	def test_a_real_target(self):
		v=1.0
		x.a_real_target=v
		self.assertEqual(x.a_real_target.get(),v)
		
	def test_a_real_dp_target(self):
		v=1.0
		x.a_real_dp_target=v
		self.assertEqual(x.a_real_dp_target.get(),v)
		
	def test_a_real_qp_target(self):
		v=1.0
		x.a_real_qp_target=v
		self.assertEqual(x.a_real_qp_target.get(),v)
		
	def test_a_str_target(self):
		v='abcdefghij'
		x.a_str_target=v
		self.assertEqual(x.a_str_target.get(),v)

	def test_a_const_cmplx(self):
		self.assertEqual(x.const_cmplx.get(),complex(1.0,1.0))
		
	def test_a_const_cmplx_dp(self):
		self.assertEqual(x.const_cmplx_dp.get(),complex(1.0,1.0))
		
	def test_a_const_cmplx_qp(self):
		self.assertEqual(x.const_cmplx_qp.get(),complex(1.0,1.0))
		
	def test_a_cmplx(self):
		v=complex(1.0,1.0)
		x.a_cmplx=v
		self.assertEqual(x.a_cmplx.get(),v)

	def test_a_cmplx_dp(self):
		v=complex(1.0,1.0)
		x.a_cmplx_dp=v
		self.assertEqual(x.a_cmplx_dp.get(),v)
		
	def test_a_cmplx_qp(self):
		v=complex(1.0,1.0)
		x.a_cmplx_qp=v
		self.assertEqual(x.a_cmplx_qp.get(),v)
		
	def test_sub_no_args(self):
		with captured_output() as (out,err):
			x.sub_no_args()
		output=out.getvalue().strip()
		self.assertEqual(output,"1")
		
	def test_sub_alter_mod(self):
		x.sub_alter_mod()
		self.assertEqual(x.a_int.get(),99)
		self.assertEqual(x.a_int_lp.get(),99)
		self.assertEqual(x.a_real.get(),99.0)
		self.assertEqual(x.a_real_dp.get(),99.0)
		#self.assertEqual(x.a_real_qp.get(),99.0)
		self.assertEqual(x.a_str.get(),"9999999999")
		self.assertEqual(x.a_cmplx.get(),complex(99.0,99.0))
		self.assertEqual(x.a_cmplx_dp.get(),complex(99.0,99.0))
		#self.assertEqual(x.a_cmplx_qp.get(),complex(99.0,99.0))	
		
	def test_alloc_1d_arrs(self):
		x.sub_alloc_int_1d_arrs()

	def test_func_int_in(self):
		v=5
		y=x.func_int_in(v)
		self.assertEqual(int(y),2*v)
		
	def test_func_int_in_multi(self):
		v=5
		w=3
		u=4
		y=x.func_int_in_multi(v,w,u)
		self.assertEqual(y,v+w+u)
		
	def test_sub_int_in(self):
		v=5
		with captured_output() as (out,err):
			x.sub_int_in(v)
		output=out.getvalue().strip()
		self.assertEqual(int(output),2*v)	

	def test_func_int_no_args(self):
		y=x.func_int_no_args()
		self.assertEqual(y,2)
		
	def test_func_real_no_args(self):
		y=x.func_real_no_args()
		self.assertEqual(y,3.0)

	def test_func_real_dp_no_args(self):
		y=x.func_real_dp_no_args()
		self.assertEqual(y,4.0)
		
	def test_sub_str_in_explicit(self):
		v='1324567980'
		with captured_output() as (out,err):
			x.sub_str_in_explicit(v)
		output=out.getvalue().strip()
		self.assertEqual(output,v)	
		
	def test_sub_str_in_implicit(self):
		v='123456789'
		with captured_output() as (out,err):
			x.sub_str_in_implicit(v)
		output=out.getvalue().strip()	
		self.assertEqual(output,v)	
	
	def test_sub_str_multi(self):
		v=5
		u='123456789'
		w=4
		with captured_output() as (out,err):
			x.sub_str_multi(v,u,w)
		output=out.getvalue().strip()	
		self.assertEqual(output,str(v+w)+' '+u)	

if __name__ == '__main__':
	unittest.main() 



