import os, sys

os.environ["_GFORT2PY_TEST_FLAG"] = "1"

import numpy as np
import gfort2py as gf

try:
	import unittest as unittest
except ImportError:
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
x=gf.fFort('./tester.so','tester.mod',rerun=True)

#Decreases recursion depth to make debugging easier
#sys.setrecursionlimit(100)


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
		
	def test_const_real_pi_dp(self):	
		self.assertEqual(x.const_real_pi_dp.get(),3.14)
		
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
		y=x.sub_alter_mod()
		self.assertEqual(x.a_int.get(),99)
		self.assertEqual(x.a_int_lp.get(),99)
		self.assertEqual(x.a_real.get(),99.0)
		self.assertEqual(x.a_real_dp.get(),99.0)
		#self.assertEqual(x.a_real_qp.get(),99.0)
		self.assertEqual(x.a_str.get(),"9999999999")
		self.assertEqual(x.a_cmplx.get(),complex(99.0,99.0))
		self.assertEqual(x.a_cmplx_dp.get(),complex(99.0,99.0))
		#self.assertEqual(x.a_cmplx_qp.get(),complex(99.0,99.0))	
		
	def test_sub_alloc_1d_arrs(self):
		y=x.sub_alloc_int_1d_arrs()

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
			y=x.sub_int_in(v)
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
		
		
	def test_sub_exp_array_int_1d(self):
		v=np.arange(0,5)
		o=' '.join([str(i) for i in v.flatten()])
		with captured_output() as (out,err):
			y=x.sub_exp_array_int_1d(v)
		output=out.getvalue().strip()	
		self.assertEqual(output,o.strip())	
		
	def test_sub_exp_array_int_2d(self):
		v=np.arange(0,5*5).reshape((5,5))
		o=''.join([str(i).zfill(2).ljust(3) for i in v.flatten()])
		with captured_output() as (out,err):
			y=x.sub_exp_array_int_2d(v)
		output=out.getvalue().strip()	
		self.assertEqual(output,o.strip())	

	def test_sub_exp_array_int_3d(self):
		v=np.arange(0,5*5*5).reshape((5,5,5))
		o=''.join([str(i).zfill(3).ljust(4) for i in v.flatten()])
		with captured_output() as (out,err):
			y=x.sub_exp_array_int_3d(v)
		output=out.getvalue().strip()	
		self.assertEqual(output,o.strip())			


	def test_sub_exp_array_real_1d(self):
		v=np.arange(0,5.0).reshape((5))
		o='  '.join(["{:>4.1f}".format(i) for i in v.flatten()])
		with captured_output() as (out,err):
			y=x.sub_exp_array_real_1d(v)
		output=out.getvalue().strip()	
		self.assertEqual(output,o.strip())	
		
	def test_sub_exp_array_real_2d(self):
		v=np.arange(0,5.0*5.0).reshape((5,5))
		o='  '.join(["{:>4.1f}".format(i) for i in v.flatten()])
		with captured_output() as (out,err):
			y=x.sub_exp_array_real_2d(v)
		output=out.getvalue().strip()	
		self.assertEqual(output,o.strip())	

	def test_sub_exp_array_real_3d(self):
		v=np.arange(0,5.0*5.0*5.0).reshape((5,5,5))
		o=' '.join(["{:>5.1f}".format(i) for i in v.flatten()])
		with captured_output() as (out,err):
			y=x.sub_exp_array_real_3d(v)
		output=out.getvalue().strip()	
		self.assertEqual(output,o.strip())	 

	def test_sub_exp_array_int_1d_multi(self):
		u=19
		w=20
		v=np.arange(0,5)
		o=' '.join([str(i) for i in v.flatten()])
		with captured_output() as (out,err):
			y=x.sub_exp_array_int_1d_multi(u,v,w)
		output=out.getvalue().strip()	
		self.assertEqual(output,str(u)+' '+o.strip()+' '+str(w)) 
 
 
	def test_sub_exp_array_real_dp_1d(self):
		v=np.arange(0,5.0).reshape((5))
		o='  '.join(["{:>4.1f}".format(i) for i in v.flatten()])
		with captured_output() as (out,err):
			y=x.sub_exp_array_real_dp_1d(v)
		output=out.getvalue().strip()	
		self.assertEqual(output,o.strip())	
		
	def test_sub_exp_array_real_dp_2d(self):
		v=np.arange(0,5.0*5.0).reshape((5,5))
		o='  '.join(["{:>4.1f}".format(i) for i in v.flatten()])
		with captured_output() as (out,err):
			y=x.sub_exp_array_real_dp_2d(v)
		output=out.getvalue().strip()	
		self.assertEqual(output,o.strip())	

	def test_sub_exp_array_real_dp_3d(self):
		v=np.arange(0,5.0*5.0*5.0).reshape((5,5,5))
		o=' '.join(["{:>5.1f}".format(i) for i in v.flatten()])
		with captured_output() as (out,err):
			y=x.sub_exp_array_real_dp_3d(v)
		output=out.getvalue().strip()	
		self.assertEqual(output,o.strip())	   

	def test_sub_int_out(self):
		v=5
		with captured_output() as (out,err):
			y=x.sub_int_out(v)
		output=out.getvalue().strip()
		self.assertEqual(y,{'x':1})		

	def test_sub_int_inout(self):
		v=5
		with captured_output() as (out,err):
			y=x.sub_int_inout(v)
		output=out.getvalue().strip()
		self.assertEqual(y,{'x':2*v})
		
	def test_sub_int_no_intent(self):
		v=5
		with captured_output() as (out,err):
			y=x.sub_int_no_intent(v)
		output=out.getvalue().strip()
		self.assertEqual(y,{'x':2*v})
		
	def test_sub_real_inout(self):
		v=5.0
		with captured_output() as (out,err):
			y=x.sub_real_inout(v)
		output=out.getvalue().strip()
		self.assertEqual(y,{'x':2*v})
		
	def test_sub_exp_inout(self):
		v=np.array([1,2,3,4,5])
		with captured_output() as (out,err):
			y=x.sub_exp_inout(v)
		output=out.getvalue().strip()

		np_test.assert_array_equal(y['x'],2*v)
		
	def test_dt_set_value(self):
		x.f_struct_simple.x=1
		x.f_struct_simple.y=0
		y=x.f_struct_simple.get()
		self.assertEqual(y,{'x':1,'y':0})
		
	def test_dt_set_dict(self):	
		x.f_struct_simple={'x':5,'y':5}
		y=x.f_struct_simple.get()
		self.assertEqual(y,{'x':5,'y':5})
			
	def test_dt_bad_dict(self):
		with self.assertRaises(ValueError) as cm:
			x.f_struct_simple = {'asw':2,'y':0}
			
	def test_dt_bad_value(self):
		with self.assertRaises(TypeError) as cm:
			x.f_struct_simple.x='asde'
	
	#def test_c_int_alloc_1d_non_alloc(self):
		#y=x.sub_alloc_int_1d_cleanup()
		#with self.assertRaises(ValueError) as cm:
			#x.c_int_alloc_1d.get()
			
	def test_c_int_alloc_1d(self):
		y=x.sub_alloc_int_1d_cleanup()
		y=x.sub_alloc_int_1d_arrs()
		v=np.zeros([5])
		v[:]=1
		np_test.assert_array_equal(x.c_int_alloc_1d.get(),v)
		y=x.sub_alloc_int_1d_cleanup()
	
	def test_c_int_alloc_2d(self):
		y=x.sub_alloc_int_1d_cleanup()
		y=x.sub_alloc_int_1d_arrs()
		v=np.zeros([5,5])
		v[:]=1
		np_test.assert_array_equal(x.c_int_alloc_2d.get(),v)
		y=x.sub_alloc_int_1d_cleanup()
	
	def test_c_int_alloc_3d(self):
		y=x.sub_alloc_int_1d_cleanup()
		y=x.sub_alloc_int_1d_arrs()
		v=np.zeros([5,5,5])
		v[:]=1
		np_test.assert_array_equal(x.c_int_alloc_3d.get(),v)
		y=x.sub_alloc_int_1d_cleanup()
	
	def test_c_int_alloc_4d(self):
		y=x.sub_alloc_int_1d_cleanup()
		y=x.sub_alloc_int_1d_arrs()
		v=np.zeros([5,5,5,5])
		v[:]=1
		np_test.assert_array_equal(x.c_int_alloc_4d.get(),v)
		y=x.sub_alloc_int_1d_cleanup()
	
	def test_c_int_alloc_5d(self):
		y=x.sub_alloc_int_1d_cleanup()
		y=x.sub_alloc_int_1d_arrs()
		v=np.zeros([5,5,5,5,5])
		v[:]=1
		np_test.assert_array_equal(x.c_int_alloc_5d.get(),v)
		y=x.sub_alloc_int_1d_cleanup()
		
	
	def test_c_int_alloc_1d_set(self):
		y=x.sub_alloc_int_1d_cleanup()
		y=x.sub_alloc_int_1d_arrs()
		v=np.zeros([5])
		v[:]=5
		x.c_int_alloc_1d = v
		np_test.assert_array_equal(x.c_int_alloc_1d.get(),v)
		y=x.sub_alloc_int_1d_cleanup()
	
	def test_c_int_alloc_2d_set(self):
		y=x.sub_alloc_int_1d_cleanup()
		y=x.sub_alloc_int_1d_arrs()
		v=np.zeros([5,5])
		v[:]=5
		x.c_int_alloc_2d = v
		np_test.assert_array_equal(x.c_int_alloc_2d.get(),v)
		y=x.sub_alloc_int_1d_cleanup()
	
	def test_c_int_alloc_3d_set(self):
		y=x.sub_alloc_int_1d_cleanup()
		y=x.sub_alloc_int_1d_arrs()
		v=np.zeros([5,5,5])
		v[:]=5
		x.c_int_alloc_3d = v
		np_test.assert_array_equal(x.c_int_alloc_3d.get(),v)
		y=x.sub_alloc_int_1d_cleanup()
	
	def test_c_int_alloc_4d_set(self):
		y=x.sub_alloc_int_1d_cleanup()
		y=x.sub_alloc_int_1d_arrs()
		v=np.zeros([5,5,5,5])
		v[:]=5
		x.c_int_alloc_4d = v
		np_test.assert_array_equal(x.c_int_alloc_4d.get(),v)
		y=x.sub_alloc_int_1d_cleanup()
	
	def test_c_int_alloc_5d_set(self):
		y=x.sub_alloc_int_1d_cleanup()
		y=x.sub_alloc_int_1d_arrs()
		v=np.zeros([5,5,5,5,5])
		v[:]=5
		x.c_int_alloc_5d = v
		np_test.assert_array_equal(x.c_int_alloc_5d.get(),v)
		y=x.sub_alloc_int_1d_cleanup()
	
	def test_c_int_alloc_1d_large(self):
		# Can have issues exiting when using large (>255) arrays
		y=x.sub_alloc_int_1d_cleanup()
		y=x.sub_alloc_int_1d_arrs()
		v=np.zeros([256])
		v[:]=5
		x.c_int_alloc_1d = v
		np_test.assert_array_equal(x.c_int_alloc_1d.get(),v)
		y=x.sub_alloc_int_1d_cleanup()


	def test_c_real_alloc_1d(self):
		y=x.sub_alloc_real_1d_cleanup()
		y=x.sub_alloc_real_1d_arrs()
		v=np.zeros([5])
		v[:]=1
		np_test.assert_array_equal(x.c_real_alloc_1d.get(),v)
		y=x.sub_alloc_real_1d_cleanup()
	
	def test_c_real_alloc_2d(self):
		y=x.sub_alloc_real_1d_cleanup()
		y=x.sub_alloc_real_1d_arrs()
		v=np.zeros([5,5])
		v[:]=1
		np_test.assert_array_equal(x.c_real_alloc_2d.get(),v)
		y=x.sub_alloc_real_1d_cleanup()
	
	def test_c_real_alloc_3d(self):
		y=x.sub_alloc_real_1d_cleanup()
		y=x.sub_alloc_real_1d_arrs()
		v=np.zeros([5,5,5])
		v[:]=1
		np_test.assert_array_equal(x.c_real_alloc_3d.get(),v)
		y=x.sub_alloc_real_1d_cleanup()
	
	def test_c_real_alloc_4d(self):
		y=x.sub_alloc_real_1d_cleanup()
		y=x.sub_alloc_real_1d_arrs()
		v=np.zeros([5,5,5,5])
		v[:]=1
		np_test.assert_array_equal(x.c_real_alloc_4d.get(),v)
		y=x.sub_alloc_real_1d_cleanup()
	
	def test_c_real_alloc_5d(self):
		y=x.sub_alloc_real_1d_cleanup()
		y=x.sub_alloc_real_1d_arrs()
		v=np.zeros([5,5,5,5,5])
		v[:]=1
		np_test.assert_array_equal(x.c_real_alloc_5d.get(),v)
		y=x.sub_alloc_real_1d_cleanup()
		
	
	def test_c_real_dp_alloc_1d_set(self):
		y=x.sub_alloc_real_dp_1d_cleanup()
		y=x.sub_alloc_real_dp_1d_arrs()
		v=np.zeros([5])
		v[:]=2.0
		x.c_real_dp_alloc_1d = v
		np_test.assert_array_equal(x.c_real_dp_alloc_1d.get(),v)
		y=x.sub_alloc_real_dp_1d_cleanup()
	
	def test_c_real_dp_alloc_2d_set(self):
		y=x.sub_alloc_real_dp_1d_cleanup()
		y=x.sub_alloc_real_dp_1d_arrs()
		v=np.zeros([5,5])
		v[:]=2.0
		x.c_real_dp_alloc_2d = v
		np_test.assert_array_equal(x.c_real_dp_alloc_2d.get(),v)
		y=x.sub_alloc_real_dp_1d_cleanup()
	
	def test_c_real_dp_alloc_3d_set(self):
		y=x.sub_alloc_real_dp_1d_cleanup()
		y=x.sub_alloc_real_dp_1d_arrs()
		v=np.zeros([5,5,5])
		v[:]=2.0
		x.c_real_dp_alloc_3d = v
		np_test.assert_array_equal(x.c_real_dp_alloc_3d.get(),v)
		y=x.sub_alloc_real_dp_1d_cleanup()
	
	def test_c_real_dp_alloc_4d_set(self):
		y=x.sub_alloc_real_dp_1d_cleanup()
		y=x.sub_alloc_real_dp_1d_arrs()
		v=np.zeros([5,5,5,5])
		v[:]=2.0
		x.c_real_dp_alloc_4d = v
		np_test.assert_array_equal(x.c_real_dp_alloc_4d.get(),v)
		y=x.sub_alloc_real_dp_1d_cleanup()
	
	def test_c_real_dp_alloc_5d_set(self):
		y=x.sub_alloc_real_dp_1d_cleanup()
		y=x.sub_alloc_real_dp_1d_arrs()
		v=np.zeros([5,5,5,5,5])
		v[:]=2.0
		x.c_real_dp_alloc_5d = v
		np_test.assert_array_equal(x.c_real_dp_alloc_5d.get(),v)
		y=x.sub_alloc_real_dp_1d_cleanup()
	

	def test_c_real_dp_alloc_1d(self):
		y=x.sub_alloc_real_dp_1d_cleanup()
		y=x.sub_alloc_real_dp_1d_arrs()
		v=np.zeros([5])
		v[:]=1
		np_test.assert_array_equal(x.c_real_dp_alloc_1d.get(),v)
		y=x.sub_alloc_real_dp_1d_cleanup()
	
	def test_c_real_dp_alloc_2d(self):
		y=x.sub_alloc_real_dp_1d_cleanup()
		y=x.sub_alloc_real_dp_1d_arrs()
		v=np.zeros([5,5])
		v[:]=1
		np_test.assert_array_equal(x.c_real_dp_alloc_2d.get(),v)
		y=x.sub_alloc_real_dp_1d_cleanup()
	
	def test_c_real_dp_alloc_3d(self):
		y=x.sub_alloc_real_dp_1d_cleanup()
		y=x.sub_alloc_real_dp_1d_arrs()
		v=np.zeros([5,5,5])
		v[:]=1
		np_test.assert_array_equal(x.c_real_dp_alloc_3d.get(),v)
		y=x.sub_alloc_real_dp_1d_cleanup()
	
	def test_c_real_dp_alloc_4d(self):
		y=x.sub_alloc_real_dp_1d_cleanup()
		y=x.sub_alloc_real_dp_1d_arrs()
		v=np.zeros([5,5,5,5])
		v[:]=1
		np_test.assert_array_equal(x.c_real_dp_alloc_4d.get(),v)
		y=x.sub_alloc_real_dp_1d_cleanup()
	
	def test_c_real_dp_alloc_5d(self):
		y=x.sub_alloc_real_dp_1d_cleanup()
		y=x.sub_alloc_real_dp_1d_arrs()
		v=np.zeros([5,5,5,5,5])
		v[:]=1
		np_test.assert_array_equal(x.c_real_dp_alloc_5d.get(),v)
		y=x.sub_alloc_real_dp_1d_cleanup()
		
	
	def test_c_real_dp_alloc_1d_set(self):
		y=x.sub_alloc_real_dp_1d_cleanup()
		y=x.sub_alloc_real_dp_1d_arrs()
		v=np.zeros([5])
		v[:]=2.0
		x.c_real_dp_alloc_1d = v
		np_test.assert_array_equal(x.c_real_dp_alloc_1d.get(),v)
		y=x.sub_alloc_real_dp_1d_cleanup()
	
	def test_c_real_dp_alloc_2d_set(self):
		y=x.sub_alloc_real_dp_1d_cleanup()
		y=x.sub_alloc_real_dp_1d_arrs()
		v=np.zeros([5,5])
		v[:]=2.0
		x.c_real_dp_alloc_2d = v
		np_test.assert_array_equal(x.c_real_dp_alloc_2d.get(),v)
		y=x.sub_alloc_real_dp_1d_cleanup()
	
	def test_c_real_dp_alloc_3d_set(self):
		y=x.sub_alloc_real_dp_1d_cleanup()
		y=x.sub_alloc_real_dp_1d_arrs()
		v=np.zeros([5,5,5])
		v[:]=2.0
		x.c_real_dp_alloc_3d = v
		np_test.assert_array_equal(x.c_real_dp_alloc_3d.get(),v)
		y=x.sub_alloc_real_dp_1d_cleanup()
	
	def test_c_real_dp_alloc_4d_set(self):
		y=x.sub_alloc_real_dp_1d_cleanup()
		y=x.sub_alloc_real_dp_1d_arrs()
		v=np.zeros([5,5,5,5])
		v[:]=2.0
		x.c_real_dp_alloc_4d = v
		np_test.assert_array_equal(x.c_real_dp_alloc_4d.get(),v)
		y=x.sub_alloc_real_dp_1d_cleanup()
	
	def test_c_real_dp_alloc_5d_set(self):
		y=x.sub_alloc_real_dp_1d_cleanup()
		y=x.sub_alloc_real_dp_1d_arrs()
		v=np.zeros([5,5,5,5,5])
		v[:]=2.0
		x.c_real_dp_alloc_5d = v
		np_test.assert_array_equal(x.c_real_dp_alloc_5d.get(),v)
		y=x.sub_alloc_real_dp_1d_cleanup()
		
	def test_func_return_res(self):
		y=x.func_return_res(2)
		self.assertEqual(y,True)
		y=x.func_return_res(10)
		self.assertEqual(y,False)		
	

	def test_func_assumed_shape_arr_1d(self):
		v=np.zeros([5],dtype='int32')
		v[0]=2.0
		y=x.func_assumed_shape_arr_1d(v)
		self.assertEqual(y,True)
		
	def test_func_assumed_shape_arr_2d(self):
		v=np.zeros([5,5],dtype='int32')
		v[1,0]=2.0
		y=x.func_assumed_shape_arr_2d(v)
		self.assertEqual(y,True)
		
	def test_func_assumed_shape_arr_3d(self):
		v=np.zeros([5,5,5],dtype='int32')
		v[2,1,0]=2.0
		y=x.func_assumed_shape_arr_3d(v)
		self.assertEqual(y,True)
		
	def test_func_assumed_shape_arr_4d(self):
		v=np.zeros([5,5,5,5],dtype='int32')
		v[3,2,1,0]=2.0
		y=x.func_assumed_shape_arr_4d(v)
		self.assertEqual(y,True)
		
	def test_func_assumed_shape_arr_5d(self):
		v=np.zeros([5,5,5,5,5],dtype='int32')
		v[4,3,2,1,0]=2.0
		y=x.func_assumed_shape_arr_5d(v)
		self.assertEqual(y,True)
		
		
	def test_func_assumed_size_arr_1d(self):
		v=np.zeros([5],dtype='int32')
		v[1]=2
		y=x.func_assumed_size_arr_1d(v)
		self.assertEqual(y,True)
		
	def test_func_assumed_size_arr_real_1d(self):
		v=np.zeros([5],dtype='float32')
		v[1]=2.0
		y=x.func_assumed_size_arr_real_1d(v)
		self.assertEqual(y,True)
		
	def test_func_assumed_size_arr_real_dp_1d(self):
		v=np.zeros([5],dtype='float64')
		v[1]=2.0
		y=x.func_assumed_size_arr_real_dp_1d(v)
		self.assertEqual(y,True)
		
	def test_sub_alloc_arr_1d(self):
		v=0
		y=x.sub_alloc_arr_1d(v)
		vTest=np.zeros(10)
		vTest[:]=10
		np_test.assert_array_equal(y['x'],vTest)
		
	def test_sub_dt_in_s_simple(self):
		with captured_output() as (out,err):
			y=x.sub_f_simple_in({'x':1,'y':10})
		output=out.getvalue().strip()
		o=' '.join([str(i) for i in [1,10]])
		self.assertEqual(output,o)
	
	def test_sub_dt_out_s_simple(self):
		with captured_output() as (out,err):
			y=x.sub_f_simple_out({})
		output=out.getvalue().strip()
		self.assertEqual(y['x'],{'x':1,'y':10})	
	
	def test_sub_dt_inout_s_simple(self):
		with captured_output() as (out,err):
			y=x.sub_f_simple_inout({'x':5,'y':3})
		output=out.getvalue().strip()
		o='  '.join([str(i) for i in [5,3]])
		self.assertEqual(output,o)
		self.assertEqual(y['zzz'],{'x':1,'y':10})
		
	def test_sub_dt_inoutp_s_simple(self):
		with captured_output() as (out,err):
			y=x.sub_f_simple_inoutp({'x':5,'y':3})
		output=out.getvalue().strip()
		o='  '.join([str(i) for i in [5,3]])
		self.assertEqual(output,o)
		self.assertEqual(y['zzz'],{'x':1,'y':10})
		
	def test_sub_int_p(self):
		with captured_output() as (out,err):
			y=x.sub_int_p(1)
		output=out.getvalue().strip()
		self.assertEqual(output,'1')
		self.assertEqual(y['zzz'],5)

	def test_sub_real_p(self):
		with captured_output() as (out,err):
			y=x.sub_real_p(1.0)
		output=out.getvalue().strip()
		self.assertEqual(output,'1.00')
		self.assertEqual(y['zzz'],5.0)
		
	def test_sub_str_p(self):
		with captured_output() as (out,err):
			y=x.sub_str_p('abcdef')
		output=out.getvalue().strip()
		self.assertEqual(output,'abcdef')
		self.assertEqual(y['zzz'],'xyzxyz')
		
	def test_sub_arr_exp_p(self):
		v=np.arange(0,5)
		o=' '.join([str(i) for i in v.flatten()])
		with captured_output() as (out,err):
			y=x.sub_exp_array_int_1d(v)
		output=out.getvalue().strip()	
		self.assertEqual(output,o.strip())
	
	#@unittest.skip("Skipping")	
	def test_sub_arr_assumed_rank_int_1d(self):
		v=np.arange(10.0,15.0)
		o=' '.join([str(i) for i in v.flatten()])
		with captured_output() as (out,err):
			y=x.sub_arr_assumed_rank_int_1d(v)
		output=out.getvalue().strip()	
		np_test.assert_array_equal(y['zzz'],np.array([100.0]*5))
	
	def test_sub_opt(self):
		with captured_output() as (out,err):
			y=x.sub_int_opt(1)
		output=out.getvalue().strip()
		self.assertEqual(output,'100')
		with captured_output() as (out,err):
			y=x.sub_int_opt()
		output=out.getvalue().strip()
		self.assertEqual(output,'200')
	
	def test_dt_copy(self):
		x.f_struct_simple.x=99
		x.f_struct_simple.y=99
		y=x.f_struct_simple.get()
		self.assertEqual(y,{'x':99,'y':99})
		y=x.f_struct_simple.get(copy=True)
		self.assertEqual(y,{'x':99,'y':99})
		y=x.f_struct_simple.get(copy=False)
		self.assertEqual(y.x,99)
		self.assertEqual(y.y,99)
		
		
	def test_second_mod(self):
		x.f_struct_simple2.x=99
		y=x.sub_use_mod()
		self.assertEqual(x.test2_x.get(),1)
		self.assertEqual(x.f_struct_simple2.get(),{'x':5,'y':6,'z':0})
	
	def test_nested_dts(self):
		x.g_struct.a_int=10
		self.assertEqual(x.g_struct.a_int,10)
		x.g_struct={'a_int':10,'f_struct':{'a_int':3}}
		self.assertEqual(x.g_struct.f_struct.a_int,3)
		x.g_struct.f_struct.a_int=8
		self.assertEqual(x.g_struct.f_struct.a_int,8)
		y=x.func_check_nested_dt()
		self.assertEqual(y,True)
	
	def test_logical_arr(self):
		xarr=np.zeros(10)
		x2arr=np.zeros(10)
		x2arr[:]=False
		xarr[:]=True
		
		y=x.func_alltrue_arr_1d(xarr)
		y2=x.func_allfalse_arr_1d(x2arr)
		self.assertEqual(y,True)
		self.assertEqual(y2,True)


	def test_logical_arr_multi(self):
		xarr=np.zeros(5)
		xarr[:]=True
		
		y=x.func_logical_multi(1.0,2.0,xarr,3.0,4.0)
		self.assertEqual(y,True)

	def test_func_set_f_struct(self):
		y = x.func_set_f_struct()
		self.assertEqual(y,True)
		
		self.assertEqual(x.f_struct.a_int,5)
		self.assertEqual(x.f_struct.a_int_lp,6)
		self.assertEqual(x.f_struct.a_real,7.0)
		self.assertEqual(x.f_struct.a_real_dp,8.0)		
		
		v=np.array([9,10,11,12,13],dtype='int32')
		np_test.assert_array_equal(x.f_struct.b_int_exp_1d,v)
		
		# v=np.array([1,2,3,4,5,6,7,8,9,10],dtype='int32')
		# np_test.assert_array_equal(x.f_struct.c_int_alloc_1d,v)
		
		# v=np.array([9,10,11,12,13],dtype='int32')
		# np_test.assert_array_equal(x.f_struct.d_int_point_1d)		
		
if __name__ == '__main__':
	unittest.main() 

	# pahole -aAdEIpr tester.so |& tee tester.pahole	



