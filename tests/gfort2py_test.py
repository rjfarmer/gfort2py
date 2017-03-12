import os, sys
import numpy as np
import gfort2py as gf
import unittest
import subprocess
import numpy.testing as np_test

os.chdir('tests')
subprocess.check_output(["make"])
x=gf.fFort('./tester.so','tester.mod',reload=True)


class TestStringMethods(unittest.TestCase):
	def test_a_str(self):
		v='123456798'
		x.a_str=v
		self.assertEqual(x.a_str,v)
		
	def test_a_str_bad_length(self):
		v='132456789kjhgjhf'
		x.a_str=v
		self.assertEqual(x.a_str,v[0:10])
		
	def test_a_int(self):
		v=1
		x.a_int=v
		self.assertEqual(x.a_int,v)
		
	def test_a_int_str(self):
		with self.assertRaises(ValueError) as cm:
			x.a_int='abc'
			
	def test_a_real(self):
		v=1.0
		x.a_real=v
		self.assertEqual(x.a_real,v)
	
	def test_a_real_str(self):	
		with self.assertRaises(ValueError) as cm:
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
		
	def test_const_real_qp(self):	
		self.assertEqual(x.const_real_qp,1.0)

	def test_const_int_arr(self):	
		np_test.assert_array_equal(x.const_int_arr,np.array([1,2,3,4,5,6,7,8,9,0]))

	def test_const_real_arr(self):	
		np_test.assert_array_equal(x.const_int_arr,np.array([1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,0.0]))

if __name__ == '__main__':
	unittest.main() 



