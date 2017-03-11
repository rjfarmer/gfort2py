import numpy as np
import gfort2py as gf
import unittest

x=gf.fFort('./tester.so','tester.mod',reload=True)

x.a_str
x.a_str='132456789'
x.a_str



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

if __name__ == '__main__':
	unittest.main() 



