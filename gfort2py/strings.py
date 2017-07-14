import ctypes
from .var import fVar	
	
	
class fStr(fVar):
	def __init__(self,lib,obj):
		self.__dict__.update(obj)
		self._lib=lib
		self._ctype=ctypes.c_char
		self._ctype_f=ctypes.c_char_p
		self._pytype=str
	
	
	def py_to_ctype(self,value):
		"""
		Pass in a python value returns the ctype representation of it
		"""
		return ctypes.c_char(value.encode())
	
	def ctype_to_py(self,value):
		"""
		Pass in a ctype value returns the python representation of it
		"""
		return self._get_var_by_iter(value,self.char_len)
		
	def ctype_def_func(self):
		"""
		The ctype type of a value suitable for use as an argument of a function
		
		May just call ctype_def
		"""
		return self._ctype_f
	
	def set_mod(self,value):
		"""
		Set a module level variable
		"""
		r=self._get_from_lib()
		self._set_var_from_iter(r,value.encode(),self.char_len)
	
	def get(self):
		"""
		Get a module level variable
		"""
		r=self._get_from_lib()
		s=self.ctype_to_py(r)
		return ''.join([i.decode() for i in s])
	
	def __len__(self):
		return len(self.get())
		
	def __add__(self,other):
		return 	NotImplemented

	def __sub__(self,other):
		return 	NotImplemented
		
	def __mul__(self,other):
		return 	NotImplemented
		
	def __truediv__(self,other):
		return 	NotImplemented
		
	def __pow__(self,other,modulo=None):
		return NotImplemented
