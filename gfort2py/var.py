try:
	import __builtin__
except ImportError:
	import builtins as __builtin__
	
import ctypes	
import numpy as np
	
				
class fVar(object):
	def __init__(self,lib,obj):
		self.__dict__.update(obj)
		self.lib=lib
		self._ctype=self.ctype_def()
		self._ctype_f=self.ctype_def_func()
		self._pytype=self.pytype_def()
		
	def py_to_ctype(self,value):
		"""
		Pass in a python value returns the ctype representation of it
		"""
		return self._cytype(value)
	
	def ctype_to_py(self,value):
		"""
		Pass in a ctype value returns the python representation of it
		"""
		return self._pytype(value.value)
		
	def pytype_def(self):
		return getattr(__builtin__,self.pytype)
	
	def ctype_def(self):
		"""
		The ctype type of this object
		"""
		return getattr(ctypes,self.ctype)
		
	def ctype_def_func(self):
		"""
		The ctype type of a value suitable for use as an argument of a function
		
		May just call ctype_def
		"""
		c=None
		if 'intent' not in self.__dict__.keys():
			c=getattr(ctypes,self.ctype)
		elif self.intent=="out" or self.intent=="inout" or self.pointer:
			c=ctypes.POINTER(self.ctype_def())
		else:
			c=getattr(ctypes,self.ctype)
		return c
	
	def set_mod(self,value):
		"""
		Set a module level variable
		"""
		r=self._get_from_lib()
		r.value=self._pytype(value)
	
	def get_mod(self):
		"""
		Get a module level variable
		"""
		r=self._get_from_lib()
		return self.ctype_to_py(r)
		
	def _get_from_lib(self):
		res=None
		try:
			res=self._ctype.in_dll(self.lib,self.mangled_name)
		except AttributeError:
			raise
		return res
		
		
	def _get_var_by_iter(self,value,size=-1):
		""" Gets a varaible where we have to iterate to get multiple elements"""
		base_address=ctypes.addressof(value)
		return self._get_var_from_address(base_address,size=size)

	def _get_var_from_address(self,ctype_address,size=-1):
		out=[]
		i=0
		sof=ctypes.sizeof(self._ctype)
		while True:
			if i==size:
				break
			x=self._ctype.from_address(ctype_address+i*sof)
			if x.value == b'\x00':
				break
			else:
				out.append(x.value)
			i=i+1
		return out
	

	def _set_var_from_iter(self,res,value,size=99999):
		base_address=ctypes.addressof(res)
		self._set_var_from_address(base_address,value,size)
		
	def _set_var_from_address(self,ctype_address,value,size=99999):
		for j in range(min(len(value),size)):
			offset=ctype_address+j*ctypes.sizeof(self._ctype)
			self._ctype.from_address(offset).value=value[j]	

class fParam(fVar):
  	def set_mod(self,value):
  		"""
  		Cant set a parameter
  		"""
  		raise ValueError("Can't alter a parameter")
  	
  	def get_mod(self):
  		"""
  		A parameters value is stored in the dict, as we cant access them 
  		from the shared lib.
  		"""
  		return self._pytype(self.value)
