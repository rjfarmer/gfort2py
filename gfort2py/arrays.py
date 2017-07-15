import ctypes
from .var import fVar,fParam	
import numpy as np
	
class fExplicitArray(fVar):
	def __init__(self,lib,obj):
		self.__dict__.update(obj)
		self._lib=lib
		self._pytype=np.array
		self._ctype=self.ctype_def()
		self._ctype_f=self.ctype_def_func()
	
	def ctype_to_py(self,value):
		"""
		Pass in a ctype value returns the python representation of it
		"""
		return self._get_var_by_iter(value,self._array_size())
	
	def set_mod(self,value):
		"""
		Set a module level variable
		"""
		r=self._get_from_lib()
		v=value.flatten(order='C')
		self._set_var_from_iter(r,v,self._array_size())
	
	def get(self):
		"""
		Get a module level variable
		"""
		r=self._get_from_lib()
		s=self.ctype_to_py(r)
		shape=self._make_array_shape()
		return np.reshape(s,shape)
	
	def _make_array_shape(self):
		bounds=self.array['bounds']
		shape=[]
		for i,j in zip(bounds[0::2],bounds[1::2]):
			shape.append(j-i+1)
		return shape	
	
	def _array_size(self):
		return np.product(self._make_array_shape())

class fDummyArray(fVar):
	_GFC_MAX_DIMENSIONS=7
	
	_GFC_DTYPE_RANK_MASK=0x07
	_GFC_DTYPE_TYPE_SHIFT=3
	_GFC_DTYPE_TYPE_MASK=0x38
	_GFC_DTYPE_SIZE_SHIFT=6
	
	_BT_UNKNOWN = 0
	_BT_INTEGER=_BT_UNKNOWN+1 
	_BT_LOGICAL=_BT_INTEGER+1
	_BT_REAL=_BT_LOGICAL+1
	_BT_COMPLEX=_BT_REAL+1
	_BT_DERIVED=_BT_COMPLEX+1
	_BT_CHARACTER=_BT_DERIVED+1
	_BT_CLASS=_BT_CHARACTER+1
	_BT_PROCEDURE=_BT_CLASS+1
	_BT_HOLLERITH=_BT_PROCEDURE+1
	_BT_VOID=_BT_HOLLERITH+1
	_BT_ASSUMED=_BT_VOID+1	
	
	_index_t = ctypes.c_int64
	_size_t = ctypes.c_int64
	def __init__(self,lib,obj):
		self.__dict__.update(obj)
		self._lib=lib
		
		self.ndim=self.array['ndims']
		self._make_array_desc()
		
	def _make_array_desc(self):
	
		self._ctype=fDerivedTypeDesc()
		self._ctype.set_fields(['base_addr','offset','dtype'],
							  [ctypes.c_void_p,self._size_t,self._index_t])
	
		self._dims=fDerivedTypeDesc()
		self._dims.set_fields=(['stride','lbound','ubound'],
							   [self._index_t,self._index_t,self._index_t])
							  
		self._ctype.add_arg(['dims',self._dims*self.ndim])	


	def _set_array(self,value):
		r=self._get_from_lib()
		
		
	def _get_array(self):
		r=self._get_from_lib()
		

		
class fParamArray(fParam):
	def get(self):
		"""
		A parameters value is stored in the dict, as we cant access them 
		from the shared lib.
		"""
		return np.array(self.value,dtype=self.pytype)
