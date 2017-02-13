import ctypes
import pickle
import parseMod as pm
import numpy as np

		
def find_key_val(list_dicts,key,value):
	for idx,i in enumerate(list_dicts):
		if i[key]==value:
			return idx		
			
			
#class fUtils(object):
	#pass
			
#class fVar(fUtils):
	#def __init__(self,attr):
		#pass
		
	#def from_param(self,x):
		#return self.ctype(x)
		
		
#class fStr(fVar):
	#def __init__(self,attr):
		#self.ctype=ctypes.c_char_p
	
	
##Inheriet from object or maybe fVar?
#class fExplicitArray(fVar):
	#def __init__(self,attr):
		#pass
	
#class fDummyArray(fVar):
	#_GFC_MAX_DIMENSIONS=7
	
	#_GFC_DTYPE_RANK_MASK=0x07
	#_GFC_DTYPE_TYPE_SHIFT=3
	#_GFC_DTYPE_TYPE_MASK=0x38
	#_GFC_DTYPE_SIZE_SHIFT=6
	
	#_BT_UNKNOWN = 0
	#_BT_INTEGER=_BT_UNKNOWN+1 
	#_BT_LOGICAL=_BT_INTEGER+1
	#_BT_REAL=_BT_LOGICAL+1
	#_BT_COMPLEX=_BT_REAL+1
	#_BT_DERIVED=_BT_COMPLEX+1
	#_BT_CHARACTER=_BT_DERIVED+1
	#_BT_CLASS=_BT_CHARACTER+1
	#_BT_PROCEDURE=_BT_CLASS+1
	#_BT_HOLLERITH=_BT_PROCEDURE+1
	#_BT_VOID=_BT_HOLLERITH+1
	#_BT_ASSUMED=_BT_VOID+1	
	
	#_index_t = ctypes.c_int64
	#_size_t = ctypes.c_int64
	#def __init__(self,attr):
		#pass
	
#class fDerivedType(fVar):	
	#def __init__(self,attr):
		#pass
	
#class fFunc(fVar):
	#def __init__(self,attr):
		#pass
			
		
class fFort(object):
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
	
	
	def __init__(self,libname,fpy):
		self.lib=ctypes.CDLL(libname)
		self.libname=libname
		self.fpy=fpy

		with open(self.fpy,'rb') as f:
			self.version=pickle.load(f)
			if self.version ==1:
				self._mod_data=pickle.load(f)
				self._mod_vars=pickle.load(f)
				self._param=pickle.load(f)
				self._funcs=pickle.load(f)
				self._dt_defs=pickle.load(f)

	def _init_mod_var(self):
		for i in self._mod_vars:
			self._init_var(i)
			
	def _init_param(self):
		for i in self._param:
			self._init_var(i)
		
	def _init_func(self,obj):
		obj['argparse']=[]
		self._init_var(obj)
		for i in obj['args']:
			self._init_var(i)
			obj['argparse'].append(i['_ctype'])
		obj['_call']=self._get_from_lib(obj)	
		obj['_call'].argparse=obj['argparse']
		obj['_call'].restype=obj['_ctype']
		
	def _init_array(self,obj):
		obj['array']['_ctype']=obj['_ctype']
		self._get_array_ctype(obj)
	
	def _init_dt(self,obj):
		for i in obj['args']:
			self._get_dt_ctype(i)
		
	def _init_array_dt(self,obj):
		pass
		
	def _get_ctype(self,obj):
		if 'intent' not in obj.keys():
			obj['_ctype']=getattr(ctypes,obj['ctype'])
		elif obj['intent']=="in":
			obj['_ctype']=getattr(ctypes,obj['ctype'])
		elif obj['intent']=="out" or obj['intent']=="inout" or obj['pointer']:
			obj['_ctype']=ctypes.POINTER(getattr(ctypes,obj['ctype']))
			
	def _get_pytype(self,obj):
		if obj['pytype']=='void':
			obj['_pytype']=None
		else:
			obj['_pytype']=getattr(__builtin__,obj['pytype'])

	def _init_var(self,obj):
		self._get_ctype(obj)
		self._get_pytype(obj)
		
		if 'array' in obj.keys() and 'dt' in obj.keys():
			if obj['array'] and obj['dt']:
					self._init_array_dt(obj)
					return
					
		if 'array' in obj.keys():
			if obj['array']:
				self._init_array(obj)
				return
				
		if 'dt' in obj.keys():
			if obj['dt']:
				self._init_dt(obj)
				return
		
	def _set_var(self,value,obj):
		res=self._get_from_lib(obj)
		self._var_to_ctype(res,value,obj)

	def _set_param(self,value,obj):
		raise AttributeError("Can't alter a parameter")
		
	def _get_var(self,obj):
		res=self._get_from_lib(obj)
		return self._ctype_to_var(res,obj)
		
	def _get_param(self,obj):
		return obj['value']
		
	#Module variables
	def _set_array(self,value,obj):
		res=self._get_from_lib(obj)
		if 'explicit' in obj['array']['atype']:
			array=self._set_explicit_array(res,value,obj)
		else:
			pass
		
	def _get_array(self,obj):
		res=self._get_from_lib(obj)
		if 'explicit' in obj['array']['atype']:
			array=self._get_explicit_array(res,obj)
		else:
			pass
		return array

	def _call(self,name,*args):
		#find function in self._funcs		
		for f in self._funcs:
			if f['name']==name:
				break

		#Convert args to ctype versions
		args_in=[]
		for i,j in zip(*args,f['args']):
			args_in.append(self.arg_to_ctype(i,j))

		

		#Call function
		res=f['_call']()
		
		#Convert back any args that changed:
		args_out=[]
		for i,j in zip(args_in,f['args']):
			args_out.append(self.ctype_to_py(i,j))
			
		return res,args_out		


	def _arg_to_ctype(self,value,obj):
		if obj['array'] and obj['dt']:
			self._array_dt_to_ctype(value,obj)
		elif obj['array']:
			self._array_to_ctype(value,obj)
		elif obj['dt']:
			self._dt_to_ctype(value,obj)
		else:
			self._var_to_ctype(value,obj)
		
	def _ctype_to_py(self,value,obj):
		if obj['array'] and obj['dt']:
			self._ctype_to_array_dt(value,obj)
		elif obj['array']:
			self._ctype_to_array(value,obj)
		elif obj['dt']:
			self._ctype_to_dt(value,obj)
		else:
			self._ctype_to_array(value,obj)
			

	def _var_to_ctype(self,ctyp,value,obj):
		if obj['pytype'] == 'str':
			self._set_char_str(ctyp,value,obj)
		else:
			ctyp.value=obj['_pytype'](value)

	def _array_dt_to_ctype(self,value,obj):
		pass
		
	def _array_to_ctype(self,value,obj):
		self.__array_to_ctype(value,obj)
		
	def _dt_to_ctype(self,value,obj):
		pass
		
		
	def _ctype_to_var(self,value,obj):
		if obj['pytype'] == 'str':
			x=self._get_string_by_name(value,obj)
		else:
			x=obj['_pytype'](value.value)
		return x

	def _ctype_to_array_dt(self,value,obj):
		pass
		
	def _ctype_to_array(self,value,obj):
		self.__ctype_to_array(value,obj)
		
	def _ctype_to_dt(self,value,obj):
		pass
		
	def _get_from_lib(self,obj):
		res=None
		try:
			res=obj['_ctype'].in_dll(self.lib,obj['mangled_name'])
		except (ValueError, AttributeError):
			print("Cant find "+obj['name'])
		return res
		
	def _get_string_by_name(self,value,obj):
		""" Gets a string"""
		base_address=ctypes.addressof(value)
		return self._get_string_from_address(base_address)

	def _get_string_from_address(self,ctype_address,debug=False):
		out=''
		i=0
		while True:
			x=ctypes.c_char.from_address(ctype_address+i)
			if debug:
				print(x.value,i,x.value==b'\x00')
			if x.value == b'\x00':
				break
			else:
				out=out+(x.value).decode()
				i=i+ctypes.sizeof(ctypes.c_char)
		return out	
		
	def _get_explicit_array(self,res,obj):
		shape=self._make_array_shape(obj)
		array=[]
		k=0
		base_address=ctypes.addressof(res)
		for i in range(len(shape)):
			for j in range(shape[i]):
				offset=base_address+k*ctypes.sizeof(obj['_ctype'])
				array.append(obj['_ctype'].from_address(offset).value)
				k=k+1
		return np.reshape(array,newshape=shape)

	def _set_explicit_array(self,res,value,obj):
		shape=self._make_array_shape(obj)
		k=0
		base_address=ctypes.addressof(res)
		flatarray=value.flatten()
		for i in range(len(shape)):
			for j in range(shape[i]):
				offset=base_address+k*ctypes.sizeof(obj['_ctype'])
				obj['_ctype'].from_address(offset).value=flatarray[k]
				k=k+1

	def _set_char_str(self,res,value,obj):
		base_address=ctypes.addressof(res)
		for j in range(obj['char_len']):
			offset=base_address+j*ctypes.sizeof(ctypes.c_char)
			obj['_ctype'].from_address(offset).value=value[j].encode()


	def _make_array_shape(self,obj):
		bounds=obj['array']['bounds']
		shape=[]
		for i,j in zip(bounds[0::2],bounds[1::2]):
			shape.append(j-i+1)
		return shape
		

	def _get_array_ctype(self,obj):
		arr=obj['array']
		if 'explicit' not in arr['atype']:
			obj['_ctype']=self.__make_array_ctype(arr['ndims'])
			#Place to store the array after ctytpe-ifed
			obj['_array']=obj['_ctype']()
			#Initilize default values for this array
			self.__init_array_ctype(obj)
		
	def _make_array_ctype(self,ndim):
		class descriptor(ctypes.Structure):
			_fields_=[("stride",self._index_t),
					("lbound",self._index_t),
					("ubound",self._index_t)]
		
		class defarray(ctypes.Structure):
			_fields_=[("base_addr",ctypes.c_void_p),
					("offset",self._size_t),
					("dtype",self._index_t),
					("dims",descriptor*ndim)]
					
		return defarray	
		
	def _make_dt_ctype(self,obj,dt_defs):
		class dt(ctypes.Structure):
			def __init__(self,lnames,lctypes):
				self._fields_=[]
				for k in obj['args']:
					for i,j in zip():
						if k['dt']:
							#Match to the list of dt's
							for l in dt_defs:
								if k['name']==l['name']:
									c=ctypes.POINTER(self._make_dt_ctype(l))
						else:
							c=k['_ctype']
						self._fields_.append((k['name'],c))
		return dt
		
	def _array_to_ctype(self,array,obj):
		obj['_arrary'].base_addr=array.ctypes.data_as(ctypes.c_void_p)
		
		obj['_arrary'].offset=self.size_t(-1)
		
		obj['_arrary'].dtype=self._get_dtype(obj)
				
		for i in range(0,obj['array']['ndim']):
			#Not sure if bounds are actually needed?
			obj['_arrary'].dims[i].stride=self._index_t(array.strides[i]//ctypes.sizeof(obj['array']['_ctype']))
			obj['_arrary'].dims[i].lbound=self._index_t(1)
			obj['_arrary'].dims[i].ubound=self._index_t(array.shape[i])
				
	def _get_dtype(self,obj):
		ftype=self._get_ftype(obj)
		dtype=obj['array']['ndims']
		dtype=dtype|(ftype<<self.GFC_DTYPE_TYPE_SHIFT)
		dtype=dtype|(ctypes.sizeof(obj['array']['_ctype'])<<self.GFC_DTYPE_SIZE_SHIFT)
		return dtype

	def _init_array_ctype(self,obj):		
		obj['_array'].base_addr=ctypes.c_void_p()
		
		obj['_array'].offset=self._size_t(-1)
		obj['_array'].dtype=0
				
		for i in range(0,obj['array']['ndims']):
			obj['_array'].dims[i].stride=self._index_t(1)
			obj['_array'].dims[i].lbound=self._index_t(1)
			obj['_array'].dims[i].ubound=self._index_t(1)	


	def _get_ftype(self,obj):
		dtype=obj['ctype']	
		if 'c_int' in dtype:
			ftype=self._BT_INTEGER
		elif 'c_double' in dtype or 'c_real' in dtype:
			ftype=self._BT_REAL
		elif 'c_bool' in dtype:
			ftype=self._BT_LOGICAL
		elif 'c_char' in dtype:
			ftype=self._BT_CHARACTER
		else:
			raise ValueError("Cant match dtype, got "+dtype)
		return ftype


	def _ctype_to_array(self,value,obj):

		sizebytes=ctypes.sizeof(obj['array']['_ctype'])
		
		shape=[]
		stride=[]
		for i in range(0,obj['array']['ndims']):
			shape.append((value.dims[i].ubound-
						value.dims[i].lbound)+1)
			stride.append(value.dims[0].stride*sizebytes)
		
		off=0
		arr=[]
		for i in range(np.product(shape)):
				off=i*stride[0]
				arr.append(ctype.from_address(value.base_addr+off).value)

		#Copy array data
		array=np.reshape(arr,newshape=shape)
		return array


x=fFort('./test_mod.so','test_mod.fpy')

num=find_key_val(x._mod_vars,'name','x_str')
x._init_var(x._mod_vars[num])
print(x._get_var(x._mod_vars[num]))
x._set_var('abcdefgdet',x._mod_vars[num])
print(x._get_var(x._mod_vars[num]))

num=find_key_val(x._mod_vars,'name','xp')
x._init_var(x._mod_vars[num])
print(x._get_var(x._mod_vars[num]))
x._set_var(2,x._mod_vars[num])
print(x._get_var(x._mod_vars[num]))

num=find_key_val(x._mod_vars,'name','aaa')
x._init_var(x._mod_vars[num])
print(x._get_array(x._mod_vars[num]))
x._set_array(np.array([5,6,7,8]),x._mod_vars[num])
print(x._get_array(x._mod_vars[num]))


num=find_key_val(x._funcs,'name','func_noargs')
#x._init_func(x._funcs[num])
#x._call(x._funcs[num])
