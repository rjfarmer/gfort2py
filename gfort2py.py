import ctypes
import pickle
import parseMod as pm
		
#class fArray(object):
	##GCC constants
	#GFC_MAX_DIMENSIONS=7
	
	#GFC_DTYPE_RANK_MASK=0x07
	#GFC_DTYPE_TYPE_SHIFT=3
	#GFC_DTYPE_TYPE_MASK=0x38
	#GFC_DTYPE_SIZE_SHIFT=6
	
	#BT_UNKNOWN = 0
	#BT_INTEGER=BT_UNKNOWN+1 
	#BT_LOGICAL=BT_INTEGER+1
	#BT_REAL=BT_LOGICAL+1
	#BT_COMPLEX=BT_REAL+1
	#BT_DERIVED=BT_COMPLEX+1
	#BT_CHARACTER=BT_DERIVED+1
	#BT_CLASS=BT_CHARACTER+1
	#BT_PROCEDURE=BT_CLASS+1
	#BT_HOLLERITH=BT_PROCEDURE+1
	#BT_VOID=BT_HOLLERITH+1
	#BT_ASSUMED=BT_VOID+1	
	
	#index_t = ctypes.c_int64
	#size_t = ctypes.c_int64
	
	#def __init__(self):
		#pass
		
	#def _set_pyarray(self,array,assumed=True):
		#self.array=array
		#self.dtype=self._get_dtype()
		#self.ftype,self._ctype=self._get_type()
		#self.ndim=self.array.ndim
		#self.shape=self.array.shape
		#self.lbound=np.array([1]*self.ndim)
		#self.ubound=np.array(self.shape)
	
		#if assumed:
			#self._desc=_build_desc()
			#self.ctype=ctypes.POINTER(self._desc)
			#self._build_array()
		#else:
			#self.ctype=ctypes.c_void_p
			#self._array(self.array.data_as(ctypes.c_void_p))
	
		
	#def _get_farray(self,assumed=True,**kwargs):
		#pass
		
	#def _make_desc(self,ndim):
		#class descriptor(ctypes.Structure):
			#_fields_=[("stride",self.index_t),
					#("lbound",self.index_t),
					#("ubound",self.index_t)]
		
		#class defarray(ctypes.Structure):
			#_fields_=[("base_addr",ctypes.c_void_p),
					#("offset",self.size_t),
					#("dtype",self.index_t),
					#("dims",descriptor*ndim)]
					
		#return defarray	
		
	#def _get_type(self):
		#dtype=self.array.dtype.kind		
		#if dtype=='i':
			#ftype=self.BT_INTEGER
			#ctype=ctypes.c_int32
		#elif dtype=='f':
			#ftype=self.BT_REAL
			#ctype=ctypes.c_double
		#elif dtype=='b':
			#ftype=self.BT_LOGICAL
			#ctype=ctypes.c_bool
		#elif dtype=='U' or dtype=='S':
			#ftype=self.BT_CHARACTER
			#ctype=ctypes.c_char
		#else:
			#raise ValueError("Cant match dtype, got "+dtype)
		
		#return ftype,ctype
				
	#def _get_dtype(self):
		#ftype,ctype=self._get_type()
		#dtype=self.array.ndim
		#dtype=dtype|(ftype<<self.GFC_DTYPE_TYPE_SHIFT)
		#dtype=dtype|(ctypes.sizeof(ctype)<<self.GFC_DTYPE_SIZE_SHIFT)
		#return dtype


	#def _find_dtype(self,dtype):
		#rank=dtype&self.GFC_DTYPE_RANK_MASK
		#ftype=(dtype&self.GFC_DTYPE_TYPE_MASK)>>self.GFC_DTYPE_TYPE_SHIFT
		#sizebytes=dtype>>self.GFC_DTYPE_SIZE_SHIFT
		#return rank,ftype,sizebytes
		
	#def _get_pytype(self,ftype,sizebytes):
		#if ftype==self.BT_INTEGER:
			#ctype=pm.get_ctype_int(sizebytes)
			#pytype=int
		#elif ftype==self.BT_REAL:
			#ctype=pm.get_ctype_float(sizebytes)
			#pytype=float
		#elif dtype==self.BT_LOGICAL:
			#pytype=np.bool
			#ctype=ctypes.c_bool
		#elif ftype==self.BT_CHARACTER:
			#pytype=np.char
			#ctype=ctypes.c_char
		#else:
			#raise ValueError("Cant match ftype, got "+ftype)		
		
		#return pytype,ctype

	#def _build_desc(self):
		#self._desc=self._make_desc(self._ndim)
		
		#self._array=self._desc()
		
		#self._array.base_addr=ctypes.c_void_p()
		
		#self._array.offset=self.size_t(-1)
		
		#self._array.dtype=0
				
		#for i in range(0,ndim):
			#self._array.dims[i].stride=self.index_t(1)
			#self._array.dims[i].lbound=self.index_t(1)
			#self._array.dims[i].ubound=self.index_t(1)	

	#def _build_array(self):
		#self._arrary.base_addr=self.array.ctypes.data_as(ctypes.c_void_p)
		
		#self._arrary.offset=self.size_t(-1)
		
		#self._arrary.dtype=dtype
				
		#for i in range(0,self._ndim):
			#self._arrary.dims[i].stride=self.index_t(self.array.strides[i]//ctypes.sizeof(ctype))
			#self._arrary.dims[i].lbound=self.index_t(1)
			#self._arrary.dims[i].ubound=self.index_t(self.array.shape[i])	
			
			
	#def ctype2array(self,carray=None):
		#if carray is None:
			##We passed the array so fortran filled our memory allready
			#return self.array
			
		#self.rank,self.ftype,self.sizebytes=self._find_dtype(carray.dtype)	
			
		#shape=[]
		#stride=[]
		#for i in range(0,ndim):
			#shape.append((carray.dims[i].ubound-
						#carray.dims[i].lbound)+1)
			#stride.append(carray.dims[0].stride*self.sizebytes)
		
		#off=0
		#arr=[]
		#for i in range(np.product(shape)):
				#off=i*stride[0]
				#arr.append(ctype.from_address(carray.base_addr+off).value)

		##Copy array data
		#self.array=np.reshape(arr,newshape=shape)	
		
	#def pass2func(self,func_name):
		#self._build_desc()
		#self._build_array()
		#func=getattr(lib,func_name)
		#func.argtypes=[self._ctype]
		#func(self._array)

	#def getFixedArray(self,name,ctype,size):
		##When we have a fixed sized array as a module varaiable
		#x=self.get_from_lib(name)
		#address=ctypes.addressof(x)
		#sizeof=ctypes.sizeof(self.ctype)
		#a=[]
		#for i in range(size):
			#a.append(ctypes.ctype.from_address(address))
			#address=adress+sizeof
		#return np.array(a)
		
	#def _get_array(self,array_in):			
		#self.ndims,self.ftype,self.sizebytes=self._find_dtype(array_in.dtype)	
			
		#shape=[]
		#stride=[]
		#for i in range(0,self.ndim):
			#shape.append((array_in.dims[i].ubound-
						#array_in.dims[i].lbound)+1)
			#stride.append(array_in.dims[0].stride*self.sizebytes)
		
		#off=0
		#arr=[]
		#for i in range(np.product(shape)):
				#off=i*stride[0]
				#arr.append(ctype.from_address(array_in.base_addr+off).value)

		##Copy array data
		#self.array=np.reshape(arr,newshape=shape)
	
#class fFunc(fUtils):
	##GCC constants
	#_GFC_MAX_DIMENSIONS=7
	
	#_GFC_DTYPE_RANK_MASK=0x07
	#_GFC_DTYPE_TYPE_SHIFT=3
	#_GFC_DTYPE_TYPE_MASK=0x38
	#_GFC_DTYPE_SIZE_SHIFT=6
	
	#_BT_UNKNOWN = 0
	#_BT_INTEGER=BT_UNKNOWN+1 
	#_BT_LOGICAL=BT_INTEGER+1
	#_BT_REAL=BT_LOGICAL+1
	#_BT_COMPLEX=BT_REAL+1
	#_BT_DERIVED=BT_COMPLEX+1
	#_BT_CHARACTER=BT_DERIVED+1
	#_BT_CLASS=BT_CHARACTER+1
	#_BT_PROCEDURE=BT_CLASS+1
	#_BT_HOLLERITH=BT_PROCEDURE+1
	#_BT_VOID=BT_HOLLERITH+1
	#_BT_ASSUMED=BT_VOID+1	
	
	#_index_t = ctypes.c_int64
	#_size_t = ctypes.c_int64	
	
	
	
	#def __init__(self,lib,**kwargs):
		#self.name=kwargs['name']
		#self.mangled_name=kwargs['mangled_name']

		#self.args=kwargs['args']
		#self._set_args()
		
		
		#self.lib=lib
		#self._kwargs=kwargs

		#if kwargs['ctype'] is not 'void':
			#self.ctype_=kwargs['ctype']
			#self.cres=getattr(ctypes,self.ctype_)
			#self.pyres=getattr(__builtin__,kwargs['type'])
		#else:
			#self.cres=None
			#self.pyres=self._null_obj
			
		#self.f=getattr(lib,self.mangled_name)

	#def _set_args(self):
		#self.arg_ctypes=[]
		#a=[]
		#for i in self.args:
			#a.append(self._convert_arg_2_fvar(i))
			#self.arg_ctypes.append(a[-1]['ctype'])
		#self.args=a

	#def _convert_arg_2_fvar(self,*arg):
		#return fVar(*arg)

	#def _convert_in_args(self,*args):
		#pass
		
	#def _convert_out_args(self,*args):
		#pass

		
	#def call_func(self,*args):
		
		#f=self._get_from_lib()
		
		##Return type
		#f.restype=self.cres
		
		#f.argtypes=self.arg_ctypes
		
		##Call function
		#if len(self.args)>0:
			#in_args=self._convert_in_args(*args)
			#res_value=f(*in_args)
			#out_args=self._convert_out_args(*in_args)
		#else:
			#res_value=f()
			
		#if self.pyres is not None:
			#res=self.pyres(res_value)
		#else:
			##Set dict based on inout/out args
			#res=out_args

		#return res


class fFort(object):
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
		obj['arrgparse']=[]
		self._init_var(i)
		for i in obj['args']:
			self._init_var(i)
			obj['argparse'].append(i['_ctype'])
		obj['_call']=self._get_from_lib(obj)	
		obj['_call'].argparse=obj['arrgparse']
		obj['_call'].restype=obj['_ctype']
		
	def _init_array(self,obj):
		obj['array']['_ctype']=obj['_ctype']
		obj['_ctype']=self._get_array_ctype(obj)
	
	def _init_dt(self,obj):
		for i in obj['args']:
			self._init_var(i)
		
	def _init_array_dt(self,obj):
		pass
		
	def _get_ctype(self,obj):
		if 'intent' not in obj.keys():
			obj['_ctype']=getattr(ctypes,obj['ctype'])
		elif obj['intent']=="in":
			obj['_ctype']=getattr(ctypes,obj['ctype'])
		elif obj['intent']=="out" or obj['intent']=="inout":
			obj['_ctype']=ctypes.POINTER(getattr(ctypes,obj['ctype']))
			
	def _get_pytype(self,obj):
		if obj['pytype']=='void':
			obj['_pytype']=None
		else:
			obj['_pytype']=getattr(__builtin__,obj['pytype'])

	def _init_var(self,obj):
		self._get_ctype(obj)
		self._get_pytype(obj)
		
		if obj['array'] and obj['dt']:
			self._init_array_dt(obj)
		elif obj['array']:
			self._init_array(obj)
		elif obj['dt']:
			self._init_dt(obj)
		
	def _set_var(self,value,obj):
		res=self._get_from_lib(obj)
		self._var_to_ctype(value,obj)

	def _set_param(self,value,obj):
		raise AttributeError("Can't alter a parameter")
		
	def _get_var(self,obj):
		res=self._get_from_lib()
		return self._ctype_to_var(res,obj)
		
	def _get_param(self,obj):
		return obj['value']
		
	#Module variables
	def _set_array(self,obj,value):
		pass
		
	def _get_array(self,obj,array):
		pass
		

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
			

	def _var_to_ctype(self,value,obj):
		value.value=obj['_pytype'](value)

	def _array_dt_to_ctype(self,value,obj):
		pass
		
	def _array_to_ctype(self,value,obj):
		self.__array_to_ctype(value,obj)
		
	def _dt_to_ctype(self,value,obj):
		pass
		
		
	def _ctype_to_var(self,value,obj):
		if obj['pytype'] is not 'str':
			x=self._var_to_ctype(value,obj)
			x=obj['_pytype'](value.value)
		else:
			x=self._get_string_by_name(value,obj)
		return x

	def _ctype_to_array_dt(self,value,obj):
		pass
		
	def _ctype_to_array(self,value,obj):
		self.__ctype_to_array(value,obj)
		
	def _ctype_to_dt(self,value,obj):
		pass
		
	def _get_from_lib(self,obj):
		try:
			res=self.ctype.in_dll(self.lib,obj['mangled_name'])
		except (ValueError, AttributeError):
			print("Cant find "+obj['name'])
		return res
		
	def _get_string_by_name(self,value,obj):
		""" Gets a string"""
		base_address=ctypes.addressof(value)
		return self.__get_string_from_address(base_address)

	def __get_string_from_address(self,ctype_address,debug=False):
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
				i=i+1
		return out	


	def _get_array_ctype(self,obj):
		arr=obj['array']
		obj['_ctype']=self.__make_array_ctype(arr['ndims'])
		#Place to store the array after eing ctytpe-ifed
		obj['_array']=obj['_ctype']()
		#Initilize default values for this array
		self.__init_array_ctype(obj)
		
	def __make_array_ctype(self,ndim):
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
		
	def __array_to_ctype(self,array,obj):
		obj['_arrary'].base_addr=array.ctypes.data_as(ctypes.c_void_p)
		
		obj['_arrary'].offset=self.size_t(-1)
		
		obj['_arrary'].dtype=self.__get_dtype(obj)
				
		for i in range(0,obj['array']['ndim']):
			#Not sure if bounds are actually needed?
			obj['_arrary'].dims[i].stride=self._index_t(array.strides[i]//ctypes.sizeof(obj['array']['_ctype']))
			obj['_arrary'].dims[i].lbound=self._index_t(1)
			obj['_arrary'].dims[i].ubound=self._index_t(array.shape[i])
				
	def __get_dtype(self,obj):
		ftype=self.__get_ftype(obj)
		dtype=obj['array']['ndims']
		dtype=dtype|(ftype<<self.GFC_DTYPE_TYPE_SHIFT)
		dtype=dtype|(ctypes.sizeof(obj['array']['_ctype'])<<self.GFC_DTYPE_SIZE_SHIFT)
		return dtype

	def __init_array_ctype(self,obj):		
		obj['_array'].base_addr=ctypes.c_void_p()
		
		obj['_array'].offset=self.size_t(-1)
		obj['_array'].dtype=0
				
		for i in range(0,ndim):
			obj['_array'].dims[i].stride=self.index_t(1)
			obj['_array'].dims[i].lbound=self.index_t(1)
			obj['_array'].dims[i].ubound=self.index_t(1)	

		return _array,desc

	def __get_ftype(self,obj):
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


	def __ctype_to_array(self,value,obj):

		sizebytes=ctypes.sizeof(obj['array']['_ctype'])
		
		shape=[]
		stride=[]
		for i in range(0,obj['array']['ndim']):
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




