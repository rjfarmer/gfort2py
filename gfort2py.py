import ctypes
import pickle
import parseMod


class fUtils(object):
	def _get_from_lib(self):
		try:
			res=self.ctype.in_dll(self.lib,self.mangled_name)
		except (ValueError, AttributeError):
			print("Cant find "+self.name)
		return res

	def _null_obj(self,*args,**kwargs):
		pass
		
	def _get_string_name(self):
		""" Gets a string"""
		res=self.get_from_lib()
		base_address=ctypes.addressof(res)
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
				i=i+1
		return out	

class fVar(fUtils):
	def __init__(self,lib,**kwargs):
		self.name=kwargs['name']
		self.mangled_name=kwargs['mangled_name']
		try:
			self.is_param=kwargs['param']
		except KeyError:
			self.is_param=False
		try:		
			self.value=kwargs['value']
		except KeyError:
			self.value=None
		self.ctype_=kwargs['ctype']
		self.ctype=getattr(ctypes,self.ctype_)
		self.pytype=getattr(__builtin__,kwargs['type'])
		self.lib=lib
		self._kwargs=kwargs
			
	def set_var(self,value):
		if not self.is_param:
			res=self._get_from_lib()
			res.value=self.pytype(value)
		else:
			raise AttributeError("Can't alter a parameter")
		
	def get_var(self):
		if not self.is_param:
			res=self._get_from_lib()
			value=res.value
		else:
			value=self.value
		
		return self.pytype(value)
		

	def py2f_mod(self,x):
		return self.ctype(x)
		
	
	def py2f_func(self,x):
		return self.ctype(x),self.ctype
	
	def f2py_mod(self,x):
		return self.pytype(x)
	
	def f2py_func(self,x):
		return self.pytype(x)
		
class fArray(object):
	#GCC constants
	GFC_MAX_DIMENSIONS=7
	
	GFC_DTYPE_RANK_MASK=0x07
	GFC_DTYPE_TYPE_SHIFT=3
	GFC_DTYPE_TYPE_MASK=0x38
	GFC_DTYPE_SIZE_SHIFT=6
	
	BT_UNKNOWN = 0
	BT_INTEGER=BT_UNKNOWN+1 
	BT_LOGICAL=BT_INTEGER+1
	BT_REAL=BT_LOGICAL+1
	BT_COMPLEX=BT_REAL+1
	BT_DERIVED=BT_COMPLEX+1
	BT_CHARACTER=BT_DERIVED+1
	BT_CLASS=BT_CHARACTER+1
	BT_PROCEDURE=BT_CLASS+1
	BT_HOLLERITH=BT_PROCEDURE+1
	BT_VOID=BT_HOLLERITH+1
	BT_ASSUMED=BT_VOID+1	
	
	index_t = ctypes.c_int64
	size_t = ctypes.c_int64
	
	"""
	Need to handle four different cases
	
	python defines array and passes as an assumed shape array
	python defines array and passes as pointer to first element
	fortran returns an assumed shape array 
	fortran returns a pointer to first element
	
	plus pointer versions?
	
	"""
	
	def __init__(self,array=None,**kwargs):
		pass
		
	def _set_pyarray(self,array,assumed=True):
		self.array=array
		self.dtype=self._get_dtype()
		self.ftype,self._ctype=self._get_type()
		self.ndim=self.array.ndim
		self.shape=self.array.shape
		self.lbound=np.array([1]*self.ndim)
		self.ubound=np.array(self.shape)
	
		if assumed:
			self._desc=_build_desc()
			self.ctype=ctypes.POINTER(self._desc)
			self._build_array()
		else:
			self.ctype=ctypes.c_void_p
			self._array(self.array.data_as(ctypes.c_void_p))
	
		
	def _get_farray(self,assumed=True,**kwargs):
		pass
		
	
	
	def _make_desc(self,ndim):
		class descriptor(ctypes.Structure):
			_fields_=[("stride",self.index_t),
					("lbound",self.index_t),
					("ubound",self.index_t)]
		
		class defarray(ctypes.Structure):
			_fields_=[("base_addr",ctypes.c_void_p),
					("offset",self.size_t),
					("dtype",self.index_t),
					("dims",descriptor*ndim)]
					
		return defarray	
		
	def _get_type(self):
		dtype=self.array.dtype.kind		
		if dtype=='i':
			ftype=self.BT_INTEGER
			ctype=ctypes.c_int32
		elif dtype=='f':
			ftype=self.BT_REAL
			ctype=ctypes.c_double
		elif dtype=='b':
			ftype=self.BT_LOGICAL
			ctype=ctypes.c_bool
		elif dtype=='U' or dtype=='S':
			ftype=self.BT_CHARACTER
			ctype=ctypes.c_char
		else:
			raise ValueError("Cant match dtype, got "+dtype)
		
		return ftype,ctype
				
	def _get_dtype(self):
		ftype,ctype=self._get_type()
		dtype=self.array.ndim
		dtype=dtype|(ftype<<self.GFC_DTYPE_TYPE_SHIFT)
		dtype=dtype|(ctypes.sizeof(ctype)<<self.GFC_DTYPE_SIZE_SHIFT)
		return dtype


	def _find_dtype(self,dtype):
		rank=dtype&self.GFC_DTYPE_RANK_MASK
		ftype=(dtype&self.GFC_DTYPE_TYPE_MASK)>>self.GFC_DTYPE_TYPE_SHIFT
		sizebytes=dtype>>self.GFC_DTYPE_SIZE_SHIFT
		return rank,ftype,sizebytes
		
	def _get_pytype(self,ftype,sizebytes):
		if ftype==self.BT_INTEGER:
			ctype=gf.get_ctype_int(sizebytes)
			pytype=int
		elif ftype==self.BT_REAL:
			ctype=gf.get_ctype_float(sizebytes)
			pytype=float
		elif dtype==self.BT_LOGICAL:
			pytype=np.bool
			ctype=ctypes.c_bool
		elif ftype==self.BT_CHARACTER:
			pytype=np.char
			ctype=ctypes.c_char
		else:
			raise ValueError("Cant match ftype, got "+ftype)		
		
		return pytype,ctype

	def _build_desc(self):
		self._desc=self._make_desc(self._ndim)
		
		self._array=self._desc()
		
		self._array.base_addr=ctypes.c_void_p()
		
		self._array.offset=self.size_t(-1)
		
		self._array.dtype=0
				
		for i in range(0,ndim):
			self._array.dims[i].stride=self.index_t(1)
			self._array.dims[i].lbound=self.index_t(1)
			self._array.dims[i].ubound=self.index_t(1)	

	def _build_array(self):
		self._arrary.base_addr=self.array.ctypes.data_as(ctypes.c_void_p)
		
		self._arrary.offset=self.size_t(-1)
		
		self._arrary.dtype=dtype
				
		for i in range(0,self._ndim):
			self._arrary.dims[i].stride=self.index_t(self.array.strides[i]//ctypes.sizeof(ctype))
			self._arrary.dims[i].lbound=self.index_t(1)
			self._arrary.dims[i].ubound=self.index_t(self.array.shape[i])	
			
			
	def ctype2array(self,carray=None):
		if carray is None:
			#We passed the array so fortran filled our memory allready
			return self.array
			
		self.rank,self.ftype,self.sizebytes=self._find_dtype(carray.dtype)	
			
		shape=[]
		stride=[]
		for i in range(0,ndim):
			shape.append((carray.dims[i].ubound-
						carray.dims[i].lbound)+1)
			stride.append(carray.dims[0].stride*self.sizebytes)
		
		off=0
		arr=[]
		for i in range(np.product(shape)):
				off=i*stride[0]
				arr.append(ctype.from_address(carray.base_addr+off).value)

		#Copy array data
		self.array=np.reshape(arr,newshape=shape)	
		
	def pass2func(self,func_name):
		self._build_desc()
		self._build_array()
		func=getattr(lib,func_name)
		func.argtypes=[self._ctype]
		func(self._array)

	def getFixedArray(self,name,ctype,size):
		#When we have a fixed sized array as a module varaiable
		x=self.get_from_lib(name)
		address=ctypes.addressof(x)
		sizeof=ctypes.sizeof(self.ctype)
		a=[]
		for i in range(size):
			a.append(ctypes.ctype.from_address(address))
			address=adress+sizeof
		return np.array(a)
		
	def _get_array(self,array_in):			
		self.ndims,self.ftype,self.sizebytes=self._find_dtype(array_in.dtype)	
			
		shape=[]
		stride=[]
		for i in range(0,self.ndim):
			shape.append((array_in.dims[i].ubound-
						array_in.dims[i].lbound)+1)
			stride.append(array_in.dims[0].stride*self.sizebytes)
		
		off=0
		arr=[]
		for i in range(np.product(shape)):
				off=i*stride[0]
				arr.append(ctype.from_address(array_in.base_addr+off).value)

		#Copy array data
		self.array=np.reshape(arr,newshape=shape)
	
class fFunc(fUtils):
	def __init__(self,lib,**kwargs):
		self.name=kwargs['name']
		self.mangled_name=kwargs['mangled_name']

		self.args=kwargs['args']
		self._set_args()
		
		
		self.lib=lib
		self._kwargs=kwargs

		if kwargs['ctype'] is not 'void':
			self.ctype_=kwargs['ctype']
			self.cres=getattr(ctypes,self.ctype_)
			self.pyres=getattr(__builtin__,kwargs['type'])
		else:
			self.cres=None
			self.pyres=self._null_obj
			
		self.f=getattr(lib,self.mangled_name)

	def _set_args(self):
		self.arg_ctypes=[]
		a=[]
		for i in self.args:
			a.append(self._convert_arg_2_fvar(i))
			self.arg_ctypes.append(a[-1]['ctype'])
		self.args=a

	def _convert_arg_2_fvar(self,*arg):
		return fVar(*arg)

	def _convert_in_args(self,*args):
		pass
		
	def _convert_out_args(self,*args):
		pass

		
	def call_func(self,*args):
		
		f=self._get_from_lib()
		
		#Return type
		f.restype=self.cres
		
		f.argtypes=self.arg_ctypes
		
		#Call function
		if len(self.args)>0:
			in_args=self._convert_in_args(*args)
			res_value=f(*in_args)
			out_args=self._convert_out_args(*in_args)
		else:
			res_value=f()
			
		if self.pyres is not None:
			res=self.pyres(res_value)
		else:
			#Set dict based on inout/out args
			res=out_args

		return res


class fFort(object):
	def __init__(self,libname,fpy):
		self.lib=ctypes.CDLL(libname)
		self.libname=libname

		with open(fpy,'rb') as f:
			self.version=pickle.load(f)
			self.mod_data=pickle.load(f)
			self.obj_all=pickle.load(f)

		for i in self.obj_all:
			if i['type'] is not 'void':
				#Variable
				pass
			elif i['ctype'] == 'void':
				#DT
				pass
			else:
				pass
			
				
				
		
	def _init_var(self):
		pass
		
	def _init_func(self):	
		pass
		
	def init_array(self):
		pass
	
	def init_dt(self):
		pass
		
	def _call(self,name,*args):
		pass
		
		
	def _get_var(self,name):
		pass
		
	def _set_var(self,name,value):
		pass

x=fFort('./test_mod.so','test_mod.fpy')




