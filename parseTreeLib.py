import ctypes
import numpy as np
import pickle


class fCtype(object):
	#######################################################################
	# ctype handling
	#######################################################################
	
	def mangle_name(self,mod,name):
		return "__"+mod+'_MOD_'+name
	
	def get_ctype_int(self,size):
		res=None
		size=int(size)
		if size==ctypes.sizeof(ctypes.c_int):
			res='c_int'
		elif size==ctypes.sizeof(ctypes.c_int16):
			res='c_int16'
		elif size==ctypes.sizeof(ctypes.c_int32):
			res='c_int32'
		elif size==ctypes.sizeof(ctypes.c_int64):
			res='c_int64'
		else:
			raise ValueError("Cant find suitable int for size "+size)	
		return res
		
	def get_ctype_float(self,size):
		res=None
		size=int(size)
		if size==ctypes.sizeof(ctypes.c_float):
			res='c_float'
		elif size==ctypes.sizeof(ctypes.c_double):
			res='c_double'
		elif size==ctypes.sizeof(ctypes.c_long):
			res='c_long'
		elif size==ctypes.sizeof(ctypes.c_longdouble):
			res='c_longdouble'
		elif size==ctypes.sizeof(ctypes.c_longlong):
			res='c_long'
		else:
			raise ValueError("Cant find suitable float for size"+size)
	
		return res
		
	def get_ctype_bool(self,size):
		return 'c_bool'	
	
	def get_ctype_str(self,size):
		return 'c_char_p'	
	
	def map_to_scalar_ctype(self,var):
		"""
		gets the approitate ctype for a variable
		
		Returns:
			String
		"""
		typ=var['type_spec']['type']
		size=var['type_spec']['size']
	
		res=None
		if typ=='int':
			res=get_ctype_int(size)
		elif typ=='float':
			res=get_ctype_float(size)	
		elif typ=='char':
			res=get_ctype_str(size)
		elif typ=='bool':
			res=get_ctype_bool(size)
		elif typ=='struct':
			raise ValueError("Should of called map_to_struct_ctype")
		else:
			raise ValueError("Not supported ctype "+var['name']+' '+str(typ)+' '+str(size))
		
		return res

class fDerivedType(fCtype):
	def __init__(self,var,module,lib):
		



class fScalarVariable(fCtype):
	def __init__(self,var,module,lib):
		self.name=var['name']
		self.is_param=var['param']
		self.value=var['value']
		ctype_=self.map_to_scalar_ctype(var)
		self.ctype=getattr(ctypes,ctype_)
		self.lib=lib
			
	def set(self,value):
		if not self.is_param:
			res=self.ctype.in_dll(self.lib,self.name)
			res.value=value
		else:
			raise AttributeError("Can't alter a parameter")
		
	def get(self):
		if not self.is_param:
			res=self.ctype.in_dll(self.lib,self.name)
			value=res.value
		else:
			value=self.value
		
		return value
		
	def __str__(self):
		return str(self.value)



filename='test_mod.flib'
with open(filename,'rb') as f:
	num_mods=pickle.load(f)
	data=[]
	for i in range(num_mods):
		data.append(pickle.load(f))

x=data[0]

lib=True

list_vars=[]

for i in x['mod_vars']:
	if not i['array'] and not i['struct']:
		list_vars.append(fScalarVariable(var=i,module=x['module_name'],lib=lib))
	
	if i['struct'] and not i['array']:
		pass
		
	if i['array'] and not i['struct']:
		pass
		
	if i['struct'] and i['array']:
		pass


















		
class fFunction(object):
	def __init__(self,name,lib,args,res):
		self.name=name
		self.ctype_name=getattr(ctypes,self.name)
		self.lib=lib	
		self.args=args
		self.ctype_args=[getattr(ctypes,x) for x in args]
		self.res=getattr(ctypes,res)

		self.ctype_name.argtypes=self.ctype_args
		self.ctype_name.restype=self.res
	

	def __call__(self,*args):
		#Do some arg handling?
		res=self.ctype_name(*args)
		return res.value
		
		
#Handles defered shape array (ie dimension(:) (both allocatable and non-allocatable) 
#gfortran passes them as structs
#Arrays with fixed size (dimension(10)) as passed as pointers to first element
class fDeferedArray(object):
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
	

	def __init__(self,array):
		self.array=array
	
	def array2ctype(self):
		#Used when python allocs the memory
		dtype=self._get_dtype()
		ftype,ctype=self._get_type()
		ndim=self.array.ndim
		
		desc=self._get_desc(self.array.ndim)
		
		result=desc()
		
		result.base_addr=self.array.ctypes.data_as(ctypes.c_void_p)
		
		result.offset=self.size_t(-1)
		
		result.dtype=dtype
				
		for i in range(0,ndim):
			result.dims[i].stride=self.index_t(self.array.strides[i]//ctypes.sizeof(ctype))
			result.dims[i].lbound=self.index_t(1)
			result.dims[i].ubound=self.index_t(self.array.shape[i])		
				
		#Must use the desc used to create result to pass to function.argtypes
		return result,desc
		
	def array2ctypeEmpty(self,ndim=1):
		#Used when fortran allocs the memoray
		dtype=self._get_dtype()
		ftype,ctype=self._get_type()
		
		desc=self._get_desc(ndim)
		
		result=desc()
		
		result.base_addr=ctypes.c_void_p()
		
		result.offset=self.size_t(-1)
		
		result.dtype=0
				
		for i in range(0,ndim):
			result.dims[i].stride=self.index_t(1)
			result.dims[i].lbound=self.index_t(1)
			result.dims[i].ubound=self.index_t(1)	
				
		#Must use the desc used to create result to pass to function.argtypes
		return result,desc
	
	def _get_desc(self,ndim):
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
		res1=self.BT_UNKNOWN
		res2=ctypes.c_int64
		
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
		
	def ctype2array(self,carray=None):
		if carray is None:
			#We passed the array so fortran filled our memory allready
			return self.array
		
		self.carray=carray
		#We didnt pass an array,fortran allocated it so we must make our own
		ndim,typ,sizebytes=self._find_dtype()
		pytype,ctype=self._get_pytype(typ,sizebytes)
		
		shape=[]
		stride=[]
		for i in range(0,ndim):
			shape.append((self.carray.dims[i].ubound-
						self.carray.dims[i].lbound)+1)
			stride.append(self.carray.dims[0].stride*sizebytes)
		
		off=0
		arr=[]
		for i in range(np.product(shape)):
				off=i*stride[0]
				arr.append(ctype.from_address(self.carray.base_addr+off).value)

		#Copy array data
		self.array=np.reshape(arr,newshape=shape)
		return self.array
		
	def _find_dtype(self):
		dtype=self.carray.dtype
		rank=dtype&self.GFC_DTYPE_RANK_MASK
		ty=(dtype&self.GFC_DTYPE_TYPE_MASK)>>self.GFC_DTYPE_TYPE_SHIFT
		sizebytes=dtype>>self.GFC_DTYPE_SIZE_SHIFT
		return rank,ty,sizebytes
		
	def _get_pytype(self,ftype,sizebytes):
		if ftype==self.BT_INTEGER:
			if sizebytes==4:
				pytype=np.int32
				ctype=ctypes.c_int32
			elif sizebytes==8:
				pytype=np.int64
				ctype=ctypes.c_int64
			else:
				raise ValueError("Cant match ftype size, got "+ftype+" "+sizebytes)
		elif ftype==self.BT_REAL:
			if sizebytes==4:
				pytype=np.float32
				ctype=ctypes.c_float
			elif sizebytes==8:
				pytype=np.float64
				ctype=ctypes.c_double
			else:
				raise ValueError("Cant match ftype size, got "+ftype+" "+sizebytes)
		elif dtype==self.BT_LOGICAL:
			pytype=np.bool
			ctype=ctypes.c_bool
		elif ftype==self.BT_CHARACTER:
			pytype=np.char
			ctype=ctypes.c_char
		else:
			raise ValueError("Cant match ftype, got "+ftype)		
		
		return pytype,ctype
