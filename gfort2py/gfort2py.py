from __future__ import print_function
try:
	import __builtin__
except ImportError:
	import builtins as __builtin__
	
import ctypes
import pickle
import gfort2py.parseMod as pm
import numpy as np
import errno
		
def find_key_val(list_dicts,key,value):
	for idx,i in enumerate(list_dicts):
		if i[key]==value:
			return idx		
			
						
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
		
class fParamArray(fVar):
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
		return np.array(self.value,dtype=self.pytype)
		
class fParamComplex(fVar):
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
		return complex(float(self.value[0]),float(self.value[1]))
			
		
class fComplex(fVar):
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
		r=self._get_from_lib()
		x=[value.real,value.imag]
		return self._set_var_from_iter(r,x,2)
	
	def ctype_to_py(self,value):
		"""
		Pass in a ctype value returns the python representation of it
		"""
		x=self._get_var_by_iter(value,2)
		return self._pytype(x[0],x[1])
		
	def pytype_def(self):
		return complex
	
	def ctype_def(self):
		"""
		The ctype type of this object
		"""
		return getattr(ctypes,self.ctype)


	def set_mod(self,value):
		if isinstance(value,complex):
			self.py_to_ctype(value)
		else:
			raise ValueError("Not complex")
		
		
	def get_mod(self):
		r=self._get_from_lib()
		s=self.ctype_to_py(r)
		return s
		
	
		
class fStr(fVar):
	def __init__(self,lib,obj):
		self.__dict__.update(obj)
		self.lib=lib
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
	
	def get_mod(self):
		"""
		Get a module level variable
		"""
		r=self._get_from_lib()
		s=self.ctype_to_py(r)
		return ''.join([i.decode() for i in s])
	
	
class fExplicitArray(fVar):
	def __init__(self,lib,obj):
		self.__dict__.update(obj)
		self.lib=lib
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
	
	def get_mod(self):
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
	
class fDerivedType(fVar):	
	def __init__(self,lib,obj):
		self.__dict__.update(obj)
		self.lib=lib
		self._ctype=ctypes.c_void_p
		
	
class fDerivedTypeDesc(ctypes.Structure):	
	def __init__(self,lib,obj):
		self.__dict__.update(obj)
		self.lib=lib
		
	def setup_desc(self):
		self._args=[]
		self._nameArgs=[]
		self._typeArgs=[]
		for i in self.args:
			self._args.append(fVar(self.lib,i))
			self._nameArgs.append(self._args[-1].name)
			self._typeArgs.append(self._args[-1]._ctype)
			
		self._set_fields(self._nameArgs,self._typeArgs)
		
	def set_fields(self,nameArgs,typeArgs):
		self._fields_=[(i,j) for i,z in zip(nameArgs,typeArgs)]
			
	def add_arg(self,name,ctype):
		self._fields_.append((name,ctype))
	
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
		self.lib=lib
		
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
		
	
class fFunc(fVar):
	def __init__(self,obj):
		pass
			
			
class fFort(object):
	
	def __init__(self,libname,ffile,reload=False):		
		self.lib=ctypes.CDLL(libname)
		self.libname=libname
		self.fpy=pm.fpyname(ffile)
		self._load_data(ffile,reload)
		self._init()

	def _load_data(self,ffile,reload=False):
		try:
			f=open(self.fpy,'rb')
		except (OSError, IOError) as e: # FileNotFoundError does not exist on Python < 3.3
			if e.errno != errno.ENOENT:
				raise
			pm.run_and_save(ffile)
		else:
			f.close()
		
		with open(self.fpy,'rb') as f:
			self.version=pickle.load(f)
			if self.version ==1:
				self._mod_data=pickle.load(f)

				if self._mod_data["checksum"] != pm.hash_file(ffile) or reload:
					x=pm.run_and_save(ffile,return_data=True)
					self._mod_data=x[0]
					self._mod_vars=x[1]
					self._param=x[2]
					self._funcs=x[3]
					self._dt_defs=x[4]
				else:
					self._mod_vars=pickle.load(f)
					self._param=pickle.load(f)
					self._funcs=pickle.load(f)
					self._dt_defs=pickle.load(f)
					

	def _init(self):
		self._listVars=[]
		self._listParams=[]
		self._listFuncs=[]
		for i in self._mod_vars:
			self._init_var(i)
			
		for i in self._param:
			self._init_param(i)
			
					
	def _init_var(self,obj):
		if obj['pytype']=='str':
			self._listVars.append(fStr(self.lib,obj))
		elif obj['pytype']=='complex':
			self._listVars.append(fComplex(self.lib,obj))
		elif obj['array']:
			self._listVars.append(fExplicitArray(self.lib,obj))
		else:
			self._listVars.append(fVar(self.lib,obj))
		
	def _init_param(self,obj):
		if obj['pytype']=='complex':
			self._listParams.append(fParamComplex(self.lib,obj))
		elif len(obj['value']):
			self._listParams.append(fParamArray(self.lib,obj))	
		else:
			self._listParams.append(fParam(self.lib,obj))	
		
		
	def __getattr__(self,name):
		if name in self.__dict__:
			return self.__dict__[name]
			
		if '_listVars' in self.__dict__ and '_listParams' in self.__dict__:	
			for i in self._listVars:
				if i.name==name:
					return i.get_mod()
			for i in self._listParams:
				if i.name==name:
					i.get_mod()
					return i.get_mod()
		raise AttributeError("No variable "+name)
					
	def __setattr__(self,name,value):
		if name in self.__dict__:
			self.__dict__[name]=value
			return
		
		if '_listVars' in self.__dict__ and '_listParams' in self.__dict__:	
			for i in self._listVars:
				if i.name==name:
					i.set_mod(value)
					return
			for i in self._listParams:
				if i.name==name:
					i.set_mod(value)	
					return
		else:
			self.__dict__[name]=value
			return
		
				
	def __dir__(self):
		if '_listVars' in self.__dict__ and '_listParams' in self.__dict__:	
			lv=[str(i.name) for i in self._listVars]
			lp=[str(i.name) for i in self._listParams]
			#lp=[str(i.name) for i in self._listFuncs]
			return lv
		else:
			return []
	
			
#class fFort(object):
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
	
	
	#def __init__(self,libname,ffile,reload=False):
		#self.lib=ctypes.CDLL(libname)
		#self.libname=libname
		#self.fpy=pm.fpyname(ffile)
		#self._load_data(ffile,reload)

	#def _load_data(self,ffile,reload=False):
		#try:
			#f=open(self.fpy,'rb')
		#except FileNotFoundError:
			#pm.run_and_save(ffile)
		#else:
			#f.close()
		
		#with open(self.fpy,'rb') as f:
			#self.version=pickle.load(f)
			#if self.version ==1:
				#self._mod_data=pickle.load(f)

				#if self._mod_data["checksum"] != pm.hash_file(ffile) or reload:
					#x=pm.run_and_save(ffile,return_data=True)
					#self._mod_data=x[0]
					#self._mod_vars=x[1]
					#self._param=x[2]
					#self._funcs=x[3]
					#self._dt_defs=x[4]
				#else:
					#self._mod_vars=pickle.load(f)
					#self._param=pickle.load(f)
					#self._funcs=pickle.load(f)
					#self._dt_defs=pickle.load(f)

	#def _init_mod_var(self):
		#for i in self._mod_vars:
			#self._init_var(i)
			
	#def _init_param(self):
		#for i in self._param:
			#self._init_var(i)
		
	#def _init_func(self,obj):
		#obj['argparse']=[]
		#self._init_var(obj)
		#for i in obj['args']:
			#self._init_var(i)
			#obj['argparse'].append(i['_ctype'])
		#if len(obj['argparse'])==0:
			#obj['argparse']=None
		#obj['_call']=getattr(self.lib,obj['mangled_name'])	
		#obj['_call'].argparse=obj['argparse']
		#obj['_call'].restype=obj['_ctype']
		
	#def _init_array(self,obj):
		#obj['array']['_ctype']=obj['_ctype']
		#self._get_array_ctype(obj)
	
	#def _init_dt(self,obj):
		#for i in obj['args']:
			#self._get_dt_ctype(i)
		
	#def _init_array_dt(self,obj):
		#pass
		
	#def _get_ctype(self,obj):
		#if 'intent' not in obj.keys():
			#if obj['ctype']=='void':
				#obj['_ctype']=None
			#else:
				#obj['_ctype']=getattr(ctypes,obj['ctype'])
		#elif obj['intent']=="in":
			#obj['_ctype']=getattr(ctypes,obj['ctype'])
		#elif obj['intent']=="out" or obj['intent']=="inout" or obj['pointer']:
			#obj['_ctype']=ctypes.POINTER(getattr(ctypes,obj['ctype']))
			
	#def _get_pytype(self,obj):
		#if obj['pytype']=='void':
			#obj['_pytype']=None
		#else:
			#obj['_pytype']=getattr(__builtin__,obj['pytype'])

	#def _init_var(self,obj):
		#self._get_ctype(obj)
		#self._get_pytype(obj)
		
		#if 'array' in obj.keys() and 'dt' in obj.keys():
			#if obj['array'] and obj['dt']:
					#self._init_array_dt(obj)
					#return
					
		#if 'array' in obj.keys():
			#if obj['array']:
				#self._init_array(obj)
				#return
				
		#if 'dt' in obj.keys():
			#if obj['dt']:
				#self._init_dt(obj)
				#return
		
	#def _set_var(self,value,obj):
		#res=self._get_from_lib(obj)
		#self._var_to_ctype(res,value,obj)

	#def _set_param(self,value,obj):
		#raise AttributeError("Can't alter a parameter")
		
	#def _get_var(self,obj):
		#res=self._get_from_lib(obj)
		#return self._ctype_to_var(res,obj)
		
	#def _get_param(self,obj):
		#return obj['value']
		
	##Module variables
	#def _set_array(self,value,obj):
		#res=self._get_from_lib(obj)
		#if 'explicit' in obj['array']['atype']:
			#array=self._set_explicit_array(res,value,obj)
		#else:
			#pass
		
	#def _get_array(self,obj):
		#res=self._get_from_lib(obj)
		#if 'explicit' in obj['array']['atype']:
			#array=self._get_explicit_array(res,obj)
		#else:
			#pass
		#return array

	#def _call(self,f,*args):
		##Convert args to ctype versions
		#args_in=[]
		#for i,j in zip(args,f['args']):
			#args_in.append(self._arg_to_ctype(i,j))

		##Call function
		#print("in",args,args_in,f['args'])
		#if len(args_in)>0:
			#res=f['_call'](args_in)
		#else:
			#res=f['_call']()

		##Convert back any args that changed:
		#args_out=[]
		#for i,j in zip(args_in,f['args']):
			#args_out.append(self.ctype_to_py(i,j))
			
		#if res is not None:
			#res=f['_pytype'](res)
			
		#return res,args_out		


	#def _arg_to_ctype(self,value,obj):
		#if obj['array'] and obj['dt']:
			#self._array_dt_to_ctype(value,obj)
		#elif obj['array']:
			#self._array_to_ctype(value,obj)
		#elif obj['dt']:
			#self._dt_to_ctype(value,obj)
		#else:
			#self._var_to_ctype(value,obj)
		
	#def _ctype_to_py(self,value,obj):
		#if obj['array'] and obj['dt']:
			#self._ctype_to_array_dt(value,obj)
		#elif obj['array']:
			#self._ctype_to_array(value,obj)
		#elif obj['dt']:
			#self._ctype_to_dt(value,obj)
		#else:
			#self._ctype_to_array(value,obj)
			

	#def _var_to_ctype(self,ctyp,value,obj):
		#if obj['pytype'] == 'str':
			#self._set_char_str(ctyp,value,obj)
		#else:
			#ctyp.value=obj['_pytype'](value)

	#def _array_dt_to_ctype(self,value,obj):
		#pass
		
	#def _array_to_ctype(self,value,obj):
		#self.__array_to_ctype(value,obj)
		
	#def _dt_to_ctype(self,value,obj):
		#pass
		
		
	#def _ctype_to_var(self,value,obj):
		#if obj['pytype'] == 'str':
			#x=self._get_string_by_name(value,obj)
		#else:
			#x=obj['_pytype'](value.value)
		#return x

	#def _ctype_to_array_dt(self,value,obj):
		#pass
		
	#def _ctype_to_array(self,value,obj):
		#self.__ctype_to_array(value,obj)
		
	#def _ctype_to_dt(self,value,obj):
		#pass
		
	#def _get_from_lib(self,obj):
		#res=None
		#try:
			#res=obj['_ctype'].in_dll(self.lib,obj['mangled_name'])
		#except (ValueError, AttributeError):
			#print("Cant find "+obj['name'])
		#return res
		
	#def _get_string_by_name(self,value,obj):
		#""" Gets a string"""
		#base_address=ctypes.addressof(value)
		#return self._get_string_from_address(base_address)

	#def _get_string_from_address(self,ctype_address,debug=False):
		#out=''
		#i=0
		#while True:
			#x=ctypes.c_char.from_address(ctype_address+i)
			#if debug:
				#print(x.value,i,x.value==b'\x00')
			#if x.value == b'\x00':
				#break
			#else:
				#out=out+(x.value).decode()
				#i=i+ctypes.sizeof(ctypes.c_char)
		#return out	
		
	#def _get_explicit_array(self,res,obj):
		#shape=self._make_array_shape(obj)
		#array=[]
		#k=0
		#base_address=ctypes.addressof(res)
		#offset=base_address
		#for i in range(np.product(shape)):
			#array.append(obj['_ctype'].from_address(offset).value)
			#offset=offset+ctypes.sizeof(obj['_ctype'])
		#return np.reshape(array,newshape=shape)

	#def _set_explicit_array(self,res,value,obj):
		#shape=self._make_array_shape(obj)
		#k=0
		#base_address=ctypes.addressof(res)
		#flatarray=value.flatten()
		#for i in range(len(shape)):
			#for j in range(shape[i]):
				#offset=base_address+k*ctypes.sizeof(obj['_ctype'])
				#obj['_ctype'].from_address(offset).value=flatarray[k]
				#k=k+1

	#def _set_char_str(self,res,value,obj):
		#base_address=ctypes.addressof(res)
		#for j in range(obj['char_len']):
			#offset=base_address+j*ctypes.sizeof(ctypes.c_char)
			#obj['_ctype'].from_address(offset).value=value[j].encode()


	#def _make_array_shape(self,obj):
		#bounds=obj['array']['bounds']
		#shape=[]
		#for i,j in zip(bounds[0::2],bounds[1::2]):
			#shape.append(j-i+1)
		#return shape
		

	#def _get_array_ctype(self,obj):
		#arr=obj['array']
		#if 'explicit' not in arr['atype']:
			#obj['_ctype']=self.__make_array_ctype(arr['ndims'])
			##Place to store the array after ctytpe-ifed
			#obj['_array']=obj['_ctype']()
			##Initilize default values for this array
			#self.__init_array_ctype(obj)
		
	#def _make_array_ctype(self,ndim):
		#class descriptor(ctypes.Structure):
			#_fields_=[("stride",self._index_t),
					#("lbound",self._index_t),
					#("ubound",self._index_t)]
		
		#class defarray(ctypes.Structure):
			#_fields_=[("base_addr",ctypes.c_void_p),
					#("offset",self._size_t),
					#("dtype",self._index_t),
					#("dims",descriptor*ndim)]
					
		#return defarray	
		
	#def _make_dt_ctype(self,obj,dt_defs):
		#class dt(ctypes.Structure):
			#def __init__(self,lnames,lctypes):
				#self._fields_=[]
				#for k in obj['args']:
					#for i,j in zip():
						#if k['dt']:
							##Match to the list of dt's
							#for l in dt_defs:
								#if k['name']==l['name']:
									#c=ctypes.POINTER(self._make_dt_ctype(l))
						#else:
							#c=k['_ctype']
						#self._fields_.append((k['name'],c))
		#return dt
		
	#def _array_to_ctype(self,array,obj):
		#obj['_arrary'].base_addr=array.ctypes.data_as(ctypes.c_void_p)
		
		#obj['_arrary'].offset=self.size_t(-1)
		
		#obj['_arrary'].dtype=self._get_dtype(obj)
				
		#for i in range(0,obj['array']['ndim']):
			##Not sure if bounds are actually needed?
			#obj['_arrary'].dims[i].stride=self._index_t(array.strides[i]//ctypes.sizeof(obj['array']['_ctype']))
			#obj['_arrary'].dims[i].lbound=self._index_t(1)
			#obj['_arrary'].dims[i].ubound=self._index_t(array.shape[i])
				
	#def _get_dtype(self,obj):
		#ftype=self._get_ftype(obj)
		#dtype=obj['array']['ndims']
		#dtype=dtype|(ftype<<self.GFC_DTYPE_TYPE_SHIFT)
		#dtype=dtype|(ctypes.sizeof(obj['array']['_ctype'])<<self.GFC_DTYPE_SIZE_SHIFT)
		#return dtype

	#def _init_array_ctype(self,obj):		
		#obj['_array'].base_addr=ctypes.c_void_p()
		
		#obj['_array'].offset=self._size_t(-1)
		#obj['_array'].dtype=0
				
		#for i in range(0,obj['array']['ndims']):
			#obj['_array'].dims[i].stride=self._index_t(1)
			#obj['_array'].dims[i].lbound=self._index_t(1)
			#obj['_array'].dims[i].ubound=self._index_t(1)	


	#def _get_ftype(self,obj):
		#dtype=obj['ctype']	
		#if 'c_int' in dtype:
			#ftype=self._BT_INTEGER
		#elif 'c_double' in dtype or 'c_real' in dtype:
			#ftype=self._BT_REAL
		#elif 'c_bool' in dtype:
			#ftype=self._BT_LOGICAL
		#elif 'c_char' in dtype:
			#ftype=self._BT_CHARACTER
		#else:
			#raise ValueError("Cant match dtype, got "+dtype)
		#return ftype


	#def _ctype_to_array(self,value,obj):

		#sizebytes=ctypes.sizeof(obj['array']['_ctype'])
		
		#shape=[]
		#stride=[]
		#for i in range(0,obj['array']['ndims']):
			#shape.append((value.dims[i].ubound-
						#value.dims[i].lbound)+1)
			#stride.append(value.dims[0].stride*sizebytes)
		
		#off=0
		#arr=[]
		#for i in range(np.product(shape)):
				#off=i*stride[0]
				#arr.append(ctype.from_address(value.base_addr+off).value)

		##Copy array data
		#array=np.reshape(arr,newshape=shape)
		#return array

