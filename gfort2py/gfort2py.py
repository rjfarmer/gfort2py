from __future__ import print_function
try:
	import __builtin__
except ImportError:
	import builtins as __builtin__
	
import ctypes
import pickle
import numpy as np
import errno

import gfort2py.parseMod as pm
from gfort2py.cmplx import fComplex,fParamComplex
from gfort2py.arrays import fExplicitArray,fDummyArray,fParamArray
from gfort2py.functions import fFunc
from gfort2py.strings import fStr
from gfort2py.types import fDerivedType,fDerivedTypeDesc
import gfort2py.utils as utils
from gfort2py.var import fVar,fParam

prefix='_f_'

		
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
			if i['dt']:
				for j in self._dt_defs:
					if i['dt'].lower()==j['name'].lower():
						i['_dt_def']=j
		
		for i in self._mod_vars:
			self._init_var(i)
			
		for i in self._param:
			self._init_param(i)
			
		#Must come last after the derived types are setup
		for i in self._funcs:
			self._init_func(i)
			
					
	def _init_var(self,obj):
		if obj['pytype']=='str':
			x=fStr(self.lib,obj)
		elif obj['pytype']=='complex':
			x=fComplex(self.lib,obj)
		elif obj['dt']:
			x=fDerivedType(self.lib,obj)
		elif obj['array']:
			x=fExplicitArray(self.lib,obj)
		else:
			x=fVar(self.lib,obj)
			
		self.__dict__[prefix+x.name]=x
		
	def _init_param(self,obj):
		if obj['pytype']=='complex':
			x=fParamComplex(self.lib,obj)
		elif len(obj['value']):
			x=fParamArray(self.lib,obj)
		else:
			x=fParam(self.lib,obj)
			
		self.__dict__[prefix+x.name]=x
		
		
	def _init_func(self,obj):
		x=fFunc(self.lib,obj)
		self.__dict__[prefix+x.name]=x
		
	def __getattr__(self,name):
		if name in self.__dict__:
			return self.__dict__[name]
			
		if prefix+name in self.__dict__:	
			return self.__dict__[prefix+name]
		raise AttributeError("No variable "+name)
					
	def __setattr__(self,name,value):
		if name in self.__dict__:
			self.__dict__[name]=value
			return
		
		if prefix+name in self.__dict__:	
			self.__dict__[prefix+name].set_mod(value)
		else:
			self.__dict__[name]=value
			return
				
	def __dir__(self):
		lv=[str(i.replace(prefix,'')) for i in self.__dict__ if prefix in i]
		return lv
