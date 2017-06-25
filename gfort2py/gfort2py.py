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
	
