import ctypes
import os
from .var import fVar
from .utils import *

class fFunc(fVar):
	def __init__(self,lib,obj):
		self.__dict__.update(obj)
		self._lib=lib
		self._call=getattr(self._lib,self.mangled_name)
		self._set_return()
		
		self.sub=False
		if self.pytype=='void':
			self.sub=True
			
		if not self.sub:
			self._restype=self.ctype_def()
			self._call.restype=self._restype
		
		
			
	def _set_args_ctype(self):
		pass
		
	def _convert_args_to_ctype(self):
		pass
		
	def _get_return(self):
		pass
		
	def __call__(self,*args):
		args_in=[]
		
		#Capture stdout messages
		pipe_out,pipe_in = os.pipe()
		stdout=os.dup(1)
		os.dup2(pipe_in,1)
		
		if len(args_in)>0:
			res=self._call()(args_in)
		else:
			res=self._call()

		#Print stdout
		os.dup2(stdout,1)
		print(read_pipe(pipe_out))
		

		return res
		
	def __str__(self):
		return str("Function: "+self.name)
		
	def __repr__(self):
		s="Function: "+self.name+"("
		if len(self.args)>0:
			s=s+"Arg list"
		else:
			s=s+"None"
		s=s+")"+os.linesep
		s=s+"Args In: "+os.linesep
		s=s+"Args Out: "+os.linesep
		s=s+"Returns: "
		if self.sub:
			s=s+"None"
		else:
			s=s+str(self.pytype)
		s=s+os.linesep
		return s
