import ctypes
import os
from .var import fVar
from .utils import *

class fFunc(fVar):
	def __init__(self,lib,obj):
		self.__dict__.update(obj)
		self.lib=lib
		self._call=getattr(self.lib,self.mangled_name)
		
		try:
			self._restype=self.ctype_def()
		except AttributeError:
			#Func returns void
			self._restype=None
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
		
	
