import ctypes
from .var import fVar

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
