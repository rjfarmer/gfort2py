from __future__ import print_function
import ctypes
from .var import fVar


class fDerivedType(fVar):
    def __init__(self, lib, obj,TEST_FLAG=False):
        self.__dict__.update(obj)
        self._lib = lib
        self._args = []
        self._nameArgs = []
        self._typeArgs = []
        
        self._desc = self.create_struct()
        self._ctype = ctypes.POINTER(self._desc)
        self._ctype_desc = self._desc()
        self.TEST_FLAG=TEST_FLAG
        
    def get(self):
        r={}
        for i in self._nameArgs:
            r[i]=self.get_single(i)
        return r
            
    def get_single(self,name):
        r = self._get_pointer()
        return getattr(r.contents,name)

    def _get_pointer(self):
        if '_func_arg' in self.__dict__:
            if self._func_arg:
                return self._desc.from_address(ctypes.addressof(self._ctype_desc))
        
        #return self._ctype.from_address(ctypes.addressof(getattr(self._lib,self.mangled_name)))
        return self._desc.from_address(ctypes.addressof(getattr(self._lib,self.mangled_name)))



    def set_mod(self,value):
        # Wants a dict
        if not all(i in self._nameArgs for i in value.keys()):
            raise ValueError("Dict contains elements not in struct")
        
        for name in value:
            self.set_single(name,value[name])
            
    def set_single(self,name,value):
        if name not in self._nameArgs:
            raise KeyError("Name not in struct")
        r = self._get_pointer()
        setattr(r,name,value)
        
    def create_struct(self):
        self.setup_desc()
        class fDerivedTypeDesc(ctypes.Structure):
            _fields_ = self.fields
        fDerivedTypeDesc.__name__ = str(self._dt_def['name'])
        return fDerivedTypeDesc
        
    
    def setup_desc(self):
        for i in self._dt_def['dt_def']['arg']:
            self._args.append(fVar(self._lib, i))
            self._args[-1]._dt_arg=True         
            self._nameArgs.append(self._args[-1].name.replace("\'", ''))
            #Overload the mangled name so we can use the get from fVar 
            self._args[-1].mangled_name=self._nameArgs[-1]
            self._typeArgs.append(self._args[-1]._ctype)

        self.set_fields(self._nameArgs, self._typeArgs)
 
    def set_fields(self, nameArgs, typeArgs):
        self.fields = [(i, j) for i, j in zip(nameArgs, typeArgs)]


    def py_to_ctype_f(self,value):
        self.set_mod(value)
        return self._ctype_desc,None


    def ctype_def_func(self):
        """
        The ctype type of a value suitable for use as an argument of a function

        May just call ctype_def
        
        Second return value is anythng that needs to go at the end of the
        arg list, like a string len
        """

        return self._desc,None

    def __dir__(self):
        return self._nameArgs

    def __str__(self):
        x=self.get()
        if x is None:
            return "<dt>"
        else:
            return str(self.get())
        
    def __repr__(self):
        x=self.get()
        if x is None:
            return "<dt>"
        else:
            return repr(self.get())
        
    def __getattr__(self, name): 
        if name in self.__dict__:
            return self.__dict__[name]

        if '_args' in self.__dict__ and '_nameArgs' in self.__dict__:
            if name in self._nameArgs:
                return self.get_single(name)

    def __setattr__(self, name, value):
        if '_nameArgs' in self.__dict__:
            if name in self._nameArgs:
                self.set_single(name,value)
                return
        
        self.__dict__[name] = value
        return    
        
    def get_dict(self):
        """
        Return a dict with the keys set suitable for this dt
        """
        x={}
        for i in self._nameArgs:
            x[i]=0
        return x
        
