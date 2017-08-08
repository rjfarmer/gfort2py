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
        
        self._ctype = self._desc
        self._ctype_desc = ctypes.POINTER(self._ctype)
        self.TEST_FLAG=TEST_FLAG
        
    def get(self):
        r={}
        v = self._get_from_lib()
        for name in self._nameArgs:
            r[name]=getattr(v,name)
        return r
            
    def set_mod(self,value):
        # Wants a dict
        if not all(i in self._nameArgs for i in value.keys()):
            raise ValueError("Dict contains elements not in struct")
        
        for name in value:
            self.set_single(name,value[name])
            
    def set_single(self,name,value):
        if name not in self._nameArgs:
            raise KeyError("Name not in struct")
        v = self._get_from_lib()
        setattr(v,name,value)
        
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


    def py_to_ctype(self, value):
        """
        Pass in a python value returns the ctype representation of it
        """
        self._value=self._ctype()

        # Wants a dict
        if not all(i in self._nameArgs for i in value.keys()):
            raise ValueError("Dict contains elements not in struct")
        
        for name in value:
            setattr(self._value,name,value[name])
        
        return self._value
        
    def py_to_ctype_f(self, value):
        """
        Pass in a python value returns the ctype representation of it, 
        suitable for a function
        
        Second return value is anythng that needs to go at the end of the
        arg list, like a string len
        """
        r=self.py_to_ctype(value)        
        return r,None

    def ctype_to_py(self, value):
        """
        Pass in a ctype value returns the python representation of it
        """
        res={}
        for i in self._nameArgs:
            res[i]=getattr(value,i)

        return res
        
    def ctype_to_py_f(self, value):
        """
        Pass in a ctype value returns the python representation of it,
        as returned by a function (may be a pointer)
        """
        return self.ctype_to_py(value)

    def ctype_def(self):
        """
        The ctype type of this object
        """
        return self._ctype

    def ctype_def_func(self,pointer=False):
        """
        The ctype type of a value suitable for use as an argument of a function

        May just call ctype_def
        
        Second return value is anythng that needs to go at the end of the
        arg list, like a string len
        """
        if pointer:
            f=ctypes.POINTER(self._ctype_desc)
        else:
            f=self._ctype_desc
            
        return f,None
        
    def py_to_ctype_p(self,value):
        """
        The ctype representation suitable for function arguments wanting a pointer
        """

        return ctypes.POINTER(self.ctype_def())(self.py_to_ctype(value))
        

    def __dir__(self):
        return self._nameArgs

    def __str__(self):
        try:
            x=self.get()
        except:
            x=None
        if x is None:
            return "<dt>"
        else:
            return str(self.get())
        
    def __repr__(self):
        try:
            x=self.get()
        except:
            x=None
            
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
        
    def __getitem__(self,name=None):
        if name is None:
            raise KeyError
        if name not in self._nameArgs:
            raise KeyError("Name not in struct")
        
        if self._value is None:
            v = self._get_from_lib()
            return getattr(v,name)
        else:
            return getattr(self._value,name)
        
        
        
        
