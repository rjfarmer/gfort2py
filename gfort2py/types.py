from __future__ import print_function
import ctypes
import functools
from .var import fVar
from .cmplx import fComplex
from .arrays import fExplicitArray, fDummyArray, fAssumedShape, fAssumedSize, fAllocatableArray
from .strings import fStr
from .errors import *

class fDerivedType(fVar):
    def __init__(self, lib, obj,dt_defs,_dt_contained=[]):
        self.__dict__.update(obj)
        self._lib = lib
        self._args = []
        self._nameArgs = []
        self._typeArgs = []
        self._dt_defs = dt_defs
        self._dt_contained=[]
        self._dt_contained=self._dt_contained+_dt_contained
        self._resolveDT()
        
        self._desc = self.create_struct()
        
        self._ctype = self._desc
        self._ctype_desc = ctypes.POINTER(self._ctype)
        self.intent=None
        self.pointer=None
        
        #Store the ref to the lib object
        try:   
            self._ref = self._get_from_lib()
        except NotInLib:
            self._ref = None
        
    def get(self,copy=True):
        res={}
        if copy:
            for name,i in zip(self._nameArgs,self._args):
                x=getattr(self._ref,name)
                res[name]=i.ctype_to_py_f(x)
        else:
            if hasattr(self._ref,'contents'):
                res =self._ref.contents
            else:
                res = self._ref
        return res
            
    def set_mod(self,value):
        # Wants a dict
        if not all(i in self._nameArgs for i in value.keys()):
            raise ValueError("Dict contains elements not in struct")
        
        for name in value:
            self.set_single(name,value[name])
            
    def set_single(self,name,value):
        self._setSingle(self._ref,name,value)
        
    def _setSingle(self,v,name,value):
        if isinstance(value,dict):
            for i in value:
                self._setSingle(getattr(v,name),i,value[i])
        else:
            setattr(v,name,value)

    def create_struct(self):
        self.setup_desc()
        class fDerivedTypeDesc(ctypes.Structure):
            _fields_ = self.fields
        fDerivedTypeDesc.__name__ = str(self._dt_def['name'])
        return fDerivedTypeDesc
        
    def setup_desc(self):
        for i in self._dt_def['dt_def']['arg']:            
            ct = i['var']['ctype']
            if ct == 'c_void_p' and 'dt' in i['var']:
                self._dt_contained.append((int(self._dt_def['num']),self._dt_def['name']))
                name=None
                for j in self._dt_contained:
                    if j[0] == int(i['var']['dt']['num']):
                        name=j[1]

                if name is not None:
                    # when nesting dt's we want to prevent recurisve loops if either a dt conatins itself
                    # it contains dt_A contains dt_b which contains dt_A
                    self._args.append(emptyfDerivedType(name))
                    self._args[-1].create_struct()
                else:
                    self._args.append(fDerivedType(self._lib,i,self._dt_defs,self._dt_contained))
                    self._args[-1].setup_desc()
            else:
                self._args.append(self._init_var(i))
                ct=self._args[-1]._ctype
                
            self._args[-1]._dt_arg=True         
            self._nameArgs.append(self._args[-1].name.replace("\'", ''))
            #Overload the mangled name so we can use the get from fVar 
            self._args[-1].mangled_name=self._nameArgs[-1]
            self._typeArgs.append(self._args[-1]._ctype)
                
        self.set_fields(self._nameArgs, self._typeArgs)
        
    def _init_var(self, obj):
        array = None
        if 'array' in obj['var']:
            array = obj['var']['array']
        
        pytype = obj['var']['pytype'] 
        
        if pytype in 'str':
            return fStr(self._lib, obj)
        elif pytype in 'complex':
            return fComplex(self._lib, obj)
        elif array is not None:
            atype = array['atype']
            if atype in 'explicit':
                return fExplicitArray(self._lib, obj)
            elif atype in 'alloc':
               return fAllocatableArray(self._lib, obj)
            elif atype in 'assumed_shape' or atype in 'pointer':
                return fAssumedShape(self._lib, obj)
            elif atype in 'assumed_size':
                return fAssumedSize(self._lib, obj)
            else:
                raise ValueError("Unknown array: "+str(obj))
        else:
           return fVar(self._lib, obj)

 
    def set_fields(self, nameArgs, typeArgs):
        self.fields = list(zip(nameArgs, typeArgs))

    def py_to_ctype(self, value):
        """
        Pass in a python value returns the ctype representation of it
        """
        self._value=self._ctype()

        # Wants a dict
        if isinstance(value,dict):
            if not all(i in self._nameArgs for i in value.keys()):
                raise ValueError("Dict contains elements not in struct")
            
            for name in value.keys():
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
        for name,i in zip(self._nameArgs,self._args):
            x=getattr(value,name)
            res[name]=i.ctype_to_py_f(x)

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

    def ctype_def_func(self,pointer=False,intent=''):
        """
        The ctype type of a value suitable for use as an argument of a function

        May just call ctype_def
        
        Second return value is anything that needs to go at the end of the
        arg list, like a string len
        """
        self.intent=intent
        self.pointer=pointer
        if pointer and intent is not 'na':
            f=ctypes.POINTER(self._ctype_desc)
        elif intent=='na':
            f=ctypes.POINTER(self._ctype_desc)
        else:
            f=self._ctype_desc
            
        return f,None
        
    def py_to_ctype_p(self,value):
        """
        The ctype representation suitable for function arguments wanting a pointer
        """
        return ctypes.POINTER(self.ctype_def())(self.py_to_ctype(value))
        
    def _pname(self):
        return str(self.name) + " <" + str(self._dt_def['name']) + ">"

    def __dir__(self):
        return self._nameArgs

    def __str__(self):
        return self._dt_def['name']+" <dt>"
        
    def __repr__(self):
        return self._dt_def['name']+" <dt>"
        
    def __getattr__(self, name): 
        if name in self.__dict__:
            return self.__dict__[name]

        if '_args' in self.__dict__ and '_nameArgs' in self.__dict__:
            if name in self._nameArgs:
                return self.__getitem__(name)

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
            return getattr(self._ref,name)
        else:
            return getattr(self._value,name)
        
    def _resolveDT(self):
        name = self.var['dt']['name']
        name = name.lower().replace("\'","")
        for j in self._dt_defs:
            if j['name']==name:
                self._dt_def = j
                return
        raise KeyError("Couldn't match "+ str(name))



class emptyfDerivedType(fDerivedType):
    def __init__(self,name, *args,**kwargs):
        self._desc = self.create_struct()
        self.name = name
        
        self._desc = self.create_struct()
        self._ctype = self._desc
        self._ctype_desc = ctypes.POINTER(self._ctype)

    def create_struct(self):
        fDerivedTypeDesc = _emptyDT
        fDerivedTypeDesc.__name__ = str(self.name)
        return fDerivedTypeDesc
        
