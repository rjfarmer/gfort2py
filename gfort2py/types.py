from __future__ import print_function
import ctypes
import functools
import collections
from .var import fVar
from .cmplx import fComplex
from .arrays import fExplicitArray, fDummyArray, fAssumedShape, fAssumedSize, fAllocatableArray
from .strings import fStr
from .errors import *


_dictAllDtDescs = {}
_dictDTDefs = {}


def getEmptyDT(name):
    class emptyDT(ctypes.Structure):
        pass
    emptyDT.__name__ = name
    emptyDT.empty = True
    return emptyDT
        
class _DTDesc(object):
    _init = False
    def __init__(self,dt_def):
        self._lib = None
        self.dt_def = dt_def['dt_def']['arg']
        self.dt_name = dt_def['name'].lower().replace("'","")
        self.empty = False
        
        self.names = [i['name'].lower().replace("'","") for i in self.dt_def]
        self.args = []
        for i in self.dt_def:
            self.args.append(self._init_var(i))
            self.args[-1]._dt_arg = True
            
        self.ctypes = []
        for i in self.args:
            try:
                self.ctypes.append(i.ctype_def())
            except AttributeError:
                if hasattr(i,'dt_desc'):
                    self.ctypes.append(i.dt_desc)
                else:
                    self.ctypes.append(ctypes.POINTER(getEmptyDT(self.dt_name)))
                
        self.dt_desc = self._create_dt()
             
    def _create_dt(self):
        class fDerivedTypeDesc(ctypes.Structure):
            _fields_ = list(zip(self.names,self.ctypes))
        fDerivedTypeDesc.__name__ = self.dt_name
        return fDerivedTypeDesc
        
    def _init_var(self, obj):
        # Placeholder for a dt
        if 'dt' in obj['var']:
            name = obj['var']['dt']['name'].lower().replace("'","")
            if _dictAllDtDescs[name].empty:
                _dictAllDtDescs[name] = _DTDesc(_dictDTDefs[name])
            return _dictAllDtDescs[name]
        
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
            
class fDerivedType(fVar):    
    def __init__(self, lib, obj):
        
        self._lib = lib

        self.__dict__.update(obj)
        self._dt_type = self.var['dt']['name'].lower().replace("'","")


        if _dictAllDtDescs[self._dt_type].empty:
            _dictAllDtDescs[self._dt_type] = _DTDesc(_dictDTDefs[self._dt_type])

        self._dt_desc = _dictAllDtDescs[self._dt_type]

        self._desc = self._dt_desc.dt_desc
        self._ctype = self._desc
        self._ctype_desc = ctypes.POINTER(self._ctype)

        self._elems=collections.OrderedDict()
        for i,j,k in zip(self._dt_desc.args,self._dt_desc.names,self._dt_desc.ctypes):
            self._elems[j]={'ctype':k,'args':i}

        self.intent=None
        self.pointer=None
        
        #Store the ref to the lib object
        try:   
            self._ref = self._get_from_lib()
        except NotInLib:
            self._ref = None
        
        self._value = collections.OrderedDict()
        
    def get(self,copy=True):
        res=collections.OrderedDict()
        if copy:
            for k,v in self._elems.items():
                x = getattr(self._ref,k)
                res[k] = v['args'].ctype_to_py_f(x)
        else:
            if hasattr(self._ref,'contents'):
                res =self._ref.contents
            else:
                res = self._ref
        return res
        
    def _set_check(self, value):
        if not all(i in self._elems.keys() for i in value.keys()):
            raise ValueError("Dict contains elements not in struct")
            
    def set_mod(self,value):
        # Wants a dict
        self._set_check(value)
        
        for name in value:
            self.set_single(name,value[name])
            
    def set_single(self,name,value):
        self._setSingle(self._ref,name,value)
        
    def _setSingle(self,v,name,value):       
        if isinstance(value,dict):
            for i in value:
                self._setSingle(getattr(v,name),i,value[i])
        else:
            if self._elems[name]['args']._array:
                setattr(v,name,self._elems[name]['args'].py_to_ctype_p(value))
            else:
                setattr(v,name,self._elems[name]['args'].py_to_ctype(value))

    def py_to_ctype(self, value):
        """
        Pass in a python value returns the ctype representation of it
        """
        self._value=self._ctype()

        # Wants a dict
        if isinstance(value,dict):
            self._set_check(value)
            
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
        res=collections.OrderedDict()            
        for k,v in self._elems.items():
            x = getattr(value,k)
            try:
                res[k] = v['args'].ctype_to_py_f(x)
            except AttributeError:
                res[k] = v['args']
            except ValueError:
                res[k] = v['args'].ctype_to_py(x) 
            
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
        return self._elems.keys()

    def __str__(self):
        return self.name+" <"+self._dt_type+" dt>"
        
    def __repr__(self):
        return self.name+" <"+self._dt_type+" dt>"
        
    def __getattr__(self, name): 
        if name in self.__dict__:
            return self.__dict__[name]

        if '_elems' in self.__dict__:
            if name in self._elems:
                return self.__getitem__(name)

    def __setattr__(self, name, value):
        if '_elems' in self.__dict__:
            if name in self._elems:
                self.set_single(name,value)
                return
        
        self.__dict__[name] = value
        return    
        
    def get_dict(self):
        """
        Return a dict with the keys set suitable for this dt
        """
        x=collections.OrderedDict()
        for i in self._elems:
            x[i]=0
        return x
        
    def __getitem__(self,name=None):
        if name is None:
            raise KeyError
        if name not in self._elems:
            raise KeyError("Name not in struct")
        
        if self._value is None or len(self._value)==0:
            r = getattr(self._ref,name)
        else:
            r = getattr(self._value,name)
        
        try:
            return self._elems[name]['args'].ctype_to_py(r)
        except AttributeError:
            return r
        
        return r

            
    def keys(self):
        return self._elems.keys()
