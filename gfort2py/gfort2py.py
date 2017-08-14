from __future__ import print_function
try:
    import __builtin__
except ImportError:
    import builtins as __builtin__

import ctypes
import pickle
import numpy as np
import errno

#from .parseMod import run_and_save, hash_file, fpyname
from .cmplx import fComplex, fParamComplex
from .arrays import fExplicitArray, fDummyArray, fParamArray, fAssumedShape , fAssumedSize
from .functions import fFunc
from .strings import fStr
from .types import fDerivedType
from .utils import *
from .var import fVar, fParam

from . import parseMod as pm

WARN_ON_SKIP=False

#https://gcc.gnu.org/onlinedocs/gcc-6.1.0/gfortran/Argument-passing-conventions.html

class fFort(object):

    def __init__(self, libname, ffile, rerun=False,TEST_FLAG=False):
        self.TEST_FLAG=TEST_FLAG
        self._lib = ctypes.CDLL(libname)
        self._all_names=[]
        self._libname = libname
        self._fpy = pm.fpyname(ffile)
        self._load_data(ffile, rerun)
        self._init()

    def _load_data(self, ffile, rerun=False):
        try:
            f = open(self._fpy, 'rb')
        # FileNotFoundError does not exist on Python < 3.3
        except (OSError, IOError) as e: 
            if e.errno != errno.ENOENT:
                raise
            pm.run(ffile,save=True)
        else:
            f.close()

        with open(self._fpy, 'rb') as f:
            self.version = pickle.load(f)
            if self.version == 1:
                self._mod_data = pickle.load(f)

                if self._mod_data["checksum"] != pm.hashFile(ffile) or rerun:
                    x = pm.run(ffile,save=True,unpack=True)
                    self._mod_data = x[0]
                    self._mod_vars = x[1]
                    self._param = x[2]
                    self._funcs = x[3]
                    self._dt_defs = x[4]
                else:
                    self._mod_vars = pickle.load(f)
                    self._param = pickle.load(f)
                    self._funcs = pickle.load(f)
                    self._dt_defs = pickle.load(f)

    def _init(self):
        self._listVars = []
        self._listParams = []
        self._listFuncs = []
        
        for i in self._mod_vars:
            self._all_names.append(i['name'])

        for i in self._param:
            self._all_names.append(i['name'])
            
        for i in self._funcs:
            self._all_names.append(i['name'])
            
        for i in self._dt_defs:
            i['name']=i['name'].lower().replace("'","")
        
        for i in self._mod_vars:
            self._init_var(i)

        for i in self._param:
            self._init_param(i)

    
        for i in self._funcs:
            if 'arg' in i and len(i['arg'])>0:
                for k in i['arg']:
                    if 'dt' in k['var']:
                        k['var']['dt']['name']=k['var']['dt']['name'].lower().replace("'","")
                        for j in self._dt_defs:
                            if k['var']['dt']['name'].lower() == j['name'].lower():
                                k['_dt_def'] = j  

        # Must come last after the derived types are setup
        for i in self._funcs:
            self._init_func(i)

    def _init_var(self, obj):
        x=None
        if obj['var']['pytype'] == 'str':
            x = fStr(self._lib, obj,self.TEST_FLAG)
        elif obj['var']['pytype'] == 'complex':
            x = fComplex(self._lib, obj,self.TEST_FLAG)
        elif 'dt' in obj['var'] and obj['var']['dt']:
            x = fDerivedType(self._lib, obj,self._dt_defs,self.TEST_FLAG)
        elif 'array' in obj['var']:
            array = obj['var']['array']['atype'] 
            if array == 'explicit':
                x = fExplicitArray(self._lib, obj,self.TEST_FLAG)
            elif array == 'alloc' or array == 'pointer':
                x = fDummyArray(self._lib, obj, self.TEST_FLAG)
            elif array == 'assumed_shape':
                x =  fAssumedShape(self._lib, obj, self.TEST_FLAG)
            elif array == 'assumed_size':
                x = fAssumedSize(self._lib, obj, self.TEST_FLAG)
        else:
            x = fVar(self._lib, obj,self.TEST_FLAG)

        if x is not None:
            self.__dict__[x.name] = x
        else:
            print("Skipping init "+obj['name'])

    def _init_param(self, obj):
        if obj['param']['pytype']=='complex':
            x = fParamComplex(self._lib, obj)
        elif obj['param']['array']:
            x = fParamArray(self._lib, obj)
        else:
            x = fParam(self._lib, obj)

        self.__dict__[x.name] = x

    def _init_func(self, obj):
        x = fFunc(self._lib, obj,self._dt_defs,self.TEST_FLAG)
        self.__dict__[x.name] = x

    def __getattr__(self, name):
        if name.lower() in self.__dict__:
            return self.__dict__[name.lower()]

        if '_all_names' in self.__dict__:
            if name.lower() in self._all_names:
                return self.__dict__[name.lower()].get()

        raise AttributeError("No variable " + name)

    def __setattr__(self, name, value):
        if '_all_names' in self.__dict__:
            if name in self._all_names:
                self.__dict__[name].set_mod(value)
                return
       
        self.__dict__[name] = value
        return
