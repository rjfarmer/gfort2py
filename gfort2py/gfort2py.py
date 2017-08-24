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
from .types import fDerivedType, _dictAllDtDescs, _DTDesc, getEmptyDT
from .utils import *
from .var import fVar, fParam
from .errors import *

from . import parseMod as pm

WARN_ON_SKIP=False

#https://gcc.gnu.org/onlinedocs/gcc-6.1.0/gfortran/Argument-passing-conventions.html

class fFort(object):

    def __init__(self, libname, ffile, rerun=False):
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
            
        self._all_names = set(self._all_names)
            
        for i in self._dt_defs:
            i['name']=i['name'].lower().replace("'","")
            
        self._init_dt_defs()
        
        for i in self._mod_vars:
            self._init_var(i)

        for i in self._param:
            self._init_param(i)

        # Must come last after the derived types are setup
        for i in self._funcs:
            self._init_func(i)

    def _init_var(self, obj):
        x=None
        if obj['var']['pytype'] == 'str':
            x = fStr(self._lib, obj)
        elif obj['var']['pytype'] == 'complex':
            x = fComplex(self._lib, obj)
        elif 'dt' in obj['var'] and obj['var']['dt']:
            x = fDerivedType(self._lib, obj)
        elif 'array' in obj['var']:
            array = obj['var']['array']['atype'] 
            if array == 'explicit':
                x = fExplicitArray(self._lib, obj)
            elif array == 'alloc' or array == 'pointer':
                x = fDummyArray(self._lib, obj)
            elif array == 'assumed_shape':
                x =  fAssumedShape(self._lib, obj)
            elif array == 'assumed_size':
                x = fAssumedSize(self._lib, obj)
        else:
            x = fVar(self._lib, obj)

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
        x = fFunc(self._lib, obj)
        self.__dict__[x.name] = x
        
    def _init_dt_defs(self):
        all_dt_defs=self._dt_defs
        
        completed = [False]*len(all_dt_defs)
        # First pass, do the very simple stuff (things wih no dt's inside them)
        for idx,i in enumerate(all_dt_defs):
            flag=True
            for j in i['dt_def']['arg']:
                if 'dt' in j['var']:
                    flag=False
            if flag:
                _dictAllDtDescs[i['name']]=_DTDesc(i)
                completed[idx]=True
                
        progress = True
        while True:     
            if all(completed):
                break
            if not progress:
                break
            progress=False
            for idx,i in enumerate(all_dt_defs):
                if completed[idx]:
                    continue
                flag=True
                for j in i['dt_def']['arg']:
                    if 'dt' in j['var']:
                        if j['var']['dt']['name'] not in _dictAllDtDescs:
                            flag=False
                            
                #All elements are either not dt's or allready in the alldict
                if flag:
                    progress = True
                    _dictAllDtDescs[i['name']]=_DTDesc(i)
                    completed[idx]=True
        
                   
        # Anything left not completed is likely to be a recurisive type
        for i,status in zip(all_dt_defs,completed):
            if not status:
                _dictAllDtDescs[i['name']]=getEmptyDT(i['name'])
        
        
        # Re-do the recurivse ones now we can add the empty dt's to them
        for i,status in zip(all_dt_defs,completed):
            if not status:
                _dictAllDtDescs[i['name']] = _DTDesc(i)

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
