# SPDX-License-Identifier: GPL-2.0+
from __future__ import print_function
try:
    import __builtin__
except ImportError:
    import builtins as __builtin__

import ctypes
import pickle
import numpy as np
import errno
import sys

from .cmplx import fComplex, fParamComplex
from .arrays import fExplicitArray, fDummyArray, fParamArray
from .functions import fFunc,  _allFuncs
from .strings import fStr
# from .types import fDerivedType, _dictAllDtDescs, getEmptyDT, _dictDTDefs
from .utils import *
from .var import fVar, fParam
from .errors import *

from .selector import _selectVar

from . import version

from . import parseMod as pm

WARN_ON_SKIP=False

if sys.version_info[0] < 3:
    FileNotFoundError = IOError
    
    
class fFort(object):
    _initialized = False

    def __init__(self, libname, ffile, rerun=False):
        self._lib = ctypes.CDLL(libname)
        self._all_names = []
        self._libname = libname
        self._ffile = ffile
        self._fpy = pm.fpyname(ffile)
        self._load_data(ffile, rerun)
        self._init()
        self._initialized = True

    def _load_data(self, ffile, rerun=False):
        try:
            f = open(self._fpy, 'rb')
        except FileNotFoundError as e: 
            if e.errno != errno.ENOENT:
                raise 
            pm.run(ffile, save=True)
        else:
            f.close()
    
        with open(self._fpy, 'rb') as f:
            self.version = pickle.load(f)
            if self.version == version.__version__:
                self._mod_data = pickle.load(f)
    
                if self._mod_data["checksum"] != pm.hashFile(ffile) or rerun:
                    self._rerun(ffile)
                else:
                    self._mod_vars = pickle.load(f)
                    self._param = pickle.load(f)
                    self._funcs = pickle.load(f)
                    self._dt_defs = pickle.load(f)
                    self._func_ptrs = pickle.load(f)
            else:
                self._rerun(ffile)
                
    def _rerun(self, ffile):
        x = pm.run(ffile, save=True, unpack=True)
        self._mod_data = x[0]
        self._mod_vars = x[1]
        self._param = x[2]
        self._funcs = x[3]
        self._dt_defs = x[4]
        self._func_ptrs = x[5]

    def _init(self):      
        #init_mod_arrays(self._mod_data['version'])
        #self._init_dt_defs()
        pass

    def _init_var(self, obj):
        x = _selectVar(obj)(self._lib, obj)

        if x is not None:
            self.__dict__[x.name.lower()] = x
        else:
            print("Skipping init " + obj['name'])

    def _init_func(self, obj):
        x = fFunc(self._lib, obj)
        _allFuncs[x.name.lower()] = x
        self.__dict__[x.name.lower()] = x
        
    def _init_func_ptr(self, obj):
        x =  fFuncPtr(self._lib, obj)
        self.__dict__[x.name.lower()] = x
        
    def _init_dt_defs(self):
        # Make empty dts first
        for i in self._dt_defs.keys():
            _dictAllDtDescs[i] = getEmptyDT(i)
            _dictDTDefs[i] = self._dt_defs[i]


    def __getattr__(self, name):
        if '_initialized' in self.__dict__ and self._initialized:
            nl = name.lower()
            if '_mod_vars' in self.__dict__:
                if nl in self._mod_vars:
                    if nl not in self.__dict__:
                        self._init_var(self._mod_vars[nl])
                    return self.__dict__[nl]
            if '_param' in self.__dict__:
                if nl in self._param:
                    if nl not in self.__dict__:
                        self._init_var(self._param[nl])
                    return self.__dict__[nl]
            if '_funcs' in self.__dict__:
                if nl in self._funcs:
                    if nl not in self.__dict__:
                        self._init_func(self._funcs[nl])
                    return self.__dict__[nl]
            # if '_func_ptrs' in self.__dict__:
                # if nl in self._func_ptrs:
                    # if nl not in self.__dict__:
                        # self._init_func_ptr(self._func_ptrs[nl])
                    # return self.__dict__[nl]
        
        if name in self.__dict__:
            return self.__dict__[name]
       

        raise AttributeError("No variable " + name)

    def __setattr__(self, name, value):
        nl = name.lower()
        if name in self.__dict__ or nl in self.__dict__:
            try:
                self.__dict__[nl].set(value)
            except AttributeError:
                self.__dict__[name] = value
        else:
            if self._initialized:
                if '_mod_vars' in self.__dict__:
                    if nl in self._mod_vars:
                        self._init_var(self._mod_vars[nl])
                        self.__dict__[nl].set(value)
                        return
                if '_param' in self.__dict__:
                    if nl in self._param:
                        self._init_var(self._param[nl])
                        self.__dict__[nl].set(value)
                        return
                # if '_func_ptrs' in self.__dict__:
                    # if nl in self._func_ptrs:
                        # self._init_func_ptr(self._func_ptrs[nl])
                        # self.__dict__[nl].set_mod(value)
                        # return
       
            self.__dict__[name] = value
        return
        
    def __dir__(self):
        if self._initialized:
            l = list(self._mod_vars.keys()) 
            l.extend(list(self._param.keys()))
            l.extend(list(self._funcs.keys()))
            # l.extend(list(self._func_ptrs.keys()))
            return l
            
    def __getstate__(self):
        return self._libname, self._ffile

    def __setstate__(self,state):
        self.__init__(*state)

