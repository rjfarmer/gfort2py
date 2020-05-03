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
from .functions import fFunc, fFuncPtr
from .strings import fStr
from .types import fDerivedType, _alldtdefs
from .utils import *
from .var import fVar, fParam
from .errors import *

from .selector import _selectVar

from . import version

from . import parseMod as pm

WARN_ON_SKIP = False

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
        self._all = {}
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
        self._init_dt_defs()

    def _init_func(self, name):
        if name in self._funcs:
            obj = self._funcs[name]
            self._all[name] = fFunc(obj)
            return self._all[name]

    def _init_dt_defs(self):
        for i in self._dt_defs.keys():
            _alldtdefs[i] = self._dt_defs[i]

    def _init_var(self, name):
        if name in self._mod_vars:
            obj = self._mod_vars[name]
            self._all[name] = self._get_fvar(obj)(obj)
            return self._all[name]

    def _init_param(self, name):
        if name in self._param:
            obj = self._param[name]
            self._all[name] = self._get_fvar(obj)(obj)
            return self._all[name]

    def _init_func_ptrs(self, name):
        if name in self._func_ptrs:
            obj = self._func_ptrs[name]
            self._all[name] = fFuncPtr(obj)
            return self._all[name]

    def __getattr__(self, name):
        if '_initialized' in self.__dict__ and self._initialized:
            nl = name.lower()
            if '_all' in self.__dict__:
                if nl in self._all:
                    return self._all[nl].in_dll(self._lib)
                else:
                    if '_mod_vars' in self.__dict__:
                        if nl in self._mod_vars:
                            return self._init_var(nl).in_dll(self._lib)
                    if '_param' in self.__dict__:
                        if nl in self._param:
                            return self._init_param(nl).in_dll(self._lib)
                    if '_funcs' in self.__dict__:
                        if nl in self._funcs:
                            return self._init_func(nl).in_dll(self._lib)
                    if '_func_ptrs' in self.__dict__:
                        if nl in self._func_ptrs:
                            return self._init_func_ptrs(nl).in_dll(self._lib)

        if name in self.__dict__:
            return self.__dict__[name]

        raise AttributeError("No variable " + name)

    def __setattr__(self, name, value):
        if '_initialized' in self.__dict__ and self._initialized:
            nl = name.lower()
            if '_all' in self.__dict__:
                if nl in self._all:
                    return self._all[nl].set_in_dll(self._lib, value)
                else:
                    if '_mod_vars' in self.__dict__:
                        if nl in self._mod_vars:
                            return self._init_var(
                                nl).set_in_dll(self._lib, value)
                    if '_param' in self.__dict__:
                        if nl in self._param:
                            return self._init_param(
                                nl).set_in_dll(self._lib, value)
                    if '_func_ptrs' in self.__dict__:
                        if nl in self._func_ptrs:
                            return self._init_func_ptrs(
                                nl).set_in_dll(self._lib, value)

        self.__dict__[name] = value
        return

    def __dir__(self):
        if self._initialized:
            l = list(self._mod_vars.keys())
            l.extend(list(self._param.keys()))
            l.extend(list(self._funcs.keys()))
            l.extend(list(self._func_ptrs.keys()))
            return l

    def __getstate__(self):
        return self._libname, self._ffile

    def __setstate__(self, state):
        self.__init__(*state)

    def _get_fvar(self, var):
        x = _selectVar(var)
        if x is None:  # Handle derived types
            if 'dt' in var['var'] and var['var']['dt']:
                x = fDerivedType
            elif ('is_func' in var['var'] and var['var']['is_func']) or ('func_arg' in var and var['func_arg']):
                x = fFuncPtr
            else:
                raise TypeError("Can't match ", var['name'])
        return x
