# SPDX-License-Identifier: GPL-2.0+
import ctypes
import numpy as np
import os

from .module_parse import module

from .fDT import _all_dts, make_dt
from .fVar import fVar
from .fProc import fProc
from .fParameters import fParam

_TEST_FLAG = os.environ.get("_GFORT2PY_TEST_FLAG") is not None


class fFort:
    _initialized = False

    def __init__(self, libname, mod_file):
        self._lib = ctypes.CDLL(libname)
        self._mod_file = mod_file
        self._module = module(self._mod_file)

        self._saved = {}
        self._initialized = True

        _all_dts_full = False
        while not _all_dts_full:
            _all_dts_full = True
            for key in self._module.keys():
                sym = self._module[key]
                if (len(sym.dt_components()) > 0
                    and not sym.name.startswith("_")
                ):
                    if sym.name not in _all_dts:
                        fields = []
                        all_components_found = True
                        for var in sym.dt_components():
                            if not var.is_derived():
                                fields.append(
                                    (
                                        var.name,
                                        fVar(var, allobjs=self._module).ctype()
                                    )
                                )
                            else:
                                var_dt_name = self._module[var.dt_type()].name
                                if var_dt_name not in _all_dts:
                                    if var.is_pointer():
                                        _all_dts[var_dt_name] = ctypes.pointer
                                    else:
                                        _all_dts_full = False
                                        all_components_found = False
                                        break
                                else:
                                    fields.append((var.name, _all_dts[var_dt_name]))

                        if all_components_found:
                            _all_dts[sym.name] = make_dt(sym.name)
                            _all_dts[sym.name]._fields_ = fields

    def keys(self):
        return self._module.keys()

    def __contains__(self, key):
        return key in self._module.keys()

    def __dir__(self):
        return self.keys()

    def __getattr__(self, key):
        if key in self.__dict__:
            return self.__dict__[key]

        if "_initialized" in self.__dict__:
            if self._initialized:
                if key not in self.keys():
                    raise AttributeError(f"{self._mod_file}  has no attribute {key}")

            if self._module[key].is_variable():
                if key not in self._saved:
                    self._saved[key] = fVar(self._module[key], allobjs=self._module)
                self._saved[key].in_dll(self._lib)
                return self._saved[key].value
            elif self._module[key].is_proc_pointer():
                # Must come before fProc
                if key not in self._saved:
                    self._saved[key] = fVar(self._module[key], allobjs=self._module)
                return self._saved[key]
            elif self._module[key].is_procedure():
                return fProc(self._lib, self._module[key], self._module)
            elif self._module[key].is_parameter():
                return fParam(self._module[key]).value
            else:
                raise NotImplementedError(
                    f"Object type {self._module[key].flavor()} not implemented yet"
                )

    def __setattr__(self, key, value):
        if key in self.__dict__:
            self.__dict__[key] = value
            return

        if "_initialized" in self.__dict__:
            if self._initialized:
                if self._module[key].is_variable():
                    if key not in self._saved:
                        self._saved[key] = fVar(self._module[key], allobjs=self._module)
                    self._saved[key].in_dll(self._lib)
                    self._saved[key].value = value
                    return
                elif self._module[key].is_parameter():
                    raise AttributeError("Can not alter a parameter")
                elif self._module[key].is_proc_pointer():
                    if key not in self._saved:
                        self._saved[key] = fVar(self._module[key], allobjs=self._module)
                    self._saved[key].value = value
                    return
                else:
                    raise NotImplementedError(
                        f"Object type {self._module[key].flavor()} not implemented yet"
                    )

        self.__dict__[key] = value

    @property
    def __doc__(self):
        return f"MODULE={self._module.filename}"

    def __str__(self):
        return f"{self._module.filename}"


def mod_info(mod_file):
    return module(mod_file)
