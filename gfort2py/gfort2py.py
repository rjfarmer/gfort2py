# SPDX-License-Identifier: GPL-2.0+
import ctypes
import numpy as np
import os
import platform

from .module_parse import module

from .fVar import fVar
from .fProc import fProc
from .fParameters import fParam
from .fCompile import compile_and_load
from .utils import library_ext

_TEST_FLAG = os.environ.get("_GFORT2PY_TEST_FLAG") is not None


class fFort:
    _initialized = False

    def __init__(self, libname, mod_file, cache_folder=None):
        """
        Loads a gfortran module given by mod_file and saved in a
        shared library libname.

        cache_folder: If not None, sets the folder location to saved
        the cached module data to. If None uses appdirs ``user_cache_dir``
        location.

        Set to False to disable caching.

        """

        self._lib = ctypes.CDLL(libname)
        self._mod_file = mod_file
        self._module = module(self._mod_file, cache_folder=cache_folder)

        self._saved = {}
        self._initialized = True

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
                if key not in self._saved:
                    self._saved[key] = fProc(self._lib, self._module[key], self._module)
                return self._saved[key]
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


def mod_info(mod_file, load_only=False):
    """
    Returns a parsed data structure that describes the module

    pprint is recommened to help understand the nested structure.
    """
    return module(mod_file, load_only=load_only)


def lib_ext():
    """
    Determine shared library extension for the current OS
    """
    return library_ext()


def compile(
    string=None,
    file=None,
    FC="/usr/bin/gfortran",
    FFLAGS="-O2",
    LDLIBS="",
    LDFLAGS="",
    output=None,
):
    """
    Compiles and loads a snippet of Fortran code.

    Either provide the code as a string in the ``string``
    argument of a filename in the ``file`` argument.

    This code will then be converted into a Fortran module and
    compiled.

    FC specifies the Fortran compilier to be used. This
    must be some version of gfortran

    FFLAGS specifies Fortran compile options. This defaults
    to -O2. We will additionaly insert flags for buidling
    shared libraries on the current platform.

    LDLIBS specifies any libraries to link agaisnt

    LDFLAGS specifies extra argument to pass to the linker
    (usally this is specifing the directory of where libraies are
    stored and passed with the -L option)

    output Path to store intermediate files. Defaults to None
    where files are stored in a temp folder. Otherwise
    stored in ``output`` folder.

    """

    library, mod_file = compile_and_load(
        string=string,
        file=file,
        FC=FC,
        FFLAGS=FFLAGS,
        LDLIBS=LDLIBS,
        LDFLAGS=LDFLAGS,
        output=output,
    )

    return fFort(library, mod_file)
