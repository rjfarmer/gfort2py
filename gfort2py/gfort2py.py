# SPDX-License-Identifier: GPL-2.0+

import ctypes
import tempfile
import typing
from pathlib import Path

import gfModParser as gf

from .compilation import Compile, CompileArgs, Modulise, factory_platform
from .procedures import factory as proc_factory
from .procedures.proc_pointer import fProcPointer
from .types import factory as type_factory
from .types import fParam, get_module
from .types.module import register_module_alias

__all__ = ["fFort", "compile"]


class fFort:
    _initialized = False

    def __init__(self, libname: str | Path, mod_file: str | Path):
        """
        Loads a gfortran module given by mod_file and saved in a
        shared library libname.
        """

        self._libname = Path(libname)
        self._mod_file = Path(mod_file)
        lib = factory_platform().load_library(self._libname)
        module = get_module(str(self._mod_file))
        self._init_from(lib, module)

    def _init_from(self, lib: ctypes.CDLL | None, module: gf.Module) -> None:
        """Shared initialiser; sets all runtime state and marks the instance ready."""
        self._lib = lib
        self._module = module

        # Symbols refer to their logical module name (e.g. "dt"); register an
        # alias to this loaded gf.Module so downstream type resolution can reuse it.
        try:
            first_key = next(iter(module.keys()))
            register_module_alias(module[first_key].module, module)
        except StopIteration:
            pass

        self._saved_parameters = gf.Parameters(module)
        self._saved_variables = gf.Variables(module)
        self._saved_procedures = gf.Procedures(module)
        self._saved_proc_pointers = {
            key
            for key in module.keys()
            if key in self._saved_procedures
            and self._module[key].properties.attributes.proc_pointer
        }
        self._initialized = True

    @classmethod
    def _create(cls, lib: ctypes.CDLL | None, module: gf.Module) -> "fFort":
        """Low-level constructor used by from_mod_string; bypasses file loading."""
        self = object.__new__(cls)
        self.__dict__["_libname"] = None
        self.__dict__["_mod_file"] = None
        self._init_from(lib, module)
        return self

    @classmethod
    def from_mod_string(cls, mod_text: str, libname: str | None = None) -> "fFort":
        """
        Create an fFort directly from the text content of a ``.mod`` file.

        Useful for unit-testing without a compiled shared library.  Pass
        *libname* to load a real ``.so``; omit it to work with parameters
        and type inspection only.
        """
        lib: ctypes.CDLL | None
        if libname is not None:
            lib = factory_platform().load_library(Path(libname))
        else:
            lib = None
        with tempfile.NamedTemporaryFile(
            suffix=".mod.txt", mode="w", delete=False
        ) as fh:
            fh.write(mod_text)
            tmp_path = Path(fh.name)
        try:
            module = gf.Module(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)
        return cls._create(lib, module)

    def keys(self) -> list[str]:
        return self._module.keys()

    def __contains__(self, key) -> bool:
        return key in self._module.keys()

    def __dir__(self):
        return self.keys()

    def __getattr__(self, key: str):
        key = key.lower()

        if "_initialized" in self.__dict__:
            if self._initialized:
                if key in self._saved_parameters:
                    return fParam(self._module[key]).value

                if key in self._saved_variables:
                    if self._lib is None:
                        raise RuntimeError(
                            "No shared library loaded; pass libname= to from_mod_string()."
                        )
                    # return type_factory(self._module[key]).in_dll(self._lib, self._module[key].mangled_name)
                    return (
                        type_factory(self._module[key])
                        .in_dll(
                            self._lib,
                            self._module[key].mangled_name,
                            symbol=self._module[key],
                        )
                        .value
                    )

                if key in self._saved_proc_pointers:
                    if self._lib is None:
                        raise RuntimeError(
                            "No shared library loaded; pass libname= to from_mod_string()."
                        )
                    return fProcPointer(self._lib, self._module[key], self._module)

                if key in self._saved_procedures:
                    if self._lib is None:
                        raise RuntimeError(
                            "No shared library loaded; pass libname= to from_mod_string()."
                        )
                    return proc_factory(self._module[key])(
                        self._lib, self._module[key], self._module
                    )

            raise AttributeError(f"Can't find symbol {key}")

    def __setattr__(self, key: str, value: typing.Any):
        key = key.lower()

        if "_initialized" in self.__dict__:
            if self._initialized:
                if key in self._saved_parameters:
                    raise AttributeError("Can not set a parameter")

                if key in self._saved_variables:
                    if self._lib is None:
                        raise RuntimeError(
                            "No shared library loaded; pass libname= to from_mod_string()."
                        )
                    type_factory(self._module[key]).in_dll(
                        self._lib,
                        self._module[key].mangled_name,
                        symbol=self._module[key],
                    ).value = value

                    return

                if key in self._saved_proc_pointers:
                    if self._lib is None:
                        raise RuntimeError(
                            "No shared library loaded; pass libname= to from_mod_string()."
                        )
                    fProcPointer(self._lib, self._module[key], self._module).bind(value)
                    return
            raise AttributeError(f"Can't find symbol {key}")

        self.__dict__[key] = value

    @property
    def __doc__(self):
        return f"MODULE={self._module.filename} LIBRARY={self._libname}"

    def __str__(self):
        return f"{self._module.filename}"


def compile(
    string=None,
    *,
    file=None,
    FC=None,
    FFLAGS="",
    LDLIBS="",
    LDFLAGS="",
    INCLUDE_FLAGS="",
):
    """
    Compiles and loads a snippet of Fortran code.

    Either provide the code as a string in the ``string``
    argument of a filename in the ``file`` argument.

    This code will then be converted into a Fortran module and
    compiled.

    FC specifies the Fortran compiler to be used. This
    must be some version of gfortran

    FFLAGS specifies Fortran compile options. This defaults
    to -O2. We will additionally insert flags for building
    shared libraries on the current platform.

    LDLIBS specifies any libraries to link against

    LDFLAGS specifies extra argument to pass to the linker
    (usually this is specifying the directory of where libraries are
    stored and passed with the -L option)

    """

    if file is not None:
        with open(file, "r") as f:
            string = "".join(f.readlines())

    mod = Modulise(text=string)

    comp = Compile(text=mod.as_module(), name=mod.strhash(), fc=FC)

    args = CompileArgs()
    if len(FFLAGS):
        args.FFLAGS = FFLAGS
    if len(LDLIBS):
        args.LDLIBS = LDLIBS
    if len(LDFLAGS):
        args.LDFLAGS = LDFLAGS
    if len(INCLUDE_FLAGS):
        args.INCLUDE_FLAGS = INCLUDE_FLAGS

    if not comp.compile(args=args):
        raise ValueError("Could not compile code")

    return fFort(comp.library_filename, comp.module_filename)
