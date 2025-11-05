# SPDX-License-Identifier: GPL-2.0+

import typing

import gfModParser as gf

# from .fVar import fVar
from .procedures import factory as proc_factory
from .fParameters import fParam
from .types import factory as type_factory, get_module

# from .fCompile import compile_and_load, common_compile

from .compilation import Compile, CompileArgs, Modulise, factory_platform

__all__ = ["fFort", "compile"]


class fFort:
    _initialized = False

    def __init__(self, libname: str, mod_file: str):
        """
        Loads a gfortran module given by mod_file and saved in a
        shared library libname.
        """

        self._libname = libname
        self._lib = factory_platform().load_library(self._libname)
        self._mod_file = mod_file
        self._module = get_module(self._mod_file)

        self._saved_parameters = gf.Parameters(self._module)
        self._saved_variables = gf.Variables(self._module)
        self._saved_procedures = gf.Procedures(self._module)

        self._initialized = True

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
                    # return type_factory(self._module[key]).in_dll(self._lib, self._module[key].mangled_name)
                    return (
                        type_factory(self._module[key])
                        .in_dll(self._lib, self._module[key].mangled_name)
                        .value
                    )

                if key in self._saved_procedures:
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
                    type_factory(self._module[key]).in_dll(
                        self._lib, self._module[key].mangled_name
                    ).value = value

                    return
            raise AttributeError(f"Can't find symbol {key}")

        self.__dict__[key] = value

    # def __getattr__(self, key):
    #     k = key
    #     key = key.lower()

    #     if key in self.__dict__:
    #         return self.__dict__[key]

    #     if "_initialized" in self.__dict__:
    #         if self._initialized:
    #             if key not in self.keys():
    #                 raise AttributeError(f"{self._mod_file}  has no attribute {k}")

    #         if self._module[key].is_variable():
    #             if key not in self._saved:
    #                 self._saved[key] = fVar(self._module[key], allobjs=self._module)
    #             self._saved[key].in_dll(self._lib)
    #             return self._saved[key].value
    #         elif self._module[key].is_proc_pointer():
    #             # Must come before fProc
    #             if key not in self._saved:
    #                 self._saved[key] = fVar(self._module[key], allobjs=self._module)
    #             return self._saved[key]
    #         elif self._module[key].is_procedure():
    #             if key not in self._saved:
    #                 self._saved[key] = fProc(self._lib, self._module[key], self._module)
    #             return self._saved[key]
    #         elif self._module[key].is_parameter():
    #             return fParam(self._module[key]).value
    #         else:
    #             raise NotImplementedError(
    #                 f"definition type {self._module[key].flavor()} not implemented yet"
    #             )

    # def __setattr__(self, key, value):
    #     k = key
    #     key = key.lower()

    #     if key in self.__dict__:
    #         self.__dict__[key] = value
    #         return

    #     if "_initialized" in self.__dict__:
    #         if self._initialized:
    #             if self._module[key].is_variable():
    #                 if key not in self._saved:
    #                     self._saved[key] = fVar(self._module[key], allobjs=self._module)
    #                 self._saved[key].in_dll(self._lib)
    #                 self._saved[key].value = value
    #                 return
    #             elif self._module[key].is_parameter():
    #                 raise AttributeError("Can not alter a parameter")
    #             elif self._module[key].is_proc_pointer():
    #                 if key not in self._saved:
    #                     self._saved[key] = fVar(self._module[key], allobjs=self._module)
    #                 self._saved[key].value = value
    #                 return
    #             else:
    #                 raise NotImplementedError(
    #                     f"definition type {self._module[key].flavor()} not implemented yet"
    #                 )

    #     self.__dict__[key] = value

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


# Does not work needs https://gcc.gnu.org/bugzilla/show_bug.cgi?id=47030 applied

# def common(
#     string=None,
#     gfort=None,
#     *,
#     FC=None,
#     FFLAGS="-O2",
#     LDLIBS="",
#     LDFLAGS="",
#     output=None,
#     cache_folder=None,
# ):
#     """
#     Compiles and loads a snippet of Fortran code for accessing common blocks.

#     Code must be provided as a valid bit of Fortran that declares the variables
#     and declares the common block, eg:

#     fstr = "
# 		integer ::  a_int,b_int,c_int
# 		common  /common_name/ a_int,b_int,c_int
#          "

#     This code will then be converted into a Fortran module and
#     compiled.

#     gfort specifies an already loaded instance of fFort contains the common block

#     FC specifies the Fortran compiler to be used. This
#     must be some version of gfortran

#     FFLAGS specifies Fortran compile options. This defaults
#     to -O2. We will additionally insert flags for building
#     shared libraries on the current platform.

#     LDLIBS specifies any libraries to link against

#     LDFLAGS specifies extra argument to pass to the linker
#     (usually this is specifying the directory of where libraries are
#     stored and passed with the -L option)

#     output Path to store intermediate files. Defaults to None
#     where files are stored in a temp folder. Otherwise
#     stored in ``output`` folder.

#     cache_folder same as for fFort, specifies location to save cached
#     mod data to.

#     """

#     library, mod_file = common_compile(
#         string=string,
#         gfort=gfort,
#         FC=FC,
#         FFLAGS=FFLAGS,
#         LDLIBS=LDLIBS,
#         LDFLAGS=LDFLAGS,
#         output=output,
#     )

#     return fFort(library, mod_file, cache_folder=cache_folder)
