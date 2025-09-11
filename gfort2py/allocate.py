# SPDX-License-Identifier: GPL-2.0+

import ctypes

from . import utils
from . import fCompile


class AllocationError(Exception):
    pass


def sizeof_allocatable(dims):
    shape = ",".join([":"] * dims)

    code = f"""
        type :: myType
            integer,allocatable,dimension({shape}) :: i
        end type
        type(myType) :: x

        write(*,*) sizeof(x)
        """

    return fCompile.program_run_compile(code)


def allocate_var(var, kind, type, shape, default=0):

    dims = ",".join([":"] * len(shape))
    shape = ",".join([str(i) for i in shape])

    code = f"""
        {type}(kind={kind}), allocatable, dimension({dims}) , intent(out) :: x
        if(allocated(x)) deallocate(x)
        allocate(x({shape}))
        x={default}
        """
    args = "x"

    def runner(name, library):
        try:
            lib = utils.load_lib(library)
        except Exception:
            raise AllocationError(
                "Allocation failed, please open bug report for gfort2py."
            )
        sub = getattr(lib, name)
        sub(ctypes.byref(var))

    return fCompile.subroutine_run_compile(code, args, runner)


def allocate_char(var, shape, length, default='""', kind=None):

    dims = ",".join([":"] * len(shape))

    shape = ",".join([str(i) for i in shape])

    code = f"""
        character(len={length}), allocatable, dimension({dims}) , intent(out) :: x
        if(allocated(x)) deallocate(x)
        allocate(x({shape}))
        x={default}
        """
    args = "x"

    def runner(name, library):
        try:
            lib = utils.load_lib(library)
        except Exception:
            raise AllocationError(
                "Allocation failed, please open bug report for gfort2py."
            )
        sub = getattr(lib, name)
        sub(ctypes.byref(var))

    return fCompile.subroutine_run_compile(code, args, runner)
