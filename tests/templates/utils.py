import string
import textwrap
import random
import numpy as np

from dataclasses import dataclass

import builtins
import contextlib


INDENT = " " * 4

py_proc = string.Template(
    textwrap.dedent(
        """
            def test_${function}(self):
            """
    ).strip()
)

py_proc_call = string.Template(
    textwrap.dedent(
        """
            result = x.${proc}(${args})
            """
    ).strip()
)

py_proc_return = string.Template(
    textwrap.dedent(
        """
            assert result.result
            """
    ).strip()
)

py_proc_arg_check = string.Template(
    textwrap.dedent(
        """
            assert result.args['${name}'] == ${value}
            """
    ).strip()
)

py_proc_arg_array_check = string.Template(
    textwrap.dedent(
        """
            assert np.array_equal(result.args['${name}'], ${value})
            """
    ).strip()
)

py_proc_stdout = string.Template(
    textwrap.dedent(
        """
            def test_${function}(self,capfd):
            """
    ).strip()
)

py_capture_stdout = string.Template(
    textwrap.dedent(
        """
            out, err = capfd.readouterr()
            """
    ).strip()
)

py_comp_stdout = string.Template(
    textwrap.dedent(
        """
            assert out.replace('\n','') == ${value}
            """
    ).strip()
)

py_value_set = string.Template(
    textwrap.dedent(
        """
            x.${name} = ${value}
            """
    ).strip()
)

py_value_comp = string.Template(
    textwrap.dedent(
        """
            assert x.${name} == ${value}
            """
    ).strip()
)

py_fail = string.Template(
    textwrap.dedent(
        """with pytest.raises(AttributeError) as cm:
                x.${var} = ${bad}
            """
    )
)

py_array_check = string.Template(
    textwrap.dedent(
        """
            assert np.allclose(x.${name}, ${value})
            """
    ).strip()
)

py_bool_true = string.Template(
    textwrap.dedent(
        """
            assert x.${name}
            """
    ).strip()
)

py_bool_false = string.Template(
    textwrap.dedent(
        """
            assert not x.${name}
            """
    ).strip()
)

fort_param = string.Template(
    textwrap.dedent(
        """
    ${type}(${kind}),parameter :: ${var} = ${value}
"""
    ).strip()
)

fort_bool_param = string.Template(
    textwrap.dedent(
        """
    logical,parameter :: ${var} = ${value}
"""
    ).strip()
)

fort_bool_var = string.Template(
    textwrap.dedent(
        """
    logical :: ${var}
"""
    ).strip()
)

fort_param_array_sing = string.Template(
    textwrap.dedent(
        """
    ${type}(${kind}),parameter,dimension(${shape}) :: ${var} = ${value}
"""
    ).strip()
)

fort_param_array_multi = string.Template(
    textwrap.dedent(
        """
    ${type}(${kind}),parameter,dimension(${shape}) :: ${var} = reshape( (/ ${value}/), shape(${var}))
"""
    ).strip()
)

fort_bool_param_array_multi = string.Template(
    textwrap.dedent(
        """
    logical,parameter,dimension(${shape}) :: ${var} = reshape( (/ ${value}/), shape(${var}))
"""
    ).strip()
)


fort_set_var = string.Template(
    textwrap.dedent(
        """
    ${type}(${kind}) :: ${var} = ${value}
"""
    ).strip()
)

fort_set_array_sing = string.Template(
    textwrap.dedent(
        """
    ${type}(${kind}),dimension(${shape}) :: ${var} = ${value}
"""
    ).strip()
)

fort_set_array_multi = string.Template(
    textwrap.dedent(
        """
    ${type}(${kind}),dimension(${shape}) :: ${var} = reshape( (/ ${value}/), shape(${var}))
"""
    ).strip()
)


fort_var = string.Template(
    textwrap.dedent(
        """
    ${type}(${kind}) :: ${var}
"""
    ).strip()
)

fort_array = string.Template(
    textwrap.dedent(
        """
    ${type}(${kind}),dimension(${shape}) :: ${var} 
"""
    ).strip()
)


fort_var_opt = string.Template(
    textwrap.dedent(
        """
    ${type}(${kind}),optional :: ${var}
"""
    ).strip()
)

fort_array_opt = string.Template(
    textwrap.dedent(
        """
    ${type}(${kind}),dimension(${shape}),optional :: ${var} 
"""
    ).strip()
)


fort_var_value = string.Template(
    textwrap.dedent(
        """
    ${type}(${kind}),value :: ${var}
"""
    ).strip()
)

fort_array_value = string.Template(
    textwrap.dedent(
        """
    ${type}(${kind}),dimension(${shape}),value :: ${var} 
"""
    ).strip()
)


fort_array_allocatable = string.Template(
    textwrap.dedent(
        """
    ${type}(${kind}),dimension(${shape}),allocatable :: ${var} 
"""
    ).strip()
)

fort_array_pointer = string.Template(
    textwrap.dedent(
        """
    ${type}(${kind}),dimension(${shape}),pointer :: ${var} 
"""
    ).strip()
)

fort_array_target = string.Template(
    textwrap.dedent(
        """
    ${type}(${kind}),dimension(${shape}),target :: ${var} 
"""
    ).strip()
)


fort_func_start = string.Template(
    textwrap.dedent(
        """
    function ${name}(${args}) result(${result}) 
        implicit none
"""
    ).strip()
)

fort_func_end = string.Template(
    textwrap.dedent(
        """
    end function ${name}
"""
    ).strip()
)


fort_sub_start = string.Template(
    textwrap.dedent(
        """
    subroutine ${name}(${args}) 
        implicit none
"""
    ).strip()
)

fort_sub_end = string.Template(
    textwrap.dedent(
        """
    end subroutine ${name}
"""
    ).strip()
)


fort_var_set = string.Template(
    textwrap.dedent(
        """
    ${name} = ${value}
"""
    ).strip()
)

fort_var_comp = string.Template(
    textwrap.dedent(
        """
    if(${name}/=${value}) return
"""
    ).strip()
)


fort_var_write = string.Template(
    textwrap.dedent(
        """
   write(*,*) ${name}
"""
    ).strip()
)

fort_module_start = string.Template(
    textwrap.dedent(
        """
! SPDX-License-Identifier: GPL-2.0+
! This file is auto generated do not edit by hand
module ${name}
    implicit none

    integer, parameter :: r4 = selected_real_kind(4)
    integer, parameter :: r8 = selected_real_kind(8)
    integer, parameter :: r16 = selected_real_kind(32)

    integer, parameter :: i1 = selected_int_kind(1)
    integer, parameter :: i2 = selected_int_kind(2)
    integer, parameter :: i4 = selected_int_kind(4)
    integer, parameter :: i8 = selected_int_kind(8)
                                            
    integer, parameter :: c1 = selected_char_kind ("ascii")
    integer, parameter :: c4  = selected_char_kind ('ISO_10646')
"""
    )
)


fort_module_mid = string.Template(
    textwrap.dedent(
        """

contains

"""
    )
)


fort_module_end = string.Template(
    textwrap.dedent(
        """
end module ${name}
"""
    )
)


py_module_start = string.Template(
    textwrap.dedent(
        """
# SPDX-License-Identifier: GPL-2.0+
# This file is auto generated do not edit by hand

import os, sys
import ctypes

os.environ["_GFORT2PY_TEST_FLAG"] = "1"

import numpy as np
import gfort2py as gf

import pytest

SO = f"./tests/${libname}.{gf.lib_ext()}"
MOD = "./tests/${modname}.mod"

x = gf.fFort(SO, MOD)

class Test_${modname}:
"""
    )
)


fort_int_kinds = {
    "i1": 1,
    "i2": 2,
    "i4": 4,
    "i8": 8,
}

fort_real_kinds = {
    "r4": 4,
    "r8": 8,
    #    'r16' :32,
}

fort_char_kinds = {
    "c1": 1,
    "c4": 4,
}


def make_array_values(array, kind=None):
    if kind is not None:
        sep = f"_{kind}"
    else:
        sep = ""

    return ", ".join([f"{i}{sep}" for i in np.asfortranarray(array).flatten()])


def write_lines(file, lines, indent_level):
    for line in lines:
        write_line(file, line, indent_level)


def write_line(file, line, indent_level):
    try:
        if line.startswith("\n"):
            line = line[1:]
    except AttributeError:
        pass
    print(f"{indent_level*INDENT}{line} \n", file=file)
